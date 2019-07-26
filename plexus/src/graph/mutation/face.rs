use itertools::Itertools;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use theon::space::{EuclideanSpace, Vector};

use crate::graph::borrow::Reborrow;
use crate::graph::core::{Bind, Core, OwnedCore, RefCore};
use crate::graph::geometry::{GraphGeometry, VertexPosition};
use crate::graph::mutation::edge::{self, ArcBridgeCache, EdgeMutation};
use crate::graph::mutation::{Consistent, Mutable, Mutate, Mutation};
use crate::graph::storage::alias::*;
use crate::graph::storage::key::{ArcKey, FaceKey, VertexKey};
use crate::graph::storage::payload::{ArcPayload, FacePayload, VertexPayload};
use crate::graph::storage::{AsStorage, StorageProxy};
use crate::graph::view::edge::ArcView;
use crate::graph::view::face::FaceView;
use crate::graph::view::vertex::VertexView;
use crate::graph::view::{FromKeyedSource, IntoView};
use crate::graph::GraphError;
use crate::{AsPosition, IteratorExt};

pub struct FaceMutation<G>
where
    G: GraphGeometry,
{
    mutation: EdgeMutation<G>,
    storage: StorageProxy<FacePayload<G>>,
}

impl<G> FaceMutation<G>
where
    G: GraphGeometry,
{
    fn core(&self) -> RefCore<G> {
        Core::empty()
            .bind(self.as_vertex_storage())
            .bind(self.as_arc_storage())
            .bind(self.as_edge_storage())
            .bind(self.as_face_storage())
    }

    pub fn insert_face(
        &mut self,
        vertices: &[VertexKey],
        geometry: (G::Arc, G::Face),
    ) -> Result<FaceKey, GraphError> {
        let cache = FaceInsertCache::snapshot(&self.core(), vertices, geometry)?;
        self.insert_face_with_cache(cache)
    }

    pub fn insert_face_with_cache(
        &mut self,
        cache: FaceInsertCache<G>,
    ) -> Result<FaceKey, GraphError> {
        let FaceInsertCache {
            vertices,
            connectivity,
            geometry,
            ..
        } = cache;
        // Insert edges and collect the interior arcs.
        let arcs = vertices
            .iter()
            .cloned()
            .perimeter()
            .map(|(a, b)| {
                self.get_or_insert_edge_with((a, b), || geometry.0.clone())
                    .map(|(_, (ab, _))| ab)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Insert the face.
        let face = self.storage.insert(FacePayload::new(arcs[0], geometry.1));
        self.connect_face_interior(&arcs, face)?;
        self.connect_face_exterior(&arcs, connectivity)?;
        Ok(face)
    }

    // TODO: Should there be a distinction between `connect_face_to_edge` and
    //       `connect_edge_to_face`?
    pub fn connect_face_to_arc(&mut self, ab: ArcKey, abc: FaceKey) -> Result<(), GraphError> {
        self.storage
            .get_mut(&abc)
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .arc = ab;
        Ok(())
    }

    fn connect_face_interior(&mut self, arcs: &[ArcKey], face: FaceKey) -> Result<(), GraphError> {
        for (ab, bc) in arcs.iter().cloned().perimeter() {
            self.connect_neighboring_arcs(ab, bc)?;
            self.connect_arc_to_face(ab, face)?;
        }
        Ok(())
    }

    fn disconnect_face_interior(&mut self, arcs: &[ArcKey]) -> Result<(), GraphError> {
        for ab in arcs {
            self.disconnect_arc_from_face(*ab)?;
        }
        Ok(())
    }

    fn connect_face_exterior(
        &mut self,
        arcs: &[ArcKey],
        connectivity: (
            HashMap<VertexKey, Vec<ArcKey>>,
            HashMap<VertexKey, Vec<ArcKey>>,
        ),
    ) -> Result<(), GraphError> {
        let (incoming, outgoing) = connectivity;
        for ab in arcs {
            let (a, b) = ab.clone().into();
            let ba = ab.opposite();
            let neighbors = {
                let core = self.core();
                if ArcView::from_keyed_source((ba, &core))
                    .ok_or_else(|| GraphError::TopologyMalformed)?
                    .is_boundary_arc()
                {
                    // The next edge of B-A is the outgoing edge of the
                    // destination vertex A that is also a boundary
                    // edge or, if there is no such outgoing edge, the
                    // next exterior edge of the face. The previous
                    // edge is similar.
                    let ax = outgoing[&a]
                        .iter()
                        .flat_map(|ax| ArcView::from_keyed_source((*ax, &core)))
                        .find(|next| next.is_boundary_arc())
                        .or_else(|| {
                            ArcView::from_keyed_source((*ab, &core))
                                .and_then(|edge| edge.into_reachable_previous_arc())
                                .and_then(|previous| previous.into_reachable_opposite_arc())
                        })
                        .map(|next| next.key());
                    let xb = incoming[&b]
                        .iter()
                        .flat_map(|xb| ArcView::from_keyed_source((*xb, &core)))
                        .find(|previous| previous.is_boundary_arc())
                        .or_else(|| {
                            ArcView::from_keyed_source((*ab, &core))
                                .and_then(|edge| edge.into_reachable_next_arc())
                                .and_then(|next| next.into_reachable_opposite_arc())
                        })
                        .map(|previous| previous.key());
                    ax.into_iter().zip(xb.into_iter()).next()
                }
                else {
                    None
                }
            };
            if let Some((ax, xb)) = neighbors {
                self.connect_neighboring_arcs(ba, ax)?;
                self.connect_neighboring_arcs(xb, ba)?;
            }
        }
        Ok(())
    }
}

impl<G> AsStorage<FacePayload<G>> for FaceMutation<G>
where
    G: GraphGeometry,
{
    fn as_storage(&self) -> &StorageProxy<FacePayload<G>> {
        &self.storage
    }
}

impl<G> Mutate for FaceMutation<G>
where
    G: GraphGeometry,
{
    type Mutant = OwnedCore<G>;
    type Error = GraphError;

    fn mutate(core: Self::Mutant) -> Self {
        // TODO: Include edges.
        let (vertices, arcs, edges, faces) = core.into_storage();
        FaceMutation {
            storage: faces,
            mutation: EdgeMutation::mutate(Core::empty().bind(vertices).bind(arcs).bind(edges)),
        }
    }

    fn commit(self) -> Result<Self::Mutant, Self::Error> {
        let FaceMutation {
            mutation,
            storage: faces,
            ..
        } = self;
        mutation.commit().and_then(move |core| {
            let (vertices, arcs, edges, ..) = core.into_storage();
            Ok(Core::empty()
                .bind(vertices)
                .bind(arcs)
                .bind(edges)
                .bind(faces))
        })
    }
}

impl<G> Deref for FaceMutation<G>
where
    G: GraphGeometry,
{
    type Target = EdgeMutation<G>;

    fn deref(&self) -> &Self::Target {
        &self.mutation
    }
}

impl<G> DerefMut for FaceMutation<G>
where
    G: GraphGeometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.mutation
    }
}

pub struct FaceInsertCache<'a, G>
where
    G: GraphGeometry,
{
    vertices: &'a [VertexKey],
    connectivity: (
        HashMap<VertexKey, Vec<ArcKey>>,
        HashMap<VertexKey, Vec<ArcKey>>,
    ),
    geometry: (G::Arc, G::Face),
}

impl<'a, G> FaceInsertCache<'a, G>
where
    G: GraphGeometry,
{
    pub fn snapshot<M>(
        storage: M,
        keys: &'a [VertexKey],
        geometry: (G::Arc, G::Face),
    ) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target:
            AsStorage<ArcPayload<G>> + AsStorage<FacePayload<G>> + AsStorage<VertexPayload<G>>,
    {
        let arity = keys.len();
        let set = keys.iter().cloned().collect::<HashSet<_>>();
        if set.len() != arity {
            // Vertex keys are not unique.
            return Err(GraphError::TopologyMalformed);
        }

        let storage = storage.reborrow();
        let vertices = keys
            .iter()
            .flat_map(|key| (*key, storage).into_view())
            .collect::<SmallVec<[VertexView<_, _>; 4]>>();
        if vertices.len() != arity {
            // Vertex keys refer to nonexistent vertices.
            return Err(GraphError::TopologyNotFound);
        }
        for (previous, next) in keys
            .iter()
            .perimeter()
            .map(|(a, b)| ArcView::from_keyed_source(((*a, *b).into(), storage)))
            .perimeter()
        {
            if let Some(previous) = previous {
                if previous.face.is_some() {
                    // An interior arc is already occuppied by a face.
                    return Err(GraphError::TopologyConflict);
                }
                // Let the previous arc be AB and the next arc be BC. The
                // vertices A, B, and C lie within the implied ring in
                // order.
                //
                // If BC does not exist and AB is neighbors with some arc BX,
                // then X must not lie within the implied ring (the
                // ordered set of vertices given to this function). If X is
                // within the path, then BX must bisect the implied ring
                // (because X cannot be C).
                if next.is_none() {
                    if let Some(next) = previous.reachable_next_arc() {
                        let (_, destination) = next.key().into();
                        if set.contains(&destination) {
                            return Err(GraphError::TopologyConflict);
                        }
                    }
                }
            }
        }

        let mut incoming = HashMap::with_capacity(arity);
        let mut outgoing = HashMap::with_capacity(arity);
        for vertex in vertices {
            let key = vertex.key();
            let connectivity = vertex.reachable_connectivity();
            incoming.insert(key, connectivity.0);
            outgoing.insert(key, connectivity.1);
        }
        Ok(FaceInsertCache {
            vertices: keys,
            connectivity: (incoming, outgoing),
            geometry,
        })
    }
}

pub struct FaceRemoveCache<G>
where
    G: GraphGeometry,
{
    abc: FaceKey,
    arcs: Vec<ArcKey>,
    phantom: PhantomData<G>,
}

impl<G> FaceRemoveCache<G>
where
    G: GraphGeometry,
{
    // TODO: Should this require consistency?
    pub fn snapshot<M>(storage: M, abc: FaceKey) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<G>>
            + AsStorage<FacePayload<G>>
            + AsStorage<VertexPayload<G>>
            + Consistent,
    {
        let face = FaceView::from_keyed_source((abc, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let arcs = face.interior_arcs().map(|arc| arc.key()).collect();
        Ok(FaceRemoveCache {
            abc,
            arcs,
            phantom: PhantomData,
        })
    }
}

pub struct FaceSplitCache<G>
where
    G: GraphGeometry,
{
    cache: FaceRemoveCache<G>,
    left: Vec<VertexKey>,
    right: Vec<VertexKey>,
    geometry: G::Face,
}

impl<G> FaceSplitCache<G>
where
    G: GraphGeometry,
{
    pub fn snapshot<M>(
        storage: M,
        abc: FaceKey,
        source: VertexKey,
        destination: VertexKey,
    ) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<G>>
            + AsStorage<FacePayload<G>>
            + AsStorage<VertexPayload<G>>
            + Consistent,
    {
        let storage = storage.reborrow();
        let face = FaceView::from_keyed_source((abc, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        face.ring()
            .distance(source.into(), destination.into())
            .and_then(|distance| {
                if distance <= 1 {
                    Err(GraphError::TopologyMalformed)
                }
                else {
                    Ok(())
                }
            })?;
        let perimeter = face
            .vertices()
            .map(|vertex| vertex.key())
            .collect::<Vec<_>>()
            .into_iter()
            .cycle();
        let left = perimeter
            .clone()
            .into_iter()
            .tuple_windows()
            .skip_while(|(_, b)| *b != source)
            .take_while(|(a, _)| *a != destination)
            .map(|(_, b)| b)
            .collect::<Vec<_>>();
        let right = perimeter
            .into_iter()
            .tuple_windows()
            .skip_while(|(_, b)| *b != destination)
            .take_while(|(a, _)| *a != source)
            .map(|(_, b)| b)
            .collect::<Vec<_>>();
        Ok(FaceSplitCache {
            cache: FaceRemoveCache::snapshot(storage, abc)?,
            left,
            right,
            geometry: face.geometry.clone(),
        })
    }
}

pub struct FacePokeCache<G>
where
    G: GraphGeometry,
{
    vertices: Vec<VertexKey>,
    geometry: G::Vertex,
    cache: FaceRemoveCache<G>,
}

impl<G> FacePokeCache<G>
where
    G: GraphGeometry,
{
    pub fn snapshot<M>(storage: M, abc: FaceKey, geometry: G::Vertex) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<G>>
            + AsStorage<FacePayload<G>>
            + AsStorage<VertexPayload<G>>
            + Consistent,
    {
        let storage = storage.reborrow();
        let vertices = FaceView::from_keyed_source((abc, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .vertices()
            .map(|vertex| vertex.key())
            .collect();
        Ok(FacePokeCache {
            vertices,
            geometry,
            cache: FaceRemoveCache::snapshot(storage, abc)?,
        })
    }
}

pub struct FaceBridgeCache<G>
where
    G: GraphGeometry,
{
    source: SmallVec<[ArcKey; 4]>,
    destination: SmallVec<[ArcKey; 4]>,
    cache: (FaceRemoveCache<G>, FaceRemoveCache<G>),
}

impl<G> FaceBridgeCache<G>
where
    G: GraphGeometry,
{
    pub fn snapshot<M>(
        storage: M,
        source: FaceKey,
        destination: FaceKey,
    ) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<G>>
            + AsStorage<FacePayload<G>>
            + AsStorage<VertexPayload<G>>
            + Consistent,
    {
        let storage = storage.reborrow();
        let cache = (
            FaceRemoveCache::snapshot(storage, source)?,
            FaceRemoveCache::snapshot(storage, destination)?,
        );
        // Ensure that the opposite face exists and has the same arity.
        let source = FaceView::from_keyed_source((source, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let destination = FaceView::from_keyed_source((destination, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        if source.arity() != destination.arity() {
            return Err(GraphError::ArityNonUniform);
        }
        Ok(FaceBridgeCache {
            source: source
                .reachable_interior_arcs()
                .map(|arc| arc.key())
                .collect(),
            destination: destination
                .reachable_interior_arcs()
                .map(|arc| arc.key())
                .collect(),
            cache,
        })
    }
}

pub struct FaceExtrudeCache<G>
where
    G: GraphGeometry,
{
    sources: Vec<VertexKey>,
    destinations: Vec<G::Vertex>,
    geometry: G::Face,
    cache: FaceRemoveCache<G>,
}
impl<G> FaceExtrudeCache<G>
where
    G: GraphGeometry,
{
    pub fn snapshot<M>(
        storage: M,
        abc: FaceKey,
        translation: Vector<VertexPosition<G>>,
    ) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<G>>
            + AsStorage<FacePayload<G>>
            + AsStorage<VertexPayload<G>>
            + Consistent,
        G::Vertex: AsPosition,
        VertexPosition<G>: EuclideanSpace,
    {
        let storage = storage.reborrow();
        let cache = FaceRemoveCache::snapshot(storage, abc)?;
        let face = FaceView::from_keyed_source((abc, storage)).unwrap();

        let sources = face.vertices().map(|vertex| vertex.key()).collect();
        let destinations = face
            .vertices()
            .map(|vertex| {
                let mut geometry = vertex.geometry.clone();
                let position = geometry.as_position().clone() + translation.clone();
                *geometry.as_position_mut() = position;
                geometry
            })
            .collect();
        Ok(FaceExtrudeCache {
            sources,
            destinations,
            geometry: face.geometry.clone(),
            cache,
        })
    }
}

// TODO: Does this require a cache (or consistency)?
pub fn remove_with_cache<M, N, G>(
    mut mutation: N,
    cache: FaceRemoveCache<G>,
) -> Result<FacePayload<G>, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Mutable<G>,
    G: GraphGeometry,
{
    let FaceRemoveCache { abc, arcs, .. } = cache;
    mutation.as_mut().disconnect_face_interior(&arcs)?;
    let face = mutation
        .as_mut()
        .storage
        .remove(&abc)
        .ok_or_else(|| GraphError::TopologyNotFound)?;
    Ok(face)
}

pub fn split_with_cache<M, N, G>(
    mut mutation: N,
    cache: FaceSplitCache<G>,
) -> Result<ArcKey, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Mutable<G>,
    G: GraphGeometry,
{
    let FaceSplitCache {
        cache,
        left,
        right,
        geometry,
        ..
    } = cache;
    remove_with_cache(mutation.as_mut(), cache)?;
    mutation
        .as_mut()
        .insert_face(&left, (Default::default(), geometry.clone()))?;
    mutation
        .as_mut()
        .insert_face(&right, (Default::default(), geometry))?;
    Ok((left[0], right[0]).into())
}

pub fn poke_with_cache<M, N, G>(
    mut mutation: N,
    cache: FacePokeCache<G>,
) -> Result<VertexKey, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Mutable<G>,
    G: GraphGeometry,
{
    let FacePokeCache {
        vertices,
        geometry,
        cache,
    } = cache;
    let face = remove_with_cache(mutation.as_mut(), cache)?;
    let c = mutation.as_mut().insert_vertex(geometry);
    for (a, b) in vertices.into_iter().perimeter() {
        mutation
            .as_mut()
            .insert_face(&[a, b, c], (Default::default(), face.geometry.clone()))?;
    }
    Ok(c)
}

pub fn bridge_with_cache<M, N, G>(
    mut mutation: N,
    cache: FaceBridgeCache<G>,
) -> Result<(), GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Mutable<G>,
    G: GraphGeometry,
{
    let FaceBridgeCache {
        source,
        destination,
        cache,
    } = cache;
    // Remove the source and destination faces. Pair the topology with edge
    // geometry for the source face.
    remove_with_cache(mutation.as_mut(), cache.0)?;
    remove_with_cache(mutation.as_mut(), cache.1)?;
    // TODO: Is it always correct to reverse the order of the opposite face's
    //       arcs?
    // Re-insert the arcs of the faces and bridge the mutual arcs.
    for (ab, cd) in source.into_iter().zip(destination.into_iter().rev()) {
        let cache = ArcBridgeCache::snapshot(mutation.as_mut(), ab, cd)?;
        edge::bridge_with_cache(mutation.as_mut(), cache)?;
    }
    // TODO: Is there any reasonable topology this can return?
    Ok(())
}

pub fn extrude_with_cache<M, N, G>(
    mut mutation: N,
    cache: FaceExtrudeCache<G>,
) -> Result<FaceKey, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Mutable<G>,
    G: GraphGeometry,
{
    let FaceExtrudeCache {
        sources,
        destinations,
        geometry,
        cache,
    } = cache;
    remove_with_cache(mutation.as_mut(), cache)?;
    let destinations = destinations
        .into_iter()
        .map(|a| mutation.as_mut().insert_vertex(a))
        .collect::<Vec<_>>();
    // Use the keys for the existing vertices and the translated geometries
    // to construct the extruded face and its connective faces.
    let extrusion = mutation
        .as_mut()
        .insert_face(&destinations, (Default::default(), geometry.clone()))?;
    for ((a, c), (b, d)) in sources
        .into_iter()
        .zip(destinations.into_iter())
        .perimeter()
    {
        // TODO: Split these faces to form triangles.
        mutation
            .as_mut()
            .insert_face(&[a, b, d, c], (Default::default(), geometry.clone()))?;
    }
    Ok(extrusion)
}
