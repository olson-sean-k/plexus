use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Mul};

use crate::geometry::alias::{ScaledFaceNormal, VertexPosition};
use crate::geometry::convert::AsPosition;
use crate::geometry::Geometry;
use crate::graph::container::alias::OwnedCore;
use crate::graph::container::{Bind, Consistent, Core, Reborrow};
use crate::graph::geometry::FaceNormal;
use crate::graph::mutation::edge::{self, ArcBridgeCache, EdgeMutation};
use crate::graph::mutation::region::{Connectivity, Region, Singularity};
use crate::graph::mutation::{Mutate, Mutation};
use crate::graph::payload::{ArcPayload, EdgePayload, FacePayload, VertexPayload};
use crate::graph::storage::convert::alias::*;
use crate::graph::storage::convert::AsStorage;
use crate::graph::storage::{ArcKey, FaceKey, Storage, VertexKey};
use crate::graph::view::convert::FromKeyedSource;
use crate::graph::view::{ArcView, FaceNeighborhood, FaceView, VertexView};
use crate::graph::GraphError;
use crate::IteratorExt;

pub struct FaceMutation<G>
where
    G: Geometry,
{
    mutation: EdgeMutation<G>,
    storage: Storage<FacePayload<G>>,
    singularities: HashMap<VertexKey, HashSet<FaceKey>>,
}

impl<G> FaceMutation<G>
where
    G: Geometry,
{
    fn core(
        &self,
    ) -> Core<
        &Storage<VertexPayload<G>>,
        &Storage<ArcPayload<G>>,
        &Storage<EdgePayload<G>>,
        &Storage<FacePayload<G>>,
    > {
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
            singularity,
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
        // If a singularity was detected, record it and its neighboring faces.
        if let Some(singularity) = singularity {
            let faces = self
                .singularities
                .entry(singularity.0)
                .or_insert_with(Default::default);
            for face in singularity.1 {
                faces.insert(face);
            }
            faces.insert(face);
        }
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
        connectivity: (Connectivity, Connectivity),
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
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<FacePayload<G>> {
        &self.storage
    }
}

impl<G> Mutate for FaceMutation<G>
where
    G: Geometry,
{
    type Mutant = OwnedCore<G>;
    type Error = GraphError;

    fn mutate(core: Self::Mutant) -> Self {
        // TODO: Include edges.
        let (vertices, arcs, edges, faces) = core.into_storage();
        FaceMutation {
            singularities: Default::default(),
            storage: faces,
            mutation: EdgeMutation::mutate(Core::empty().bind(vertices).bind(arcs).bind(edges)),
        }
    }

    fn commit(self) -> Result<Self::Mutant, Self::Error> {
        let FaceMutation {
            mutation,
            storage: faces,
            singularities,
            ..
        } = self;
        mutation.commit().and_then(move |core| {
            let (vertices, arcs, edges, ..) = core.into_storage();
            {
                // TODO: Rejection of pinwheel connectivity has been removed.
                //       Determine if this check is related in a way that is
                //       inconsistent. If so, this should probably be removed.
                let core = Core::empty().bind(&vertices).bind(&arcs).bind(&faces);
                for (vertex, faces) in singularities {
                    // Determine if any unreachable faces exist in the mesh. This
                    // cannot happen if the mesh is ultimately a manifold and edge
                    // connectivity heals.
                    if let Some(vertex) = VertexView::from_keyed_source((vertex, &core)) {
                        for unreachable in faces.difference(
                            &vertex
                                .reachable_neighboring_faces()
                                .map(|face| face.key())
                                .collect(),
                        ) {
                            if core.as_face_storage().contains_key(unreachable) {
                                // Non-manifold connectivity.
                                return Err(GraphError::TopologyMalformed);
                            }
                        }
                    }
                }
            }
            // TODO: Include edges.
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
    G: Geometry,
{
    type Target = EdgeMutation<G>;

    fn deref(&self) -> &Self::Target {
        &self.mutation
    }
}

impl<G> DerefMut for FaceMutation<G>
where
    G: Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.mutation
    }
}

pub struct FaceInsertCache<'a, G>
where
    G: Geometry,
{
    vertices: &'a [VertexKey],
    connectivity: (Connectivity, Connectivity),
    singularity: Option<Singularity>,
    geometry: (G::Arc, G::Face),
}

impl<'a, G> FaceInsertCache<'a, G>
where
    G: Geometry,
{
    pub fn snapshot<M>(
        storage: M,
        vertices: &'a [VertexKey],
        geometry: (G::Arc, G::Face),
    ) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target:
            AsStorage<ArcPayload<G>> + AsStorage<FacePayload<G>> + AsStorage<VertexPayload<G>>,
    {
        // Verify that the closed path is not already occupied by a face and
        // collect the incoming and outgoing arcs for each vertex in the
        // region.
        let region = Region::from_keyed_storage(vertices, storage)?;
        if region.face().is_some() {
            return Err(GraphError::TopologyConflict);
        }
        let (connectivity, singularity) = region.reachable_connectivity();
        Ok(FaceInsertCache {
            vertices,
            connectivity,
            singularity,
            geometry,
        })
    }
}

pub struct FaceRemoveCache<G>
where
    G: Geometry,
{
    abc: FaceKey,
    arcs: Vec<ArcKey>,
    phantom: PhantomData<G>,
}

impl<G> FaceRemoveCache<G>
where
    G: Geometry,
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

pub struct FaceBisectCache<G>
where
    G: Geometry,
{
    cache: FaceRemoveCache<G>,
    left: Vec<VertexKey>,
    right: Vec<VertexKey>,
    geometry: G::Face,
}

impl<G> FaceBisectCache<G>
where
    G: Geometry,
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
        face.interior_path()
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
        Ok(FaceBisectCache {
            cache: FaceRemoveCache::snapshot(storage, abc)?,
            left,
            right,
            geometry: face.geometry.clone(),
        })
    }
}

pub struct FacePokeCache<G>
where
    G: Geometry,
{
    vertices: Vec<VertexKey>,
    geometry: G::Vertex,
    cache: FaceRemoveCache<G>,
}

impl<G> FacePokeCache<G>
where
    G: Geometry,
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
    G: Geometry,
{
    source: FaceNeighborhood,
    destination: FaceNeighborhood,
    cache: (FaceRemoveCache<G>, FaceRemoveCache<G>),
}

impl<G> FaceBridgeCache<G>
where
    G: Geometry,
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
            return Err(GraphError::ArityNonConstant);
        }
        Ok(FaceBridgeCache {
            source: source.neighborhood(),
            destination: destination.neighborhood(),
            cache,
        })
    }
}

pub struct FaceExtrudeCache<G>
where
    G: FaceNormal + Geometry,
    G::Vertex: AsPosition,
{
    sources: Vec<VertexKey>,
    destinations: Vec<G::Vertex>,
    geometry: G::Face,
    cache: FaceRemoveCache<G>,
}

impl<G> FaceExtrudeCache<G>
where
    G: FaceNormal + Geometry,
    G::Vertex: AsPosition,
{
    pub fn snapshot<M, T>(storage: M, abc: FaceKey, distance: T) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<G>>
            + AsStorage<FacePayload<G>>
            + AsStorage<VertexPayload<G>>
            + Consistent,
        G::Normal: Mul<T>,
        ScaledFaceNormal<G, T>: Clone,
        VertexPosition<G>: Add<ScaledFaceNormal<G, T>, Output = VertexPosition<G>> + Clone,
    {
        let storage = storage.reborrow();
        let cache = FaceRemoveCache::snapshot(storage, abc)?;
        let face = FaceView::from_keyed_source((abc, storage)).unwrap();
        let translation = face.normal()? * distance;

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
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: Geometry,
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

pub fn bisect_with_cache<M, N, G>(
    mut mutation: N,
    cache: FaceBisectCache<G>,
) -> Result<ArcKey, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: Geometry,
{
    let FaceBisectCache {
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
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: Geometry,
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
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: Geometry,
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
    for (source, destination) in source
        .interior_arcs()
        .iter()
        .zip(destination.interior_arcs().iter().rev())
    {
        let ab = source.key();
        let cd = destination.key();
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
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: FaceNormal + Geometry,
    G::Vertex: AsPosition,
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
