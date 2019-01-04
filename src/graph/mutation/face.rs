use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Mul};

use crate::geometry::convert::AsPosition;
use crate::geometry::Geometry;
use crate::graph::container::alias::OwnedCore;
use crate::graph::container::{Bind, Consistent, Core, Reborrow};
use crate::graph::geometry::alias::{ScaledFaceNormal, VertexPosition};
use crate::graph::geometry::{FaceCentroid, FaceNormal};
use crate::graph::mutation::edge::{self, EdgeBridgeCache, EdgeMutation};
use crate::graph::mutation::region::{Connectivity, Region, Singularity};
use crate::graph::mutation::{Mutate, Mutation};
use crate::graph::storage::convert::alias::*;
use crate::graph::storage::convert::AsStorage;
use crate::graph::storage::{EdgeKey, FaceKey, Storage, VertexKey};
use crate::graph::topology::{Edge, Face, Vertex};
use crate::graph::view::convert::FromKeyedSource;
use crate::graph::view::{EdgeKeyTopology, EdgeView, FaceKeyTopology, FaceView, VertexView};
use crate::graph::{GraphError, IteratorExt};

pub struct FaceMutation<G>
where
    G: Geometry,
{
    mutation: EdgeMutation<G>,
    storage: Storage<Face<G>>,
    singularities: HashMap<VertexKey, HashSet<FaceKey>>,
}

impl<G> FaceMutation<G>
where
    G: Geometry,
{
    pub fn insert_face(
        &mut self,
        vertices: &[VertexKey],
        geometry: (G::Edge, G::Face),
    ) -> Result<FaceKey, GraphError> {
        let cache = FaceInsertCache::snapshot(
            &Core::empty()
                .bind(self.as_vertex_storage())
                .bind(self.as_edge_storage())
                .bind(self.as_face_storage()),
            vertices,
            geometry,
        )?;
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
        // Insert composite edges and collect the interior edges.
        let edges = vertices
            .iter()
            .cloned()
            .perimeter()
            .map(|ab| {
                self.get_or_insert_composite_edge(ab, geometry.0.clone())
                    // TODO: Do not use `unwrap`. In the worst case, unroll the
                    //       iterator expression.
                    .unwrap()
                    .0
            })
            .collect::<Vec<_>>();
        // Insert the face.
        let face = self.storage.insert(Face::new(edges[0], geometry.1));
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
        self.connect_face_interior(&edges, face)?;
        self.connect_face_exterior(&edges, connectivity)?;
        Ok(face)
    }

    pub fn remove_face_with_cache(
        &mut self,
        cache: FaceRemoveCache<G>,
    ) -> Result<Face<G>, GraphError> {
        let FaceRemoveCache {
            abc,
            mutuals,
            edges,
            boundaries,
            ..
        } = cache;
        // Iterate over the set of vertices shared between the face and all of
        // its neighbors. These are potential singularities.
        for vertex in mutuals {
            let core = Core::empty()
                .bind(self.as_vertex_storage())
                .bind(self.as_edge_storage())
                .bind(self.as_face_storage());
            // Circulate (in order) over the neighboring faces of the potential
            // singularity, ignoring the face to be removed.  Count the number
            // of gaps, where neighboring faces do not share any edges. Because
            // a face is being ignored, exactly one gap is expected. If any
            // additional gaps exist, then removal will create a singularity.
            let vertex = VertexView::from_keyed_source((vertex, &core))
                .ok_or_else(|| GraphError::TopologyNotFound)?;
            let n = vertex
                .reachable_neighboring_faces()
                .filter(|face| face.key() != abc)
                .perimeter()
                .filter(|&(previous, next)| {
                    let exterior = previous
                        .reachable_interior_edges()
                        .flat_map(|edge| edge.into_reachable_opposite_edge())
                        .map(|edge| edge.key())
                        .collect::<HashSet<_>>();
                    let interior = next
                        .reachable_interior_edges()
                        .map(|edge| edge.key())
                        .collect::<HashSet<_>>();
                    exterior.intersection(&interior).count() == 0
                })
                .count();
            if n > 1 {
                return Err(GraphError::TopologyConflict);
            }
        }
        self.disconnect_face_interior(&edges)?;
        for ab in boundaries {
            self.remove_composite_edge(ab)?;
        }
        let face = self
            .storage
            .remove(&abc)
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        Ok(face)
    }

    // TODO: Should there be a distinction between `connect_face_to_edge` and
    //       `connect_edge_to_face`?
    pub fn connect_face_to_edge(&mut self, ab: EdgeKey, abc: FaceKey) -> Result<(), GraphError> {
        self.storage
            .get_mut(&abc)
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .edge = ab;
        Ok(())
    }

    fn connect_face_interior(
        &mut self,
        edges: &[EdgeKey],
        face: FaceKey,
    ) -> Result<(), GraphError> {
        for (ab, bc) in edges.iter().cloned().perimeter() {
            self.connect_neighboring_edges(ab, bc)?;
            self.connect_edge_to_face(ab, face)?;
        }
        Ok(())
    }

    fn disconnect_face_interior(&mut self, edges: &[EdgeKey]) -> Result<(), GraphError> {
        for ab in edges {
            self.disconnect_edge_from_face(*ab)?;
        }
        Ok(())
    }

    fn connect_face_exterior(
        &mut self,
        edges: &[EdgeKey],
        connectivity: (Connectivity, Connectivity),
    ) -> Result<(), GraphError> {
        let (incoming, outgoing) = connectivity;
        for (a, b) in edges.iter().map(|edge| edge.to_vertex_keys()) {
            let neighbors = {
                let core = Core::empty()
                    .bind(self.as_vertex_storage())
                    .bind(self.as_edge_storage());
                // Only boundary edges must be connected.
                EdgeView::from_keyed_source(((b, a).into(), &core))
                    .filter(|edge| edge.is_boundary_edge())
                    .and_then(|_| {
                        // The next edge of B-A is the outgoing edge of the
                        // destination vertex A that is also a boundary
                        // edge or, if there is no such outgoing edge, the
                        // next exterior edge of the face. The previous
                        // edge is similar.
                        let ax = outgoing[&a]
                            .iter()
                            .flat_map(|ax| EdgeView::from_keyed_source((*ax, &core)))
                            .find(|edge| edge.is_boundary_edge())
                            .or_else(|| {
                                EdgeView::from_keyed_source(((a, b).into(), &core))
                                    .and_then(|edge| edge.into_reachable_previous_edge())
                                    .and_then(|edge| edge.into_reachable_opposite_edge())
                            })
                            .map(|edge| edge.key());
                        let xb = incoming[&b]
                            .iter()
                            .flat_map(|xb| EdgeView::from_keyed_source((*xb, &core)))
                            .find(|edge| edge.is_boundary_edge())
                            .or_else(|| {
                                EdgeView::from_keyed_source(((a, b).into(), &core))
                                    .and_then(|edge| edge.into_reachable_next_edge())
                                    .and_then(|edge| edge.into_reachable_opposite_edge())
                            })
                            .map(|edge| edge.key());
                        ax.into_iter().zip(xb.into_iter()).next()
                    })
            };
            if let Some((ax, xb)) = neighbors {
                self.connect_neighboring_edges((b, a).into(), ax)?;
                self.connect_neighboring_edges(xb, (b, a).into())?;
            }
        }
        Ok(())
    }
}

impl<G> AsStorage<Face<G>> for FaceMutation<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<Face<G>> {
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
        let (vertices, edges, faces) = core.into_storage();
        FaceMutation {
            singularities: Default::default(),
            storage: faces,
            mutation: EdgeMutation::mutate(Core::empty().bind(vertices).bind(edges)),
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
            let (vertices, edges, ..) = core.into_storage();
            {
                let core = Core::empty().bind(&vertices).bind(&edges).bind(&faces);
                for (vertex, faces) in singularities {
                    // TODO: This will not detect exactly two faces joined by a single
                    //       vertex. This is technically supported, but perhaps should
                    //       be detected and rejected.
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
            Ok(Core::empty().bind(vertices).bind(edges).bind(faces))
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
    geometry: (G::Edge, G::Face),
}

impl<'a, G> FaceInsertCache<'a, G>
where
    G: Geometry,
{
    pub fn snapshot<M>(
        storage: M,
        vertices: &'a [VertexKey],
        geometry: (G::Edge, G::Face),
    ) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    {
        // Verify that the region is not already occupied by a face and collect
        // the incoming and outgoing edges for each vertex in the region.
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
    mutuals: Vec<VertexKey>,
    edges: Vec<EdgeKey>,
    boundaries: Vec<EdgeKey>,
    phantom: PhantomData<G>,
}

impl<G> FaceRemoveCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M>(storage: M, abc: FaceKey) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    {
        let face = match FaceView::from_keyed_source((abc, storage)) {
            Some(face) => face,
            _ => return Err(GraphError::TopologyNotFound),
        };
        let edges = face.interior_edges().map(|edge| edge.key()).collect();
        let boundaries = face
            .interior_edges()
            .flat_map(|edge| edge.into_boundary_edge())
            .map(|edge| edge.key())
            .collect();
        Ok(FaceRemoveCache {
            abc,
            mutuals: face.reachable_mutuals().into_iter().collect(),
            edges,
            // Find any boundary edges. Once this face is removed, such edges
            // will have no face on either side.
            boundaries,
            phantom: PhantomData,
        })
    }
}

pub struct FaceTriangulateCache<G>
where
    G: FaceCentroid<Centroid = <G as Geometry>::Vertex> + Geometry,
{
    vertices: Vec<VertexKey>,
    centroid: <G as FaceCentroid>::Centroid,
    geometry: G::Face,
    cache: FaceRemoveCache<G>,
}

impl<G> FaceTriangulateCache<G>
where
    G: FaceCentroid<Centroid = <G as Geometry>::Vertex> + Geometry,
{
    pub fn snapshot<M>(storage: M, abc: FaceKey) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    {
        let storage = storage.reborrow();
        let face = match FaceView::from_keyed_source((abc, storage)) {
            Some(face) => face,
            _ => return Err(GraphError::TopologyNotFound),
        };
        let vertices = face.vertices().map(|vertex| vertex.key()).collect();
        Ok(FaceTriangulateCache {
            vertices,
            centroid: face.centroid()?,
            geometry: face.geometry.clone(),
            cache: FaceRemoveCache::snapshot(storage, abc)?,
        })
    }
}

pub struct FaceBridgeCache<G>
where
    G: Geometry,
{
    sources: Vec<(EdgeKeyTopology, G::Edge)>,
    destination: FaceKeyTopology,
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
        M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    {
        let storage = storage.reborrow();
        let cache = (
            FaceRemoveCache::snapshot(storage, source)?,
            FaceRemoveCache::snapshot(storage, destination)?,
        );
        // Ensure that the opposite face exists and has the same arity.
        let source =
            FaceView::from_keyed_source((source, storage)).ok_or(GraphError::TopologyNotFound)?;
        let destination = FaceView::from_keyed_source((destination, storage))
            .ok_or(GraphError::TopologyNotFound)?;
        if source.arity() != destination.arity() {
            return Err(GraphError::ArityNonConstant);
        }
        Ok(FaceBridgeCache {
            sources: source
                .to_key_topology()
                .interior_edges()
                .iter()
                .map(|topology| {
                    (
                        topology.clone(),
                        EdgeView::from_keyed_source((topology.key(), storage))
                            .unwrap()
                            .geometry
                            .clone(),
                    )
                })
                .collect::<Vec<_>>(),
            destination: destination.to_key_topology(),
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
        M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
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

pub fn triangulate_with_cache<M, N, G>(
    mut mutation: N,
    cache: FaceTriangulateCache<G>,
) -> Result<Option<VertexKey>, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: FaceCentroid<Centroid = <G as Geometry>::Vertex> + Geometry,
{
    let FaceTriangulateCache {
        vertices,
        centroid,
        geometry,
        cache,
    } = cache;
    if vertices.len() <= 3 {
        return Ok(None);
    }
    mutation.as_mut().remove_face_with_cache(cache)?;
    let c = mutation.as_mut().insert_vertex(centroid);
    for (a, b) in vertices.into_iter().perimeter() {
        mutation
            .as_mut()
            .insert_face(&[a, b, c], (Default::default(), geometry.clone()))?;
    }
    Ok(Some(c))
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
        sources,
        destination,
        cache,
    } = cache;
    // Remove the source and destination faces. Pair the topology with edge
    // geometry for the source face.
    mutation.as_mut().remove_face_with_cache(cache.0)?;
    mutation.as_mut().remove_face_with_cache(cache.1)?;
    // TODO: Is it always correct to reverse the order of the opposite
    //       face's edges?
    // Re-insert the edges of the faces and bridge the mutual edges.
    for (source, destination) in sources
        .into_iter()
        .zip(destination.interior_edges().iter().rev())
    {
        let (a, b) = source.0.vertices();
        let (c, d) = destination.vertices();
        let ab = mutation.as_mut().insert_edge((a, b), source.1.clone())?;
        let cd = mutation.as_mut().insert_edge((c, d), source.1)?;
        let cache = EdgeBridgeCache::snapshot(mutation.as_mut(), ab, cd)?;
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
    mutation.as_mut().remove_face_with_cache(cache)?;
    let destinations = destinations
        .into_iter()
        .map(|vertex| mutation.as_mut().insert_vertex(vertex))
        .collect::<Vec<_>>();
    // Use the keys for the existing vertices and the translated geometries
    // to construct the extruded face and its connective faces.
    let extrusion = mutation
        .as_mut()
        .insert_face(&destinations, (Default::default(), geometry))?;
    for ((a, c), (b, d)) in sources
        .into_iter()
        .zip(destinations.into_iter())
        .perimeter()
    {
        // TODO: Split these faces to form triangles.
        mutation
            .as_mut()
            .insert_face(&[a, b, d, c], Default::default())?;
    }
    Ok(extrusion)
}
