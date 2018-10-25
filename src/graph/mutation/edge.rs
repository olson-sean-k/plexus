use std::ops::{Add, Deref, DerefMut, Mul};

use geometry::convert::AsPosition;
use geometry::Geometry;
use graph::container::alias::OwnedCore;
use graph::container::{Bind, Consistent, Core, Reborrow};
use graph::geometry::alias::{ScaledEdgeLateral, VertexPosition};
use graph::geometry::{EdgeLateral, EdgeMidpoint};
use graph::mutation::vertex::VertexMutation;
use graph::mutation::{Mutate, Mutation};
use graph::storage::convert::AsStorage;
use graph::storage::{EdgeKey, FaceKey, Storage, VertexKey};
use graph::topology::{Edge, Face, Vertex};
use graph::view::convert::{FromKeyedSource, IntoView};
use graph::view::EdgeView;
use graph::{GraphError, IteratorExt, ResultExt};

pub struct EdgeMutation<G>
where
    G: Geometry,
{
    mutation: VertexMutation<G>,
    storage: Storage<Edge<G>>,
}

impl<G> EdgeMutation<G>
where
    G: Geometry,
{
    pub fn as_edge_storage(&self) -> &Storage<Edge<G>> {
        self.as_storage()
    }

    pub fn insert_edge(
        &mut self,
        vertices: (VertexKey, VertexKey),
        geometry: G::Edge,
    ) -> Result<EdgeKey, GraphError> {
        let (a, b) = vertices;
        let ab = (a, b).into();
        let ba = (b, a).into();
        // If the edge already exists, then fail. This ensures an important
        // invariant: edges may only have two adjacent faces. That is, a
        // half-edge may only have one associated face, at most one preceding
        // half-edge, at most one following half-edge, and may form at most one
        // closed loop.
        if self.storage.contains_key(&ab) {
            return Err(GraphError::TopologyConflict);
        }
        if !self.mutation.as_storage().contains_key(&b) {
            return Err(GraphError::TopologyNotFound);
        }
        let mut edge = Edge::new(b, geometry);
        if let Some(opposite) = self.storage.get_mut(&ba) {
            edge.opposite = Some(ba);
            opposite.opposite = Some(ab);
        }
        self.storage.insert_with_key(&ab, edge);
        self.connect_outgoing_edge(a, ab)?;
        Ok(ab)
    }

    pub fn get_or_insert_edge(
        &mut self,
        vertices: (VertexKey, VertexKey),
        geometry: G::Edge,
    ) -> Result<EdgeKey, GraphError> {
        self.insert_edge(vertices, geometry)
            .or_if_conflict(|| Ok(vertices.into()))
    }

    pub fn get_or_insert_composite_edge(
        &mut self,
        vertices: (VertexKey, VertexKey),
        geometry: G::Edge,
    ) -> Result<(EdgeKey, EdgeKey), GraphError> {
        let (a, b) = vertices;
        let ab = self.get_or_insert_edge((a, b), geometry.clone())?;
        let ba = self.get_or_insert_edge((b, a), geometry)?;
        Ok((ab, ba))
    }

    pub fn remove_edge(&mut self, ab: EdgeKey) -> Result<Edge<G>, GraphError> {
        let (a, _) = ab.to_vertex_keys();
        let (xa, bx) = {
            self.storage
                .get(&ab)
                .ok_or_else(|| GraphError::TopologyNotFound)
                .map(|edge| (edge.previous, edge.next))
        }?;
        if let Some(xa) = xa {
            self.disconnect_next_edge(xa)?;
        }
        if let Some(bx) = bx {
            self.disconnect_previous_edge(bx)?;
        }
        self.disconnect_outgoing_edge(a)?;
        Ok(self.storage.remove(&ab).unwrap())
    }

    pub fn remove_composite_edge(&mut self, ab: EdgeKey) -> Result<(Edge<G>, Edge<G>), GraphError> {
        let (a, b) = ab.to_vertex_keys();
        let edge = self.remove_edge((a, b).into())?;
        let opposite = self.remove_edge((b, a).into())?;
        Ok((edge, opposite))
    }

    pub fn connect_neighboring_edges(
        &mut self,
        ab: EdgeKey,
        bc: EdgeKey,
    ) -> Result<(), GraphError> {
        if ab.to_vertex_keys().1 == bc.to_vertex_keys().0 {
            let previous = match self.storage.get_mut(&ab) {
                Some(previous) => {
                    previous.next = Some(bc);
                    Ok(())
                }
                _ => Err(GraphError::TopologyNotFound),
            };
            let next = match self.storage.get_mut(&bc) {
                Some(next) => {
                    next.previous = Some(ab);
                    Ok(())
                }
                _ => Err(GraphError::TopologyNotFound),
            };
            previous.and(next)
        }
        else {
            Err(GraphError::TopologyMalformed)
        }
    }

    pub fn disconnect_next_edge(&mut self, ab: EdgeKey) -> Result<Option<EdgeKey>, GraphError> {
        let bx = {
            self.storage
                .get_mut(&ab)
                .ok_or_else(|| GraphError::TopologyNotFound)?
                .next
                .take()
        };
        if let Some(bx) = bx.as_ref() {
            self.storage.get_mut(bx).unwrap().previous = None;
        }
        Ok(bx)
    }

    pub fn disconnect_previous_edge(&mut self, ab: EdgeKey) -> Result<Option<EdgeKey>, GraphError> {
        let xa = {
            self.storage
                .get_mut(&ab)
                .ok_or_else(|| GraphError::TopologyNotFound)?
                .previous
                .take()
        };
        if let Some(xa) = xa.as_ref() {
            self.storage.get_mut(xa).unwrap().previous = None;
        }
        Ok(xa)
    }

    pub fn connect_edge_to_face(&mut self, ab: EdgeKey, abc: FaceKey) -> Result<(), GraphError> {
        let edge = self
            .storage
            .get_mut(&ab)
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        edge.face = Some(abc);
        Ok(())
    }

    pub fn disconnect_edge_from_face(
        &mut self,
        ab: EdgeKey,
    ) -> Result<Option<FaceKey>, GraphError> {
        let face = self
            .storage
            .get_mut(&ab)
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .face
            .take();
        Ok(face)
    }
}

impl<G> AsStorage<Edge<G>> for EdgeMutation<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<Edge<G>> {
        &self.storage
    }
}

impl<G> Mutate for EdgeMutation<G>
where
    G: Geometry,
{
    type Mutant = Core<Storage<Vertex<G>>, Storage<Edge<G>>, ()>;
    type Error = GraphError;

    fn mutate(mutant: Self::Mutant) -> Self {
        let (vertices, edges, ..) = mutant.into_storage();
        EdgeMutation {
            mutation: VertexMutation::mutate(Core::empty().bind(vertices)),
            storage: edges,
        }
    }

    fn commit(self) -> Result<Self::Mutant, Self::Error> {
        let EdgeMutation {
            mutation,
            storage: edges,
            ..
        } = self;
        mutation.commit().and_then(move |core| {
            let (vertices, ..) = core.into_storage();
            Ok(Core::empty().bind(vertices).bind(edges))
        })
    }
}

impl<G> Deref for EdgeMutation<G>
where
    G: Geometry,
{
    type Target = VertexMutation<G>;

    fn deref(&self) -> &Self::Target {
        &self.mutation
    }
}

impl<G> DerefMut for EdgeMutation<G>
where
    G: Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.mutation
    }
}

pub struct EdgeSplitCache<G>
where
    G: Geometry,
{
    ab: EdgeKey,
    ba: EdgeKey,
    midpoint: G::Vertex,
}

impl<G> EdgeSplitCache<G>
where
    G: EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
    G::Vertex: AsPosition,
{
    pub fn snapshot<M>(storage: M, ab: EdgeKey) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    {
        let (a, b) = ab.to_vertex_keys();
        let edge: EdgeView<M, G> = match (ab, storage).into_view() {
            Some(edge) => edge,
            _ => return Err(GraphError::TopologyNotFound),
        };
        let mut midpoint = edge
            .reachable_source_vertex()
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .geometry
            .clone();
        *midpoint.as_position_mut() = EdgeMidpoint::midpoint(edge)?;
        Ok(EdgeSplitCache {
            ab,
            ba: (b, a).into(),
            midpoint,
        })
    }
}

pub struct EdgeJoinCache<G>
where
    G: Geometry,
{
    source: EdgeKey,
    destination: EdgeKey,
    edge: G::Edge,
    face: G::Face,
}

impl<G> EdgeJoinCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M>(
        storage: M,
        source: EdgeKey,
        destination: EdgeKey,
    ) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    {
        let storage = storage.reborrow();
        let (a, b) = source.to_vertex_keys();
        let (c, d) = destination.to_vertex_keys();
        let source = match (
            EdgeView::from_keyed_source((source, storage)),
            EdgeView::from_keyed_source((destination, storage)),
        ) {
            (Some(source), Some(_)) => source,
            _ => return Err(GraphError::TopologyNotFound),
        };
        // At this point, we can assume the points a, b, c, and d exist in the
        // mesh. Before mutating the mesh, ensure that existing interior edges
        // are boundaries.
        for edge in [a, b, c, d]
            .into_iter()
            .cloned()
            .perimeter()
            .flat_map(|ab| EdgeView::from_keyed_source((ab.into(), storage)))
        {
            if !edge.is_boundary_edge() {
                return Err(GraphError::TopologyConflict);
            }
        }
        Ok(EdgeJoinCache {
            source: source.key(),
            destination,
            edge: source.geometry.clone(),
            face: source
                .reachable_opposite_edge()
                .and_then(|opposite| opposite.into_reachable_face())
                .map(|face| face.geometry.clone())
                .unwrap_or_else(Default::default),
        })
    }
}

pub struct EdgeExtrudeCache<G>
where
    G: Geometry,
{
    ab: EdgeKey,
    vertices: (G::Vertex, G::Vertex),
    edge: G::Edge,
}

impl<G> EdgeExtrudeCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M, T>(storage: M, ab: EdgeKey, distance: T) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
        G: Geometry + EdgeLateral,
        G::Lateral: Mul<T>,
        G::Vertex: AsPosition,
        ScaledEdgeLateral<G, T>: Clone,
        VertexPosition<G>: Add<ScaledEdgeLateral<G, T>, Output = VertexPosition<G>> + Clone,
    {
        // Get the extruded geometry.
        let (vertices, edge) = {
            let edge = match EdgeView::from_keyed_source((ab, storage)) {
                Some(edge) => edge,
                _ => return Err(GraphError::TopologyNotFound),
            };
            if !edge.is_boundary_edge() {
                return Err(GraphError::TopologyConflict.into());
            }
            let mut vertices = (
                edge.reachable_destination_vertex()
                    .ok_or_else(|| GraphError::TopologyConflict)?
                    .geometry
                    .clone(),
                edge.reachable_source_vertex()
                    .ok_or_else(|| GraphError::TopologyConflict)?
                    .geometry
                    .clone(),
            );
            let translation = edge.lateral()? * distance;
            *vertices.0.as_position_mut() = vertices.0.as_position().clone() + translation.clone();
            *vertices.1.as_position_mut() = vertices.1.as_position().clone() + translation;
            (vertices, edge.geometry.clone())
        };
        Ok(EdgeExtrudeCache { ab, vertices, edge })
    }
}

pub fn split_with_cache<M, N, G>(
    mut mutation: N,
    cache: EdgeSplitCache<G>,
) -> Result<VertexKey, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
    G::Vertex: AsPosition,
{
    fn split_at_vertex<M, N, G>(
        mut mutation: N,
        ab: EdgeKey,
        m: VertexKey,
    ) -> Result<(EdgeKey, EdgeKey), GraphError>
    where
        N: AsMut<Mutation<M, G>>,
        M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
        G: EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
        G::Vertex: AsPosition,
    {
        let mutation = mutation.as_mut();
        // Remove the edge and insert two truncated edges in its place.
        let (a, b) = ab.to_vertex_keys();
        let span = mutation.remove_edge(ab)?;
        let am = mutation.insert_edge((a, m), span.geometry.clone())?;
        let mb = mutation.insert_edge((m, b), span.geometry.clone())?;
        // Connect the new edges to each other and their leading edges.
        mutation.connect_neighboring_edges(am, mb)?;
        if let Some(xa) = span.previous {
            mutation.connect_neighboring_edges(xa, am)?;
        }
        if let Some(bx) = span.next {
            mutation.connect_neighboring_edges(mb, bx)?;
        }
        // Update the associated face, if any, because it may refer to the
        // removed edge.
        if let Some(abc) = span.face {
            mutation.connect_face_to_edge(am, abc)?;
            mutation.connect_edge_to_face(am, abc)?;
            mutation.connect_edge_to_face(mb, abc)?;
        }
        Ok((am, mb))
    }

    let EdgeSplitCache { ab, ba, midpoint } = cache;
    let m = mutation.as_mut().insert_vertex(midpoint);
    // Split the half-edges.
    split_at_vertex(mutation.as_mut(), ab, m)?;
    split_at_vertex(mutation.as_mut(), ba, m)?;
    Ok(m)
}

pub fn join_with_cache<M, N, G>(
    mut mutation: N,
    cache: EdgeJoinCache<G>,
) -> Result<EdgeKey, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: Geometry,
{
    let EdgeJoinCache {
        source,
        destination,
        edge,
        face,
    } = cache;
    let (a, b) = source.to_vertex_keys();
    let (c, d) = destination.to_vertex_keys();
    // TODO: Split the face to form triangles.
    mutation.as_mut().insert_face(&[a, b, c, d], (edge, face))?;
    Ok(source)
}

pub fn extrude_with_cache<M, N, G, T>(
    mut mutation: N,
    cache: EdgeExtrudeCache<G>,
) -> Result<EdgeKey, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: Geometry + EdgeLateral,
    G::Lateral: Mul<T>,
    G::Vertex: AsPosition,
    ScaledEdgeLateral<G, T>: Clone,
    VertexPosition<G>: Add<ScaledEdgeLateral<G, T>, Output = VertexPosition<G>> + Clone,
{
    let mutation = mutation.as_mut();
    let EdgeExtrudeCache { ab, vertices, edge } = cache;
    let c = mutation.insert_vertex(vertices.0);
    let d = mutation.insert_vertex(vertices.1);
    let cd = mutation.insert_edge((c, d), edge)?;
    let cache = EdgeJoinCache::snapshot(
        &Core::empty()
            .bind(mutation.as_vertex_storage())
            .bind(mutation.as_edge_storage())
            .bind(mutation.as_face_storage()),
        ab,
        cd,
    )?;
    join_with_cache(mutation, cache)
}
