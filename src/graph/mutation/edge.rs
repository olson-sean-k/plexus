use std::ops::{Add, Deref, DerefMut, Mul};

use crate::geometry::convert::AsPosition;
use crate::geometry::Geometry;
use crate::graph::container::alias::OwnedCore;
use crate::graph::container::{Bind, Consistent, Core, Reborrow};
use crate::graph::geometry::alias::{ScaledEdgeLateral, VertexPosition};
use crate::graph::geometry::{EdgeLateral, EdgeMidpoint};
use crate::graph::mutation::face::{self, FaceRemoveCache};
use crate::graph::mutation::vertex::VertexMutation;
use crate::graph::mutation::{Mutate, Mutation};
use crate::graph::storage::convert::alias::*;
use crate::graph::storage::convert::AsStorage;
use crate::graph::storage::{EdgeKey, FaceKey, Storage, VertexKey};
use crate::graph::topology::{Edge, Face, Vertex};
use crate::graph::view::convert::FromKeyedSource;
use crate::graph::view::EdgeView;
use crate::graph::GraphError;
use crate::IteratorExt;

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
    pub fn get_or_insert_composite_edge(
        &mut self,
        span: (VertexKey, VertexKey),
    ) -> Result<(EdgeKey, EdgeKey), GraphError> {
        self.get_or_insert_composite_edge_with(span, || Default::default())
    }

    pub fn get_or_insert_composite_edge_with<F>(
        &mut self,
        span: (VertexKey, VertexKey),
        f: F,
    ) -> Result<(EdgeKey, EdgeKey), GraphError>
    where
        F: Copy + Fn() -> G::Edge,
    {
        fn get_or_insert_edge_with<G, F>(
            mutation: &mut EdgeMutation<G>,
            span: (VertexKey, VertexKey),
            f: F,
        ) -> EdgeKey
        where
            G: Geometry,
            F: Fn() -> G::Edge,
        {
            let (a, _) = span;
            let ab = span.into();
            if mutation.storage.contains_key(&ab) {
                ab
            }
            else {
                mutation.storage.insert_with_key(&ab, Edge::new(f()));
                let _ = mutation.connect_outgoing_edge(a, ab);
                ab
            }
        }

        let (a, b) = span;
        Ok((
            get_or_insert_edge_with(self, (a, b), f),
            get_or_insert_edge_with(self, (b, a), f),
        ))
    }

    pub fn connect_neighboring_edges(
        &mut self,
        ab: EdgeKey,
        bc: EdgeKey,
    ) -> Result<(), GraphError> {
        self.storage
            .get_mut(&ab)
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .next = Some(bc);
        self.storage
            .get_mut(&bc)
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .previous = Some(ab);
        Ok(())
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
            self.storage
                .get_mut(bx)
                .ok_or_else(|| GraphError::TopologyMalformed)?
                .previous
                .take();
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
            self.storage
                .get_mut(xa)
                .ok_or_else(|| GraphError::TopologyMalformed)?
                .next
                .take();
        }
        Ok(xa)
    }

    pub fn connect_edge_to_face(&mut self, ab: EdgeKey, abc: FaceKey) -> Result<(), GraphError> {
        self.storage
            .get_mut(&ab)
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .face = Some(abc);
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

struct EdgeRemoveCache<G>
where
    G: Geometry,
{
    ab: EdgeKey,
    xa: Option<EdgeKey>,
    bx: Option<EdgeKey>,
    cache: Option<FaceRemoveCache<G>>,
}

impl<G> EdgeRemoveCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M>(storage: M, ab: EdgeKey) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    {
        let storage = storage.reborrow();
        let edge = EdgeView::from_keyed_source((ab, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        // If the composite edge has no neighbors, then `xa` and `bx` will
        // refer to the opposite edge of `ab`. In this case, the vertices `a`
        // and `b` should have no leading edges after the removal. The cache
        // will have its `xa` and `bx` fields set to `None` in this case.
        let ba = edge.opposite_edge().key();
        let xa = edge.previous_edge().key();
        let bx = edge.next_edge().key();
        let cache = if let Some(face) = edge.face() {
            Some(FaceRemoveCache::snapshot(storage, face.key())?)
        }
        else {
            None
        };
        Ok(EdgeRemoveCache {
            ab,
            xa: if xa != ba { Some(xa) } else { None },
            bx: if bx != ba { Some(bx) } else { None },
            cache,
        })
    }
}

pub struct CompositeEdgeRemoveCache<G>
where
    G: Geometry,
{
    a: VertexKey,
    b: VertexKey,
    edge: EdgeRemoveCache<G>,
    opposite: EdgeRemoveCache<G>,
}

impl<G> CompositeEdgeRemoveCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M>(storage: M, ab: EdgeKey) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    {
        let storage = storage.reborrow();
        let edge = EdgeView::from_keyed_source((ab, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let a = edge.source_vertex().key();
        let b = edge.destination_vertex().key();
        let ba = edge.opposite_edge().key();
        Ok(CompositeEdgeRemoveCache {
            a,
            b,
            edge: EdgeRemoveCache::snapshot(storage, ab)?,
            opposite: EdgeRemoveCache::snapshot(storage, ba)?,
        })
    }
}

pub struct CompositeEdgeSplitCache<G>
where
    G: Geometry,
{
    a: VertexKey,
    b: VertexKey,
    ab: EdgeKey,
    ba: EdgeKey,
    midpoint: G::Vertex,
}

impl<G> CompositeEdgeSplitCache<G>
where
    G: EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
    G::Vertex: AsPosition,
{
    pub fn snapshot<M>(storage: M, ab: EdgeKey) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    {
        let storage = storage.reborrow();
        let edge = EdgeView::from_keyed_source((ab, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let opposite = edge
            .reachable_opposite_edge()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let source = opposite
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let destination = edge
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let mut midpoint = source.geometry.clone();
        *midpoint.as_position_mut() = EdgeMidpoint::midpoint(edge)?;
        Ok(CompositeEdgeSplitCache {
            a: source.key(),
            b: destination.key(),
            ab,
            ba: opposite.key(),
            midpoint,
        })
    }
}

pub struct EdgeBridgeCache<G>
where
    G: Geometry,
{
    a: VertexKey,
    b: VertexKey,
    c: VertexKey,
    d: VertexKey,
    edge: G::Edge,
    face: G::Face,
}

impl<G> EdgeBridgeCache<G>
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
        let source = EdgeView::from_keyed_source((source, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let destination = EdgeView::from_keyed_source((destination, storage))
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let a = source
            .reachable_source_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?
            .key();
        let b = source
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?
            .key();
        let c = destination
            .reachable_source_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?
            .key();
        let d = destination
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?
            .key();
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
        Ok(EdgeBridgeCache {
            a,
            b,
            c,
            d,
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
            let edge = EdgeView::from_keyed_source((ab, storage))
                .ok_or_else(|| GraphError::TopologyNotFound)?;
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

pub fn remove_composite_with_cache<M, N, G>(
    mut mutation: N,
    cache: CompositeEdgeRemoveCache<G>,
) -> Result<(Edge<G>, Edge<G>), GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: Geometry,
{
    fn remove<M, N, G>(mut mutation: N, cache: EdgeRemoveCache<G>) -> Result<Edge<G>, GraphError>
    where
        N: AsMut<Mutation<M, G>>,
        M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
        G: Geometry,
    {
        let EdgeRemoveCache { ab, cache, .. } = cache;
        if let Some(cache) = cache {
            face::remove_with_cache(mutation.as_mut(), cache)?;
        }
        mutation
            .as_mut()
            .storage
            .remove(&ab)
            .ok_or_else(|| GraphError::TopologyNotFound)
    }

    let CompositeEdgeRemoveCache {
        a,
        b,
        edge,
        opposite,
        ..
    } = cache;
    // Connect each vertex to a remaining outgoing edge.
    if let Some(ax) = opposite.bx {
        mutation.as_mut().connect_outgoing_edge(a, ax)?;
    }
    if let Some(bx) = edge.bx {
        mutation.as_mut().connect_outgoing_edge(b, bx)?;
    }
    // Connect previous and next edges across the composite edge to be removed.
    if let (Some(xa), Some(ax)) = (edge.xa, opposite.bx) {
        mutation.as_mut().connect_neighboring_edges(xa, ax)?;
    }
    if let (Some(xb), Some(bx)) = (opposite.xa, edge.bx) {
        mutation.as_mut().connect_neighboring_edges(xb, bx)?;
    }
    Ok((
        remove(mutation.as_mut(), edge)?,
        remove(mutation.as_mut(), opposite)?,
    ))
}

pub fn split_composite_with_cache<M, N, G>(
    mut mutation: N,
    cache: CompositeEdgeSplitCache<G>,
) -> Result<VertexKey, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: Geometry,
{
    fn remove<M, N, G>(mut mutation: N, ab: EdgeKey) -> Result<Edge<G>, GraphError>
    where
        N: AsMut<Mutation<M, G>>,
        M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
        G: Geometry,
    {
        let (a, _) = ab.into();
        mutation.as_mut().disconnect_outgoing_edge(a)?;
        let xa = mutation.as_mut().disconnect_previous_edge(ab)?;
        let bx = mutation.as_mut().disconnect_next_edge(ab)?;
        let mut edge = mutation.as_mut().storage.remove(&ab).unwrap();
        // Restore the connectivity of the edge. The mutations will clear this
        // data, because it is still a part of the mesh at that point.
        edge.previous = xa;
        edge.next = bx;
        Ok(edge)
    }

    fn split_at_vertex<M, N, G>(
        mut mutation: N,
        a: VertexKey,
        b: VertexKey,
        m: VertexKey,
        ab: EdgeKey,
    ) -> Result<(EdgeKey, EdgeKey), GraphError>
    where
        N: AsMut<Mutation<M, G>>,
        M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
        G: Geometry,
    {
        // Remove the edge and insert two truncated edges in its place.
        let edge = remove(mutation.as_mut(), ab)?;
        let am = mutation
            .as_mut()
            .get_or_insert_composite_edge_with((a, m), || edge.geometry.clone())
            .map(|(am, _)| am)?;
        let mb = mutation
            .as_mut()
            .get_or_insert_composite_edge_with((m, b), || edge.geometry.clone())
            .map(|(mb, _)| mb)?;
        // Connect the new edges to each other and their leading edges.
        mutation.as_mut().connect_neighboring_edges(am, mb)?;
        if let Some(xa) = edge.previous {
            mutation.as_mut().connect_neighboring_edges(xa, am)?;
        }
        if let Some(bx) = edge.next {
            mutation.as_mut().connect_neighboring_edges(mb, bx)?;
        }
        // Update the associated face, if any, because it may refer to the
        // removed edge.
        if let Some(abc) = edge.face {
            mutation.as_mut().connect_face_to_edge(am, abc)?;
            mutation.as_mut().connect_edge_to_face(am, abc)?;
            mutation.as_mut().connect_edge_to_face(mb, abc)?;
        }
        Ok((am, mb))
    }

    let CompositeEdgeSplitCache {
        a,
        b,
        ab,
        ba,
        midpoint,
        ..
    } = cache;
    let m = mutation.as_mut().insert_vertex(midpoint);
    // Split the half-edges.
    split_at_vertex(mutation.as_mut(), a, b, m, ab)?;
    split_at_vertex(mutation.as_mut(), b, a, m, ba)?;
    Ok(m)
}

pub fn bridge_with_cache<M, N, G>(
    mut mutation: N,
    cache: EdgeBridgeCache<G>,
) -> Result<FaceKey, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: Geometry,
{
    let EdgeBridgeCache {
        a,
        b,
        c,
        d,
        edge,
        face,
        ..
    } = cache;
    mutation.as_mut().insert_face(&[a, b, c, d], (edge, face))
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
    let EdgeExtrudeCache {
        ab, vertices, edge, ..
    } = cache;
    let mutation = mutation.as_mut();
    let c = mutation.insert_vertex(vertices.0);
    let d = mutation.insert_vertex(vertices.1);
    // TODO: If this edge already exists, then this should probably return an
    //       error.
    let cd = mutation
        .get_or_insert_composite_edge_with((c, d), || edge.clone())
        .map(|(cd, _)| cd)?;
    let cache = EdgeBridgeCache::snapshot(
        &Core::empty()
            .bind(mutation.as_vertex_storage())
            .bind(mutation.as_edge_storage())
            .bind(mutation.as_face_storage()),
        ab,
        cd,
    )?;
    bridge_with_cache(mutation, cache).map(|_| cd)
}
