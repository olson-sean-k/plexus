use fool::and;
use std::ops::{Deref, DerefMut};
use theon::space::{EuclideanSpace, Vector};
use theon::AsPosition;

use crate::graph::borrow::Reborrow;
use crate::graph::core::{Core, Fuse};
use crate::graph::geometry::{Geometric, Geometry, GraphGeometry, VertexPosition};
use crate::graph::mutation::face::{self, FaceInsertCache, FaceRemoveCache};
use crate::graph::mutation::vertex::VertexMutation;
use crate::graph::mutation::{Consistent, Mutable, Mutation};
use crate::graph::storage::key::{ArcKey, EdgeKey, FaceKey, VertexKey};
use crate::graph::storage::payload::{Arc, Edge, Face, Vertex};
use crate::graph::storage::{AsStorage, StorageProxy};
use crate::graph::view::edge::ArcView;
use crate::graph::view::{ClosedView, View};
use crate::graph::GraphError;
use crate::transact::Transact;
use crate::IteratorExt as _;

pub type CompositeEdgeKey = (EdgeKey, (ArcKey, ArcKey));
pub type CompositeEdge<G> = (Edge<G>, (Arc<G>, Arc<G>));

#[allow(clippy::type_complexity)]
type Mutant<G> = Core<G, StorageProxy<Vertex<G>>, StorageProxy<Arc<G>>, StorageProxy<Edge<G>>, ()>;

pub struct EdgeMutation<M>
where
    M: Geometric,
{
    inner: VertexMutation<M>,
    // TODO: Split this into two fields.
    #[allow(clippy::type_complexity)]
    storage: (
        StorageProxy<Arc<Geometry<M>>>,
        StorageProxy<Edge<Geometry<M>>>,
    ),
}

impl<M, G> EdgeMutation<M>
where
    M: Geometric<Geometry = G>,
    G: GraphGeometry,
{
    // TODO: Refactor this into a non-associated function.
    pub fn get_or_insert_edge_with<F>(
        &mut self,
        span: (VertexKey, VertexKey),
        f: F,
    ) -> Result<CompositeEdgeKey, GraphError>
    where
        F: FnOnce() -> (G::Edge, G::Arc),
    {
        fn get_or_insert_arc<M>(
            mutation: &mut EdgeMutation<M>,
            span: (VertexKey, VertexKey),
            geometry: <M::Geometry as GraphGeometry>::Arc,
        ) -> (Option<EdgeKey>, ArcKey)
        where
            M: Geometric,
        {
            let (a, _) = span;
            let ab = span.into();
            if let Some(arc) = mutation.storage.0.get(&ab) {
                (arc.edge, ab)
            }
            else {
                mutation.storage.0.insert_with_key(ab, Arc::new(geometry));
                let _ = mutation.connect_outgoing_arc(a, ab);
                (None, ab)
            }
        }

        let geometry = f();
        let (a, b) = span;
        let (e1, ab) = get_or_insert_arc(self, (a, b), geometry.1);
        let (e2, ba) = get_or_insert_arc(self, (b, a), geometry.1);
        match (e1, e2) {
            (Some(e1), Some(e2)) if e1 == e2 => Ok((e1, (ab, ba))),
            (None, None) => {
                let ab_ba = self.storage.1.insert(Edge::new(ab, geometry.0));
                self.connect_arc_to_edge(ab, ab_ba)?;
                self.connect_arc_to_edge(ba, ab_ba)?;
                Ok((ab_ba, (ab, ba)))
            }
            // It should not be possible to insert or remove individual arcs and
            // mutations should not allow arcs to be assigned to edges
            // independently of their opposite arcs.
            _ => Err(GraphError::TopologyMalformed),
        }
    }

    pub fn connect_neighboring_arcs(&mut self, ab: ArcKey, bc: ArcKey) -> Result<(), GraphError> {
        self.with_arc_mut(ab, |arc| arc.next = Some(bc))?;
        self.with_arc_mut(bc, |arc| arc.previous = Some(ab))?;
        Ok(())
    }

    pub fn disconnect_next_arc(&mut self, ab: ArcKey) -> Result<Option<ArcKey>, GraphError> {
        let bx = self.with_arc_mut(ab, |arc| arc.next.take())?;
        if let Some(bx) = bx.as_ref() {
            self.with_arc_mut(*bx, |arc| arc.previous.take())
                .map_err(|_| GraphError::TopologyMalformed)?;
        }
        Ok(bx)
    }

    pub fn disconnect_previous_arc(&mut self, ab: ArcKey) -> Result<Option<ArcKey>, GraphError> {
        let xa = self.with_arc_mut(ab, |arc| arc.previous.take())?;
        if let Some(xa) = xa.as_ref() {
            self.with_arc_mut(*xa, |arc| arc.next.take())
                .map_err(|_| GraphError::TopologyMalformed)?;
        }
        Ok(xa)
    }

    pub fn connect_arc_to_edge(&mut self, ab: ArcKey, ab_ba: EdgeKey) -> Result<(), GraphError> {
        self.with_arc_mut(ab, |arc| arc.edge = Some(ab_ba))
    }

    pub fn connect_arc_to_face(&mut self, ab: ArcKey, abc: FaceKey) -> Result<(), GraphError> {
        self.with_arc_mut(ab, |arc| arc.face = Some(abc))
    }

    pub fn disconnect_arc_from_face(&mut self, ab: ArcKey) -> Result<Option<FaceKey>, GraphError> {
        self.with_arc_mut(ab, |arc| arc.face.take())
    }

    fn with_arc_mut<T, F>(&mut self, ab: ArcKey, mut f: F) -> Result<T, GraphError>
    where
        F: FnMut(&mut Arc<G>) -> T,
    {
        let arc = self
            .storage
            .0
            .get_mut(&ab)
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        Ok(f(arc))
    }
}

impl<M, G> AsStorage<Arc<G>> for EdgeMutation<M>
where
    M: Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn as_storage(&self) -> &StorageProxy<Arc<G>> {
        &self.storage.0
    }
}

impl<M, G> AsStorage<Edge<G>> for EdgeMutation<M>
where
    M: Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn as_storage(&self) -> &StorageProxy<Edge<G>> {
        &self.storage.1
    }
}

// TODO: This is a hack. Replace this with delegation.
impl<M> Deref for EdgeMutation<M>
where
    M: Geometric,
{
    type Target = VertexMutation<M>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<M> DerefMut for EdgeMutation<M>
where
    M: Geometric,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<M, G> From<Mutant<G>> for EdgeMutation<M>
where
    M: Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(core: Mutant<G>) -> Self {
        let (vertices, arcs, edges, ..) = core.unfuse();
        EdgeMutation {
            inner: Core::empty().fuse(vertices).into(),
            storage: (arcs, edges),
        }
    }
}

impl<M, G> Transact<Mutant<G>> for EdgeMutation<M>
where
    M: Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Output = Mutant<G>;
    type Error = GraphError;

    fn commit(self) -> Result<Self::Output, Self::Error> {
        let EdgeMutation {
            inner,
            storage: (arcs, edges),
            ..
        } = self;
        // In a consistent graph, all arcs must have neighboring arcs and an
        // associated edge.
        for (_, arc) in arcs.iter() {
            if !(and!(&arc.next, &arc.previous, &arc.edge)) {
                return Err(GraphError::TopologyMalformed);
            }
        }
        inner.commit().map(move |core| core.fuse(arcs).fuse(edges))
    }
}

struct ArcRemoveCache {
    ab: ArcKey,
    xa: Option<ArcKey>,
    bx: Option<ArcKey>,
    cache: Option<FaceRemoveCache>,
}

impl ArcRemoveCache {
    pub fn snapshot<B>(storage: B, ab: ArcKey) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Geometry<B>>>
            + AsStorage<Face<Geometry<B>>>
            + AsStorage<Vertex<Geometry<B>>>
            + Consistent
            + Geometric,
    {
        let storage = storage.reborrow();
        let arc = View::bind(storage, ab)
            .map(ArcView::from)
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        // If the edge has no neighbors, then `xa` and `bx` will refer to the
        // opposite arc of `ab`. In this case, the vertices `a` and `b` should
        // have no leading arcs after the removal. The cache will have its `xa`
        // and `bx` fields set to `None` in this case.
        let ba = arc.opposite_arc().key();
        let xa = arc.previous_arc().key();
        let bx = arc.next_arc().key();
        let cache = if let Some(face) = arc.face() {
            Some(FaceRemoveCache::snapshot(storage, face.key())?)
        }
        else {
            None
        };
        Ok(ArcRemoveCache {
            ab,
            xa: if xa != ba { Some(xa) } else { None },
            bx: if bx != ba { Some(bx) } else { None },
            cache,
        })
    }
}

pub struct EdgeRemoveCache {
    a: VertexKey,
    b: VertexKey,
    ab_ba: EdgeKey,
    arc: ArcRemoveCache,
    opposite: ArcRemoveCache,
}

impl EdgeRemoveCache {
    pub fn snapshot<B>(storage: B, ab: ArcKey) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Geometry<B>>>
            + AsStorage<Edge<Geometry<B>>>
            + AsStorage<Face<Geometry<B>>>
            + AsStorage<Vertex<Geometry<B>>>
            + Consistent
            + Geometric,
    {
        let storage = storage.reborrow();
        let arc = View::bind(storage, ab)
            .map(ArcView::from)
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let a = arc.source_vertex().key();
        let b = arc.destination_vertex().key();
        let ba = arc.opposite_arc().key();
        let ab_ba = arc.edge().key();
        Ok(EdgeRemoveCache {
            a,
            b,
            ab_ba,
            arc: ArcRemoveCache::snapshot(storage, ab)?,
            opposite: ArcRemoveCache::snapshot(storage, ba)?,
        })
    }
}

pub struct EdgeSplitCache {
    a: VertexKey,
    b: VertexKey,
    ab: ArcKey,
    ba: ArcKey,
    ab_ba: EdgeKey,
}

impl EdgeSplitCache {
    pub fn snapshot<B>(storage: B, ab: ArcKey) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Geometry<B>>>
            + AsStorage<Edge<Geometry<B>>>
            + AsStorage<Vertex<Geometry<B>>>
            + Geometric,
    {
        let storage = storage.reborrow();
        let arc = View::bind(storage, ab)
            .map(ArcView::from)
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let opposite = arc
            .reachable_opposite_arc()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let source = opposite
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let destination = arc
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let edge = arc
            .reachable_edge()
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        Ok(EdgeSplitCache {
            a: source.key(),
            b: destination.key(),
            ab: arc.key(),
            ba: opposite.key(),
            ab_ba: edge.key(),
        })
    }
}

pub struct ArcBridgeCache {
    a: VertexKey,
    b: VertexKey,
    c: VertexKey,
    d: VertexKey,
}

impl ArcBridgeCache {
    pub fn snapshot<B>(storage: B, source: ArcKey, destination: ArcKey) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Geometry<B>>>
            + AsStorage<Face<Geometry<B>>>
            + AsStorage<Vertex<Geometry<B>>>
            + Geometric,
    {
        let storage = storage.reborrow();
        let source = View::bind(storage, source)
            .map(ArcView::from)
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let destination = View::bind(storage, destination)
            .map(ArcView::from)
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
        // At this point, we can assume the vertices A, B, C, and D exist in the
        // mesh. Before mutating the mesh, ensure that existing interior arcs
        // are boundaries.
        for arc in [a, b, c, d]
            .iter()
            .cloned()
            .perimeter()
            .flat_map(|ab| View::bind(storage, ab.into()).map(ArcView::from))
        {
            if !arc.is_boundary_arc() {
                return Err(GraphError::TopologyConflict);
            }
        }
        Ok(ArcBridgeCache {
            a,
            b,
            c,
            d,
            /*arc: source.geometry,
             *face: source
             *    .reachable_opposite_arc()
             *    .and_then(|opposite| opposite.into_reachable_face())
             *    .map(|face| face.geometry)
             *    .unwrap_or_else(Default::default), */
        })
    }
}

pub struct ArcExtrudeCache {
    ab: ArcKey,
}

impl ArcExtrudeCache {
    pub fn snapshot<B>(storage: B, ab: ArcKey) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Geometry<B>>>
            + AsStorage<Face<Geometry<B>>>
            + AsStorage<Vertex<Geometry<B>>>
            + Consistent
            + Geometric,
    {
        let arc = View::bind(storage, ab)
            .map(ArcView::from)
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        if !arc.is_boundary_arc() {
            Err(GraphError::TopologyConflict)
        }
        else {
            Ok(ArcExtrudeCache { ab })
        }
    }
}

// TODO: Removing arcs must also remove disjoint vertices. More importantly, the
//       leading arc of vertices may be invalidated by this operation and must
//       be healed. This code does not handle these cases, and so can become
//       inconsistent.
pub fn remove<M, N>(
    mut mutation: N,
    cache: EdgeRemoveCache,
) -> Result<CompositeEdge<Geometry<M>>, GraphError>
where
    N: AsMut<Mutation<M>>,
    M: Mutable,
{
    fn remove_arc_with_cache<M, N>(
        mut mutation: N,
        cache: ArcRemoveCache,
    ) -> Result<Arc<Geometry<M>>, GraphError>
    where
        N: AsMut<Mutation<M>>,
        M: Mutable,
    {
        let ArcRemoveCache { ab, cache, .. } = cache;
        if let Some(cache) = cache {
            face::remove(mutation.as_mut(), cache)?;
        }
        mutation
            .as_mut()
            .storage
            .0
            .remove(&ab)
            .ok_or_else(|| GraphError::TopologyNotFound)
    }

    let EdgeRemoveCache {
        a,
        b,
        ab_ba,
        arc,
        opposite,
    } = cache;
    // Connect each vertex to a remaining outgoing edge.
    if let Some(ax) = opposite.bx {
        mutation.as_mut().connect_outgoing_arc(a, ax)?;
    }
    if let Some(bx) = arc.bx {
        mutation.as_mut().connect_outgoing_arc(b, bx)?;
    }
    // Connect previous and next arcs across the edge to be removed.
    if let (Some(xa), Some(ax)) = (arc.xa, opposite.bx) {
        mutation.as_mut().connect_neighboring_arcs(xa, ax)?;
    }
    if let (Some(xb), Some(bx)) = (opposite.xa, arc.bx) {
        mutation.as_mut().connect_neighboring_arcs(xb, bx)?;
    }
    let edge = mutation
        .as_mut()
        .storage
        .1
        .remove(&ab_ba)
        .ok_or_else(|| GraphError::TopologyNotFound)?;
    Ok((
        edge,
        (
            remove_arc_with_cache(mutation.as_mut(), arc)?,
            remove_arc_with_cache(mutation.as_mut(), opposite)?,
        ),
    ))
}

pub fn split_with<M, N, F>(
    mut mutation: N,
    cache: EdgeSplitCache,
    f: F,
) -> Result<VertexKey, GraphError>
where
    N: AsMut<Mutation<M>>,
    M: Mutable,
    F: FnOnce() -> <Geometry<M> as GraphGeometry>::Vertex,
{
    fn remove<M, N>(mut mutation: N, ab: ArcKey) -> Result<Arc<Geometry<M>>, GraphError>
    where
        N: AsMut<Mutation<M>>,
        M: Mutable,
    {
        // TODO: Is is probably more correct to disconnect the source vertex
        //       from its leading arc. However, this interacts poorly with
        //       `split_at_vertex`, and on the second pass will orphan the
        //       vertex B.
        //
        // let (a, _) = ab.into();
        // mutation.as_mut().disconnect_outgoing_arc(a)?;
        let xa = mutation.as_mut().disconnect_previous_arc(ab)?;
        let bx = mutation.as_mut().disconnect_next_arc(ab)?;
        let mut arc = mutation.as_mut().storage.0.remove(&ab).unwrap();
        // Restore the connectivity of the arc. The mutations will clear this
        // data, because it is still a part of the mesh at that point.
        arc.previous = xa;
        arc.next = bx;
        Ok(arc)
    }

    fn split_at_vertex<M, N>(
        mut mutation: N,
        a: VertexKey,
        b: VertexKey,
        m: VertexKey,
        ab: ArcKey,
    ) -> Result<(ArcKey, ArcKey), GraphError>
    where
        N: AsMut<Mutation<M>>,
        M: Mutable,
    {
        // Remove the arc and insert two truncated arcs in its place.
        let Arc {
            next,
            previous,
            face,
            geometry,
            ..
        } = remove(mutation.as_mut(), ab)?;
        let am = mutation
            .as_mut()
            .get_or_insert_edge_with((a, m), || (Default::default(), geometry))
            .map(|(_, (am, _))| am)?;
        let mb = mutation
            .as_mut()
            .get_or_insert_edge_with((m, b), || (Default::default(), geometry))
            .map(|(_, (mb, _))| mb)?;
        // Connect the new arcs to each other and their leading arcs.
        mutation.as_mut().connect_neighboring_arcs(am, mb)?;
        if let Some(xa) = previous {
            mutation.as_mut().connect_neighboring_arcs(xa, am)?;
        }
        if let Some(bx) = next {
            mutation.as_mut().connect_neighboring_arcs(mb, bx)?;
        }
        // Update the associated face, if any, because it may refer to the
        // removed arc.
        if let Some(abc) = face {
            mutation.as_mut().connect_face_to_arc(am, abc)?;
            mutation.as_mut().connect_arc_to_face(am, abc)?;
            mutation.as_mut().connect_arc_to_face(mb, abc)?;
        }
        Ok((am, mb))
    }

    let EdgeSplitCache {
        a,
        b,
        ab,
        ba,
        ab_ba,
    } = cache;
    let m = mutation.as_mut().insert_vertex(f());
    // Remove the edge.
    let _ = mutation
        .as_mut()
        .storage
        .1
        .remove(&ab_ba)
        .ok_or_else(|| GraphError::TopologyMalformed)?;
    // Split the arcs.
    split_at_vertex(mutation.as_mut(), a, b, m, ab)?;
    split_at_vertex(mutation.as_mut(), b, a, m, ba)?;
    Ok(m)
}

pub fn bridge<M, N>(mut mutation: N, cache: ArcBridgeCache) -> Result<FaceKey, GraphError>
where
    N: AsMut<Mutation<M>>,
    M: Mutable,
{
    let ArcBridgeCache { a, b, c, d } = cache;
    let cache = FaceInsertCache::snapshot(mutation.as_mut(), &[a, b, c, d])?;
    mutation.as_mut().insert_face_with(cache, Default::default)
}

pub fn extrude_with<M, N, F>(
    mut mutation: N,
    cache: ArcExtrudeCache,
    f: F,
) -> Result<ArcKey, GraphError>
where
    N: AsMut<Mutation<M>>,
    M: Mutable,
    F: FnOnce() -> Vector<VertexPosition<Geometry<M>>>,
    <Geometry<M> as GraphGeometry>::Vertex: AsPosition,
    VertexPosition<Geometry<M>>: EuclideanSpace,
{
    let ArcExtrudeCache { ab } = cache;
    let (c, d) = {
        let translation = f();
        let arc = View::bind(mutation.as_mut(), ab)
            .map(ArcView::from)
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let mut c = arc
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?
            .geometry;
        let mut d = arc
            .reachable_source_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?
            .geometry;
        c.transform(|position| *position + translation);
        d.transform(|position| *position + translation);
        (c, d)
    };
    let c = mutation.as_mut().insert_vertex(c);
    let d = mutation.as_mut().insert_vertex(d);
    // TODO: If this arc already exists, then this should probably return an
    //       error.
    let cd = mutation
        .as_mut()
        .get_or_insert_edge_with((c, d), Default::default)
        .map(|(_, (cd, _))| cd)?;
    let cache = ArcBridgeCache::snapshot(mutation.as_mut(), ab, cd)?;
    bridge(mutation, cache).map(|_| cd)
}
