use fool::and;
use std::ops::{Deref, DerefMut};

use crate::entity::borrow::Reborrow;
use crate::entity::storage::prelude::*;
use crate::entity::storage::{AsStorage, AsStorageMut, Fuse, StorageTarget};
use crate::entity::view::{Bind, ClosedView, Rebind};
use crate::graph::core::Core;
use crate::graph::data::{Data, GraphData, Parametric};
use crate::graph::edge::{Arc, ArcKey, ArcView, Edge, EdgeKey};
use crate::graph::face::{Face, FaceKey};
use crate::graph::mutation::face::{self, FaceInsertCache, FaceRemoveCache};
use crate::graph::mutation::vertex::{self, VertexMutation};
use crate::graph::mutation::{Consistent, Immediate, Mode, Mutable, Mutation};
use crate::graph::vertex::{Vertex, VertexKey, VertexView};
use crate::graph::GraphError;
use crate::transact::{Bypass, Transact};
use crate::IteratorExt as _;

pub type CompositeEdgeKey = (EdgeKey, (ArcKey, ArcKey));
pub type CompositeEdge<G> = (Edge<G>, (Arc<G>, Arc<G>));

type ModalCore<P> = Core<
    Data<<P as Mode>::Graph>,
    <P as Mode>::VertexStorage,
    <P as Mode>::ArcStorage,
    <P as Mode>::EdgeStorage,
    (),
>;
#[cfg(not(all(nightly, feature = "unstable")))]
type RefCore<'a, G> = Core<
    G,
    &'a StorageTarget<Vertex<G>>,
    &'a StorageTarget<Arc<G>>,
    &'a StorageTarget<Edge<G>>,
    (),
>;
#[cfg(all(nightly, feature = "unstable"))]
type RefCore<'a, G> = Core<
    G,
    &'a StorageTarget<'a, Vertex<G>>,
    &'a StorageTarget<'a, Arc<G>>,
    &'a StorageTarget<'a, Edge<G>>,
    (),
>;

pub struct EdgeMutation<P>
where
    P: Mode,
{
    inner: VertexMutation<P>,
    // TODO: Split this into two fields.
    storage: (P::ArcStorage, P::EdgeStorage),
}

impl<P> EdgeMutation<P>
where
    P: Mode,
{
    pub fn to_ref_core(&self) -> RefCore<Data<P::Graph>> {
        self.inner
            .to_ref_core()
            .fuse(self.storage.0.as_storage())
            .fuse(self.storage.1.as_storage())
    }

    pub fn connect_adjacent_arcs(&mut self, ab: ArcKey, bc: ArcKey) -> Result<(), GraphError> {
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
        F: FnMut(&mut Arc<Data<P::Graph>>) -> T,
    {
        let arc = self
            .storage
            .0
            .as_storage_mut()
            .get_mut(&ab)
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        Ok(f(arc))
    }
}

impl<P> AsStorage<Arc<Data<P::Graph>>> for EdgeMutation<P>
where
    P: Mode,
{
    fn as_storage(&self) -> &StorageTarget<Arc<Data<P::Graph>>> {
        self.storage.0.as_storage()
    }
}

impl<P> AsStorage<Edge<Data<P::Graph>>> for EdgeMutation<P>
where
    P: Mode,
{
    fn as_storage(&self) -> &StorageTarget<Edge<Data<P::Graph>>> {
        self.storage.1.as_storage()
    }
}

impl<M> Bypass<ModalCore<Immediate<M>>> for EdgeMutation<Immediate<M>>
where
    M: Parametric,
{
    fn bypass(self) -> Self::Commit {
        let EdgeMutation {
            inner,
            storage: (arcs, edges),
            ..
        } = self;
        inner.bypass().fuse(arcs).fuse(edges)
    }
}

// TODO: This is a hack. Replace this with delegation.
impl<P> Deref for EdgeMutation<P>
where
    P: Mode,
{
    type Target = VertexMutation<P>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<P> DerefMut for EdgeMutation<P>
where
    P: Mode,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<P> From<ModalCore<P>> for EdgeMutation<P>
where
    P: Mode,
{
    fn from(core: ModalCore<P>) -> Self {
        let (vertices, arcs, edges, ..) = core.unfuse();
        EdgeMutation {
            inner: Core::empty().fuse(vertices).into(),
            storage: (arcs, edges),
        }
    }
}

impl<M> Transact<ModalCore<Immediate<M>>> for EdgeMutation<Immediate<M>>
where
    M: Parametric,
{
    type Commit = ModalCore<Immediate<M>>;
    type Abort = ();
    type Error = GraphError;

    fn commit(self) -> Result<Self::Commit, (Self::Abort, Self::Error)> {
        let EdgeMutation {
            inner,
            storage: (arcs, edges),
            ..
        } = self;
        // In a consistent graph, all arcs must have adjacent arcs and an
        // associated edge.
        for (_, arc) in arcs.as_storage().iter() {
            if !(and!(&arc.next, &arc.previous, &arc.edge)) {
                return Err(((), GraphError::TopologyMalformed));
            }
        }
        inner.commit().map(move |core| core.fuse(arcs).fuse(edges))
    }

    fn abort(self) -> Self::Abort {}
}

struct ArcRemoveCache {
    ab: ArcKey,
    xa: Option<ArcKey>,
    bx: Option<ArcKey>,
    cache: Option<FaceRemoveCache>,
}

impl ArcRemoveCache {
    pub fn from_arc<B>(arc: ArcView<B>) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Data<B>>>
            + AsStorage<Face<Data<B>>>
            + AsStorage<Vertex<Data<B>>>
            + Consistent
            + Parametric,
    {
        // If the edge has no neighbors, then `xa` and `bx` will refer to the
        // opposite arc of `ab`. In this case, the vertices `a` and `b` should
        // have no leading arcs after the removal. The cache will have its `xa`
        // and `bx` fields set to `None` in this case.
        let ab = arc.key();
        let ba = arc.opposite_arc().key();
        let xa = arc.previous_arc().key();
        let bx = arc.next_arc().key();
        let cache = if let Some(face) = arc.face() {
            Some(FaceRemoveCache::from_face(face)?)
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
    pub fn from_arc<B>(arc: ArcView<B>) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Data<B>>>
            + AsStorage<Edge<Data<B>>>
            + AsStorage<Face<Data<B>>>
            + AsStorage<Vertex<Data<B>>>
            + Consistent
            + Parametric,
    {
        let a = arc.source_vertex().key();
        let b = arc.destination_vertex().key();
        let ab_ba = arc.edge().key();
        Ok(EdgeRemoveCache {
            a,
            b,
            ab_ba,
            arc: ArcRemoveCache::from_arc(arc.to_ref())?,
            opposite: ArcRemoveCache::from_arc(arc.into_opposite_arc())?,
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
    pub fn from_arc<B>(arc: ArcView<B>) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Data<B>>>
            + AsStorage<Edge<Data<B>>>
            + AsStorage<Vertex<Data<B>>>
            + Parametric,
    {
        let opposite = arc
            .to_ref()
            .into_reachable_opposite_arc()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let source = opposite
            .to_ref()
            .into_reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let destination = arc
            .to_ref()
            .into_reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let edge = arc
            .to_ref()
            .into_reachable_edge()
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
    pub fn from_arc<B>(arc: ArcView<B>, destination: ArcKey) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Data<B>>> + AsStorage<Vertex<Data<B>>> + Parametric,
    {
        let destination: ArcView<_> = arc
            .to_ref()
            .rebind(destination)
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let a = arc
            .to_ref()
            .into_reachable_source_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?
            .key();
        let b = arc
            .to_ref()
            .into_reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?
            .key();
        let c = destination
            .to_ref()
            .into_reachable_source_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?
            .key();
        let d = destination
            .to_ref()
            .into_reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyMalformed)?
            .key();
        // Ensure that existing interior arcs are boundaries.
        for arc in [a, b, c, d]
            .iter()
            .cloned()
            .perimeter()
            .flat_map(|ab| -> Option<ArcView<_>> { arc.to_ref().rebind(ab.into()) })
        {
            if !arc.is_boundary_arc() {
                return Err(GraphError::TopologyConflict);
            }
        }
        Ok(ArcBridgeCache { a, b, c, d })
    }

    pub fn from_storage<B>(
        storage: B,
        source: ArcKey,
        destination: ArcKey,
    ) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Data<B>>> + AsStorage<Vertex<Data<B>>> + Parametric,
    {
        ArcBridgeCache::from_arc(
            ArcView::bind(storage, source).ok_or_else(|| GraphError::TopologyNotFound)?,
            destination,
        )
    }
}

pub struct ArcExtrudeCache {
    ab: ArcKey,
}

impl ArcExtrudeCache {
    pub fn from_arc<B>(arc: ArcView<B>) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Data<B>>>
            + AsStorage<Face<Data<B>>>
            + AsStorage<Vertex<Data<B>>>
            + Consistent
            + Parametric,
    {
        if !arc.is_boundary_arc() {
            Err(GraphError::TopologyConflict)
        }
        else {
            Ok(ArcExtrudeCache { ab: arc.key() })
        }
    }
}

pub fn get_or_insert_with<N, P, F>(
    mut mutation: N,
    endpoints: (VertexKey, VertexKey),
    f: F,
) -> Result<CompositeEdgeKey, GraphError>
where
    N: AsMut<Mutation<P>>,
    P: Mode,
    P::Graph: Mutable,
    F: FnOnce() -> (
        <Data<P::Graph> as GraphData>::Edge,
        <Data<P::Graph> as GraphData>::Arc,
    ),
{
    fn get_or_insert_arc<N, P>(
        mut mutation: N,
        endpoints: (VertexKey, VertexKey),
        geometry: <Data<P::Graph> as GraphData>::Arc,
    ) -> (Option<EdgeKey>, ArcKey)
    where
        N: AsMut<Mutation<P>>,
        P: Mode,
        P::Graph: Mutable,
    {
        let (a, _) = endpoints;
        let ab = endpoints.into();
        if let Some(arc) = mutation.as_mut().storage.0.as_storage().get(&ab) {
            (arc.edge, ab)
        }
        else {
            mutation
                .as_mut()
                .storage
                .0
                .as_storage_mut()
                .insert_with_key(&ab, Arc::new(geometry));
            let _ = mutation.as_mut().connect_outgoing_arc(a, ab);
            (None, ab)
        }
    }

    let geometry = f();
    let (a, b) = endpoints;
    let (e1, ab) = get_or_insert_arc(mutation.as_mut(), (a, b), geometry.1);
    let (e2, ba) = get_or_insert_arc(mutation.as_mut(), (b, a), geometry.1);
    match (e1, e2) {
        (Some(e1), Some(e2)) if e1 == e2 => Ok((e1, (ab, ba))),
        (None, None) => {
            let ab_ba = mutation
                .as_mut()
                .storage
                .1
                .as_storage_mut()
                .insert(Edge::new(ab, geometry.0));
            mutation.as_mut().connect_arc_to_edge(ab, ab_ba)?;
            mutation.as_mut().connect_arc_to_edge(ba, ab_ba)?;
            Ok((ab_ba, (ab, ba)))
        }
        // It should not be possible to insert or remove individual arcs and
        // mutations should not allow arcs to be assigned to edges
        // independently of their opposite arcs.
        _ => Err(GraphError::TopologyMalformed),
    }
}

// TODO: Removing arcs must also remove disjoint vertices. More importantly, the
//       leading arc of vertices may be invalidated by this operation and must
//       be healed. This code does not handle these cases, and so can become
//       inconsistent.
pub fn remove<N, P>(
    mut mutation: N,
    cache: EdgeRemoveCache,
) -> Result<CompositeEdge<Data<P::Graph>>, GraphError>
where
    N: AsMut<Mutation<P>>,
    P: Mode,
    P::Graph: Mutable,
{
    fn remove_arc<N, P>(
        mut mutation: N,
        cache: ArcRemoveCache,
    ) -> Result<Arc<Data<P::Graph>>, GraphError>
    where
        N: AsMut<Mutation<P>>,
        P: Mode,
        P::Graph: Mutable,
    {
        let ArcRemoveCache { ab, cache, .. } = cache;
        if let Some(cache) = cache {
            face::remove(mutation.as_mut(), cache)?;
        }
        mutation
            .as_mut()
            .storage
            .0
            .as_storage_mut()
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
        mutation.as_mut().connect_adjacent_arcs(xa, ax)?;
    }
    if let (Some(xb), Some(bx)) = (opposite.xa, arc.bx) {
        mutation.as_mut().connect_adjacent_arcs(xb, bx)?;
    }
    let edge = mutation
        .as_mut()
        .storage
        .1
        .as_storage_mut()
        .remove(&ab_ba)
        .ok_or_else(|| GraphError::TopologyNotFound)?;
    Ok((
        edge,
        (
            remove_arc(mutation.as_mut(), arc)?,
            remove_arc(mutation.as_mut(), opposite)?,
        ),
    ))
}

pub fn split_with<N, P, F>(
    mut mutation: N,
    cache: EdgeSplitCache,
    f: F,
) -> Result<VertexKey, GraphError>
where
    N: AsMut<Mutation<P>>,
    P: Mode,
    P::Graph: Mutable,
    F: FnOnce() -> <Data<P::Graph> as GraphData>::Vertex,
{
    fn remove<N, P>(mut mutation: N, ab: ArcKey) -> Result<Arc<Data<P::Graph>>, GraphError>
    where
        N: AsMut<Mutation<P>>,
        P: Mode,
        P::Graph: Mutable,
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
        let mut arc = mutation
            .as_mut()
            .storage
            .0
            .as_storage_mut()
            .remove(&ab)
            .unwrap();
        // Restore the connectivity of the arc. The mutations will clear this
        // data, because it is still a part of the mesh at that point.
        arc.previous = xa;
        arc.next = bx;
        Ok(arc)
    }

    fn split_at_vertex<N, P>(
        mut mutation: N,
        a: VertexKey,
        b: VertexKey,
        m: VertexKey,
        ab: ArcKey,
    ) -> Result<(ArcKey, ArcKey), GraphError>
    where
        N: AsMut<Mutation<P>>,
        P: Mode,
        P::Graph: Mutable,
    {
        // Remove the arc and insert two truncated arcs in its place.
        let Arc {
            next,
            previous,
            face,
            data: geometry,
            ..
        } = remove(mutation.as_mut(), ab)?;
        let am = get_or_insert_with(mutation.as_mut(), (a, m), || (Default::default(), geometry))
            .map(|(_, (am, _))| am)?;
        let mb = get_or_insert_with(mutation.as_mut(), (m, b), || (Default::default(), geometry))
            .map(|(_, (mb, _))| mb)?;
        // Connect the new arcs to each other and their leading arcs.
        mutation.as_mut().connect_adjacent_arcs(am, mb)?;
        if let Some(xa) = previous {
            mutation.as_mut().connect_adjacent_arcs(xa, am)?;
        }
        if let Some(bx) = next {
            mutation.as_mut().connect_adjacent_arcs(mb, bx)?;
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
    let m = vertex::insert(mutation.as_mut(), f());
    // Remove the edge.
    let _ = mutation
        .as_mut()
        .storage
        .1
        .as_storage_mut()
        .remove(&ab_ba)
        .ok_or_else(|| GraphError::TopologyMalformed)?;
    // Split the arcs.
    split_at_vertex(mutation.as_mut(), a, b, m, ab)?;
    split_at_vertex(mutation.as_mut(), b, a, m, ba)?;
    Ok(m)
}

pub fn bridge<N, P>(mut mutation: N, cache: ArcBridgeCache) -> Result<FaceKey, GraphError>
where
    N: AsMut<Mutation<P>>,
    P: Mode,
    P::Graph: Mutable,
{
    let ArcBridgeCache { a, b, c, d } = cache;
    let cache = FaceInsertCache::from_storage(mutation.as_mut(), &[a, b, c, d])?;
    face::insert_with(mutation.as_mut(), cache, Default::default)
}

// The identifiers `a`, `b`, `c`, and `d` are probably well understood in this
// context and `f` is used in a manner that is consistent with the standard
// library.
#[allow(clippy::many_single_char_names)]
pub fn extrude_with<N, P, F>(
    mut mutation: N,
    cache: ArcExtrudeCache,
    f: F,
) -> Result<ArcKey, GraphError>
where
    N: AsMut<Mutation<P>>,
    P: Mode,
    P::Graph: Mutable,
    F: Fn(<Data<P::Graph> as GraphData>::Vertex) -> <Data<P::Graph> as GraphData>::Vertex,
{
    let ArcExtrudeCache { ab } = cache;
    let (c, d) = {
        let (a, b) = ab.into();
        let c = VertexView::bind(mutation.as_mut(), b)
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .data;
        let d = VertexView::bind(mutation.as_mut(), a)
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .data;
        (f(c), f(d))
    };
    let c = vertex::insert(mutation.as_mut(), c);
    let d = vertex::insert(mutation.as_mut(), d);
    let cd =
        get_or_insert_with(mutation.as_mut(), (c, d), Default::default).map(|(_, (cd, _))| cd)?;
    let cache = ArcBridgeCache::from_storage(mutation.as_mut(), ab, cd)?;
    bridge(mutation, cache).map(|_| cd)
}
