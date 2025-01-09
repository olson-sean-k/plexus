use derivative::Derivative;
use smallvec::SmallVec;
use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use theon::space::Vector;
use theon::AsPosition;

use crate::entity::borrow::{Reborrow, ReborrowInto, ReborrowMut};
use crate::entity::dijkstra;
use crate::entity::storage::prelude::*;
use crate::entity::storage::{
    AsStorage, AsStorageMut, AsStorageOf, HashStorage, IncrementalKeyer, Key,
};
use crate::entity::traverse::{Adjacency, Breadth, Depth, Trace, TraceAny, TraceFirst, Traversal};
use crate::entity::view::{ClosedView, Orphan, Rebind, Unbind, View};
use crate::entity::{Entity, Payload};
use crate::geometry::Metric;
use crate::graph::data::{Data, GraphData, Parametric};
use crate::graph::edge::{Arc, ArcKey, ArcOrphan, ArcView, Edge};
use crate::graph::face::{Face, FaceOrphan, FaceView};
use crate::graph::geometry::{VertexCentroid, VertexNormal, VertexPosition};
use crate::graph::mutation::vertex::{self, VertexRemoveCache};
use crate::graph::mutation::{self, Consistent, Immediate, Mutable};
use crate::graph::path::Path;
use crate::graph::{
    Circulator, GraphError, OptionExt as _, OrphanCirculator, ResultExt as _, ViewCirculator,
};
use crate::transact::{BypassOrCommit, Mutate};
use crate::IteratorExt as _;

type Mutation<M> = mutation::Mutation<Immediate<M>>;

/// Vertex entity.
#[derive(Derivative)]
#[derivative(Debug, Hash)]
pub struct Vertex<G>
where
    G: GraphData,
{
    /// User data.
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub(crate) data: G::Vertex,
    /// Required key into the leading arc.
    pub(crate) arc: Option<ArcKey>,
}

impl<G> Vertex<G>
where
    G: GraphData,
{
    pub fn new(data: G::Vertex) -> Self {
        Vertex { data, arc: None }
    }
}

impl<G> Entity for Vertex<G>
where
    G: GraphData,
{
    type Key = VertexKey;
    type Storage = HashStorage<Self, IncrementalKeyer>;
}

impl<G> Payload for Vertex<G>
where
    G: GraphData,
{
    type Data = G::Vertex;

    fn get(&self) -> &Self::Data {
        &self.data
    }

    fn get_mut(&mut self) -> &mut Self::Data {
        &mut self.data
    }
}

/// Vertex key.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct VertexKey(u64);

impl Key for VertexKey {
    type Inner = u64;

    fn from_inner(key: Self::Inner) -> Self {
        VertexKey(key)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

/// View of a vertex entity.
///
/// See the [`graph`] module documentation for more information about views.
///
/// [`graph`]: crate::graph
pub struct VertexView<B>
where
    B: Reborrow,
    B::Target: AsStorage<Vertex<Data<B>>> + Parametric,
{
    inner: View<B, Vertex<Data<B>>>,
}

impl<B, M> VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<Data<B>>> + Parametric,
{
    pub fn to_ref(&self) -> VertexView<&M> {
        self.inner.to_ref().into()
    }
}

impl<B, M> VertexView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorage<Vertex<Data<B>>> + Parametric,
{
    #[allow(clippy::wrong_self_convention)]
    fn to_mut_unchecked(&mut self) -> VertexView<&mut M> {
        self.inner.to_mut_unchecked().into()
    }
}

impl<'a, B, M, G> VertexView<B>
where
    B: ReborrowInto<'a, Target = M>,
    M: AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    // TODO: Relocate this documentation of `into_ref`.
    /// # Examples
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::R64;
    /// use nalgebra::Point3;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    ///
    /// type E3 = Point3<R64>;
    ///
    /// let mut graph: MeshGraph<E3> = Cube::new().polygons::<Position<E3>>().collect();
    /// let key = graph.arcs().nth(0).unwrap().key();
    /// let vertex = graph.arc_mut(key).unwrap().split_at_midpoint().into_ref();
    ///
    /// // This would not be possible without conversion into an immutable view.
    /// let _ = vertex.into_outgoing_arc().into_face().unwrap();
    /// let _ = vertex
    ///     .into_outgoing_arc()
    ///     .into_opposite_arc()
    ///     .into_face()
    ///     .unwrap();
    /// ```
    pub fn into_ref(self) -> VertexView<&'a M> {
        self.inner.into_ref().into()
    }
}

impl<B, M, G> VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    pub fn get<'a>(&'a self) -> &'a G::Vertex
    where
        G: 'a,
    {
        self.inner.get()
    }

    pub fn position<'a>(&'a self) -> &'a VertexPosition<G>
    where
        G: 'a,
        G::Vertex: AsPosition,
    {
        self.data.as_position()
    }
}

impl<B, M, G> VertexView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorageMut<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    pub fn get_mut<'a>(&'a mut self) -> &'a mut G::Vertex
    where
        G: 'a,
    {
        self.inner.get_mut()
    }
}

/// Reachable API.
impl<B, M, G> VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    pub(in crate::graph) fn into_reachable_outgoing_arc(self) -> Option<ArcView<B>> {
        let key = self.arc;
        key.and_then(|key| self.rebind(key))
    }
}

impl<B, M, G> VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    /// Converts the vertex into its outgoing (leading) arc.
    pub fn into_outgoing_arc(self) -> ArcView<B> {
        self.into_reachable_outgoing_arc().expect_consistent()
    }

    /// Gets the outgoing (leading) arc of the vertex.
    pub fn outgoing_arc(&self) -> ArcView<&M> {
        self.to_ref().into_outgoing_arc()
    }

    pub fn shortest_path(&self, key: VertexKey) -> Result<Path<'static, &M>, GraphError> {
        self.to_ref().into_shortest_path(key)
    }

    pub fn into_shortest_path(self, key: VertexKey) -> Result<Path<'static, B>, GraphError> {
        self.into_shortest_path_with(key, |_, _| 1usize)
    }

    pub fn shortest_path_with<Q, F>(
        &self,
        key: VertexKey,
        f: F,
    ) -> Result<Path<'static, &M>, GraphError>
    where
        Q: Copy + Metric,
        F: Fn(VertexView<&M>, VertexView<&M>) -> Q,
    {
        self.to_ref().into_shortest_path_with(key, f)
    }

    pub fn into_shortest_path_with<Q, F>(
        self,
        mut key: VertexKey,
        f: F,
    ) -> Result<Path<'static, B>, GraphError>
    where
        Q: Copy + Metric,
        F: Fn(VertexView<&M>, VertexView<&M>) -> Q,
    {
        let metrics = dijkstra::metrics_with(self.to_ref(), Some(key), f)?;
        let mut keys = vec![key];
        while let Some((Some(previous), _)) = metrics.get(&key) {
            key = *previous;
            keys.push(key);
        }
        if keys.len() < 2 {
            return Err(GraphError::TopologyUnreachable);
        }
        let (storage, _) = self.unbind();
        Path::bind(storage, keys.iter().rev())
    }

    /// Gets the valence of the vertex.
    ///
    /// A vertex's _valence_ is the number of adjacent vertices to which it is
    /// connected by arcs. The valence of a vertex is the same as its _degree_,
    /// which is the number of edges to which the vertex is connected.
    pub fn valence(&self) -> usize {
        self.adjacent_vertices().count()
    }

    pub fn centroid(&self) -> VertexPosition<G>
    where
        G: VertexCentroid,
        G::Vertex: AsPosition,
    {
        <G as VertexCentroid>::centroid(self.to_ref()).expect_consistent()
    }
}

impl<B, M, G> VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Parametric<Data = G>,
    G: GraphData,
{
    pub fn normal(&self) -> Result<Vector<VertexPosition<G>>, GraphError>
    where
        G: VertexNormal,
        G::Vertex: AsPosition,
    {
        <G as VertexNormal>::normal(self.to_ref())
    }
}

/// Reachable API.
impl<'a, B, M, G> VertexView<B>
where
    B: ReborrowInto<'a, Target = M>,
    M: 'a + AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    pub(in crate::graph) fn into_reachable_incoming_arcs(
        self,
    ) -> impl Clone + Iterator<Item = ArcView<&'a M>> {
        // This reachable circulator is needed for face insertions.
        ArcCirculator::<TraceAny<_>, _>::from(self.into_ref())
    }

    pub(in crate::graph) fn into_reachable_outgoing_arcs(
        self,
    ) -> impl Clone + Iterator<Item = ArcView<&'a M>> {
        // This reachable circulator is needed for face insertions.
        self.into_reachable_incoming_arcs()
            .flat_map(|arc| arc.into_reachable_opposite_arc())
    }
}

/// Reachable API.
impl<B, G> VertexView<B>
where
    B: Reborrow,
    B::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    pub(in crate::graph) fn reachable_incoming_arcs(
        &self,
    ) -> impl Clone + Iterator<Item = ArcView<&B::Target>> {
        self.to_ref().into_reachable_incoming_arcs()
    }

    pub(in crate::graph) fn reachable_outgoing_arcs(
        &self,
    ) -> impl Clone + Iterator<Item = ArcView<&B::Target>> {
        self.to_ref().into_reachable_outgoing_arcs()
    }
}

impl<'a, B, M, G> VertexView<B>
where
    B: ReborrowInto<'a, Target = M>,
    M: 'a + AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    pub fn into_adjacent_vertices(self) -> impl Clone + Iterator<Item = VertexView<&'a M>> {
        VertexCirculator::from(ArcCirculator::<TraceFirst<_>, _>::from(self.into_ref()))
    }

    pub fn into_incoming_arcs(self) -> impl Clone + Iterator<Item = ArcView<&'a M>> {
        ArcCirculator::<TraceFirst<_>, _>::from(self.into_ref())
    }

    pub fn into_outgoing_arcs(self) -> impl Clone + Iterator<Item = ArcView<&'a M>> {
        self.into_incoming_arcs().map(|arc| arc.into_opposite_arc())
    }
}

impl<B, G> VertexView<B>
where
    B: Reborrow,
    B::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    pub fn adjacent_vertices(&self) -> impl Clone + Iterator<Item = VertexView<&B::Target>> {
        self.to_ref().into_adjacent_vertices()
    }

    /// Gets an iterator of views over the incoming arcs of the vertex.
    ///
    /// The ordering of arcs is deterministic and is based on the leading arc of
    /// the vertex.
    pub fn incoming_arcs(&self) -> impl Clone + Iterator<Item = ArcView<&B::Target>> {
        self.to_ref().into_incoming_arcs()
    }

    /// Gets an iterator of views over the outgoing arcs of the vertex.
    ///
    /// The ordering of arcs is deterministic and is based on the leading arc of
    /// the vertex.
    pub fn outgoing_arcs(&self) -> impl Clone + Iterator<Item = ArcView<&B::Target>> {
        self.to_ref().into_outgoing_arcs()
    }

    /// Gets an iterator that traverses adjacent vertices by breadth.
    ///
    /// The traversal moves from the vertex to its adjacent vertices and so on.
    /// If there are disjoint subgraphs in the graph, then a traversal will not
    /// reach every vertex in the graph.
    pub fn traverse_by_breadth(&self) -> impl Clone + Iterator<Item = VertexView<&B::Target>> {
        Traversal::<_, _, Breadth>::from(self.to_ref())
    }

    /// Gets an iterator that traverses adjacent vertices by depth.
    ///
    /// The traversal moves from the vertex to its adjacent vertices and so on.
    /// If there are disjoint subgraphs in the graph, then a traversal will not
    /// reach every vertex in the graph.
    pub fn traverse_by_depth(&self) -> impl Clone + Iterator<Item = VertexView<&B::Target>> {
        Traversal::<_, _, Depth>::from(self.to_ref())
    }
}

impl<'a, B, M, G> VertexView<B>
where
    B: ReborrowInto<'a, Target = M>,
    M: 'a
        + AsStorage<Arc<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Parametric<Data = G>,
    G: GraphData,
{
    pub fn into_adjacent_faces(self) -> impl Clone + Iterator<Item = FaceView<&'a M>> {
        FaceCirculator::from(ArcCirculator::<TraceFirst<_>, _>::from(self.into_ref()))
    }
}

impl<B, G> VertexView<B>
where
    B: Reborrow,
    B::Target: AsStorage<Arc<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Parametric<Data = G>,
    G: GraphData,
{
    /// Gets an iterator of views over the adjacent faces of the vertex.
    ///
    /// The ordering of faces is deterministic and is based on the leading arc
    /// of the vertex.
    pub fn adjacent_faces(&self) -> impl Clone + Iterator<Item = FaceView<&B::Target>> {
        self.to_ref().into_adjacent_faces()
    }
}

impl<'a, M> VertexView<&'a mut M>
where
    M: AsStorage<Arc<M::Data>> + AsStorageMut<Vertex<M::Data>> + Consistent + Parametric,
{
    pub fn into_adjacent_vertex_orphans(self) -> impl Iterator<Item = VertexOrphan<'a, M::Data>> {
        VertexCirculator::from(ArcCirculator::<TraceFirst<_>, _>::from(self))
    }
}

impl<B> VertexView<B>
where
    B: ReborrowMut,
    B::Target: AsStorage<Arc<Data<B>>> + AsStorageMut<Vertex<Data<B>>> + Consistent + Parametric,
{
    pub fn adjacent_vertex_orphans(&mut self) -> impl Iterator<Item = VertexOrphan<Data<B>>> {
        self.to_mut_unchecked().into_adjacent_vertex_orphans()
    }
}

impl<'a, M> VertexView<&'a mut M>
where
    M: AsStorageMut<Arc<M::Data>> + AsStorage<Vertex<M::Data>> + Consistent + Parametric,
{
    pub fn into_incoming_arc_orphans(self) -> impl Iterator<Item = ArcOrphan<'a, M::Data>> {
        ArcCirculator::<TraceFirst<_>, _>::from(self)
    }
}

impl<B> VertexView<B>
where
    B: ReborrowMut,
    B::Target: AsStorageMut<Arc<Data<B>>> + AsStorage<Vertex<Data<B>>> + Consistent + Parametric,
{
    /// Gets an iterator of orphan views over the incoming arcs of the vertex.
    ///
    /// The ordering of arcs is deterministic and is based on the leading arc of
    /// the vertex.
    pub fn incoming_arc_orphans(&mut self) -> impl Iterator<Item = ArcOrphan<Data<B>>> {
        self.to_mut_unchecked().into_incoming_arc_orphans()
    }
}

impl<'a, M> VertexView<&'a mut M>
where
    M: AsStorage<Arc<M::Data>>
        + AsStorageMut<Face<M::Data>>
        + AsStorage<Vertex<M::Data>>
        + Consistent
        + Parametric,
{
    pub fn into_adjacent_face_orphans(self) -> impl Iterator<Item = FaceOrphan<'a, M::Data>> {
        FaceCirculator::from(ArcCirculator::<TraceFirst<_>, _>::from(self))
    }
}

impl<B> VertexView<B>
where
    B: ReborrowMut,
    B::Target: AsStorage<Arc<Data<B>>>
        + AsStorageMut<Face<Data<B>>>
        + AsStorage<Vertex<Data<B>>>
        + Consistent
        + Parametric,
{
    /// Gets an iterator of orphan views over the adjacent faces of the vertex.
    ///
    /// The ordering of faces is deterministic and is based on the leading arc
    /// of the vertex.
    pub fn adjacent_face_orphans(&mut self) -> impl Iterator<Item = FaceOrphan<Data<B>>> {
        self.to_mut_unchecked().into_adjacent_face_orphans()
    }
}

impl<M, G> VertexView<&'_ mut M>
where
    M: AsStorage<Arc<G>>
        + AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Default
        + Mutable<Data = G>,
    G: GraphData,
{
    // TODO: This is not yet implemented, so examples use `no_run`. Run these
    //       examples in doc tests once this no longer intentionally panics.
    /// Removes the vertex.
    ///
    /// Any and all dependent entities are also removed, such as arcs and edges
    /// connected to the vertex, faces connected to such arcs, vertices with no
    /// remaining leading arc, etc.
    ///
    /// Vertex removal is the most destructive removal, because vertices are a
    /// dependency of all other entities.
    ///
    /// # Examples
    ///
    /// Removing a corner from a cube by removing its vertex:
    ///
    /// ```rust,no_run
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::R64;
    /// use nalgebra::Point3;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    ///
    /// type E3 = Point3<R64>;
    ///
    /// let mut graph: MeshGraph<E3> = Cube::new().polygons::<Position<E3>>().collect();
    /// let key = graph.vertices().nth(0).unwrap().key();
    /// graph.vertex_mut(key).unwrap().remove();
    /// ```
    pub fn remove(self) {
        // This should never fail here.
        let cache = VertexRemoveCache::from_vertex(self.to_ref()).expect_consistent();
        let (storage, _) = self.unbind();
        Mutation::take(storage)
            .bypass_or_commit_with(|mutation| vertex::remove(mutation, cache))
            .map(|_| ())
            .map_err(|(_, error)| error)
            .expect_consistent()
    }
}

impl<B, M, G> Adjacency for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    type Output = SmallVec<[Self::Key; 8]>;

    fn adjacency(&self) -> Self::Output {
        self.adjacent_vertices().keys().collect()
    }
}

impl<B> Borrow<VertexKey> for VertexView<B>
where
    B: Reborrow,
    B::Target: AsStorage<Vertex<Data<B>>> + Parametric,
{
    fn borrow(&self) -> &VertexKey {
        self.inner.as_ref()
    }
}

impl<B, M, G> Clone for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
    View<B, Vertex<G>>: Clone,
{
    fn clone(&self) -> Self {
        VertexView {
            inner: self.inner.clone(),
        }
    }
}

impl<B, M, G> ClosedView for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    type Key = VertexKey;
    type Entity = Vertex<G>;

    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<B, M, G> Copy for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
    View<B, Vertex<G>>: Copy,
{
}

impl<B, M, G> Deref for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    type Target = Vertex<G>;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl<B, M, G> DerefMut for VertexView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorageMut<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

impl<B, M, G> Eq for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
}

impl<B, M, G> From<View<B, Vertex<G>>> for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn from(view: View<B, Vertex<G>>) -> Self {
        VertexView { inner: view }
    }
}

impl<B, M, G> From<VertexView<B>> for View<B, Vertex<G>>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn from(vertex: VertexView<B>) -> Self {
        let VertexView { inner, .. } = vertex;
        inner
    }
}

impl<B, M, G> Hash for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.inner.hash(state);
    }
}

impl<B, M, G> PartialEq for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Orphan view of a vertex entity.
pub struct VertexOrphan<'a, G>
where
    G: GraphData,
{
    inner: Orphan<'a, Vertex<G>>,
}

impl<G> VertexOrphan<'_, G>
where
    G: GraphData,
{
    pub fn position(&self) -> &VertexPosition<G>
    where
        G::Vertex: AsPosition,
    {
        self.inner.get().as_position()
    }
}

impl<G> VertexOrphan<'_, G>
where
    G: GraphData,
{
    pub fn get(&self) -> &G::Vertex {
        self.inner.get()
    }

    pub fn get_mut(&mut self) -> &mut G::Vertex {
        self.inner.get_mut()
    }
}

impl<G> Borrow<VertexKey> for VertexOrphan<'_, G>
where
    G: GraphData,
{
    fn borrow(&self) -> &VertexKey {
        self.inner.as_ref()
    }
}

impl<G> ClosedView for VertexOrphan<'_, G>
where
    G: GraphData,
{
    type Key = VertexKey;
    type Entity = Vertex<G>;

    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<G> Eq for VertexOrphan<'_, G> where G: GraphData {}

impl<'a, G> From<Orphan<'a, Vertex<G>>> for VertexOrphan<'a, G>
where
    G: GraphData,
{
    fn from(inner: Orphan<'a, Vertex<G>>) -> Self {
        VertexOrphan { inner }
    }
}

impl<'a, M> From<View<&'a mut M, Vertex<M::Data>>> for VertexOrphan<'a, M::Data>
where
    M: AsStorageMut<Vertex<M::Data>> + Parametric,
{
    fn from(view: View<&'a mut M, Vertex<M::Data>>) -> Self {
        VertexOrphan { inner: view.into() }
    }
}

impl<'a, M> From<VertexView<&'a mut M>> for VertexOrphan<'a, M::Data>
where
    M: AsStorageMut<Vertex<M::Data>> + Parametric,
{
    fn from(vertex: VertexView<&'a mut M>) -> Self {
        Orphan::from(vertex.inner).into()
    }
}

impl<G> Hash for VertexOrphan<'_, G>
where
    G: GraphData,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.inner.hash(state);
    }
}

impl<G> PartialEq for VertexOrphan<'_, G>
where
    G: GraphData,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

pub struct VertexCirculator<P, B>
where
    P: Trace<ArcKey>,
    B: Reborrow,
    B::Target: AsStorage<Arc<Data<B>>> + AsStorage<Vertex<Data<B>>> + Parametric,
{
    inner: ArcCirculator<P, B>,
}

impl<P, B, M, G> Circulator<B> for VertexCirculator<P, B>
where
    P: Trace<ArcKey>,
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    type Entity = Vertex<G>;

    fn next(&mut self) -> Option<<Self::Entity as Entity>::Key> {
        self.inner.next().map(|arc| {
            let (source, _) = arc.into();
            source
        })
    }
}

impl<P, B, M, G> Clone for VertexCirculator<P, B>
where
    P: Clone + Trace<ArcKey>,
    B: Clone + Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn clone(&self) -> Self {
        VertexCirculator {
            inner: self.inner.clone(),
        }
    }
}

impl<P, B, M, G> From<ArcCirculator<P, B>> for VertexCirculator<P, B>
where
    P: Trace<ArcKey>,
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn from(inner: ArcCirculator<P, B>) -> Self {
        VertexCirculator { inner }
    }
}

impl<'a, P, M> Iterator for VertexCirculator<P, &'a M>
where
    P: Trace<ArcKey>,
    M: AsStorage<Arc<M::Data>> + AsStorage<Vertex<M::Data>> + Parametric,
{
    type Item = VertexView<&'a M>;

    fn next(&mut self) -> Option<Self::Item> {
        self.bind_next_view()
    }
}

impl<'a, M> Iterator for VertexCirculator<TraceAny<ArcKey>, &'a mut M>
where
    M: AsStorage<Arc<M::Data>> + AsStorageMut<Vertex<M::Data>> + Parametric,
{
    type Item = VertexOrphan<'a, M::Data>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.bind_next_orphan() }
    }
}

impl<'a, M> Iterator for VertexCirculator<TraceFirst<ArcKey>, &'a mut M>
where
    M: AsStorage<Arc<M::Data>> + AsStorageMut<Vertex<M::Data>> + Consistent + Parametric,
{
    type Item = VertexOrphan<'a, M::Data>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.bind_next_orphan() }
    }
}

impl<'a, P, M> OrphanCirculator<'a, M> for VertexCirculator<P, &'a mut M>
where
    P: Trace<ArcKey>,
    M: AsStorage<Arc<M::Data>> + AsStorageMut<Vertex<M::Data>> + Parametric,
{
    fn target(&mut self) -> &mut M {
        self.inner.storage
    }
}

impl<'a, P, M> ViewCirculator<'a, M> for VertexCirculator<P, &'a M>
where
    P: Trace<ArcKey>,
    M: AsStorage<Arc<M::Data>> + AsStorage<Vertex<M::Data>> + Parametric,
{
    fn target(&self) -> &'a M {
        self.inner.storage
    }
}

pub struct ArcCirculator<P, B>
where
    P: Trace<ArcKey>,
    B: Reborrow,
    B::Target: AsStorage<Arc<Data<B>>> + Parametric,
{
    storage: B,
    outgoing: Option<ArcKey>,
    trace: P,
}

impl<P, B, M, G> Circulator<B> for ArcCirculator<P, B>
where
    P: Trace<ArcKey>,
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
{
    type Entity = Arc<G>;

    fn next(&mut self) -> Option<<Self::Entity as Entity>::Key> {
        self.outgoing
            .and_then(|outgoing| self.trace.insert(outgoing).then_some(outgoing))
            .map(|outgoing| outgoing.into_opposite())
            .and_then(|incoming| {
                self.storage
                    .reborrow()
                    .as_storage()
                    .get(&incoming)
                    .map(|incoming| incoming.next)
                    .map(|outgoing| (incoming, outgoing))
            })
            .map(|(incoming, outgoing)| {
                self.outgoing = outgoing;
                incoming
            })
    }
}

impl<P, B, M, G> Clone for ArcCirculator<P, B>
where
    P: Clone + Trace<ArcKey>,
    B: Clone + Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn clone(&self) -> Self {
        ArcCirculator {
            storage: self.storage.clone(),
            outgoing: self.outgoing,
            trace: self.trace.clone(),
        }
    }
}

impl<P, B, M, G> From<VertexView<B>> for ArcCirculator<P, B>
where
    P: Default + Trace<ArcKey>,
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn from(vertex: VertexView<B>) -> Self {
        let key = vertex.arc;
        let (storage, _) = vertex.unbind();
        ArcCirculator {
            storage,
            outgoing: key,
            trace: Default::default(),
        }
    }
}

impl<'a, P, M> Iterator for ArcCirculator<P, &'a M>
where
    P: Trace<ArcKey>,
    M: AsStorage<Arc<M::Data>> + Parametric,
{
    type Item = ArcView<&'a M>;

    fn next(&mut self) -> Option<Self::Item> {
        self.bind_next_view()
    }
}

impl<'a, M> Iterator for ArcCirculator<TraceAny<ArcKey>, &'a mut M>
where
    M: AsStorageMut<Arc<M::Data>> + Parametric,
{
    type Item = ArcOrphan<'a, M::Data>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.bind_next_orphan() }
    }
}

impl<'a, M> Iterator for ArcCirculator<TraceFirst<ArcKey>, &'a mut M>
where
    M: AsStorageMut<Arc<M::Data>> + Consistent + Parametric,
{
    type Item = ArcOrphan<'a, M::Data>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.bind_next_orphan() }
    }
}

impl<'a, P, M> OrphanCirculator<'a, M> for ArcCirculator<P, &'a mut M>
where
    P: Trace<ArcKey>,
    M: AsStorageMut<Arc<M::Data>> + Parametric,
{
    fn target(&mut self) -> &mut M {
        self.storage
    }
}

impl<'a, P, M> ViewCirculator<'a, M> for ArcCirculator<P, &'a M>
where
    P: Trace<ArcKey>,
    M: AsStorage<Arc<M::Data>> + Parametric,
{
    fn target(&self) -> &'a M {
        self.storage
    }
}

pub struct FaceCirculator<P, B>
where
    P: Trace<ArcKey>,
    B: Reborrow,
    B::Target: AsStorage<Arc<Data<B>>> + AsStorage<Face<Data<B>>> + Parametric,
{
    inner: ArcCirculator<P, B>,
}

impl<P, B, M, G> Circulator<B> for FaceCirculator<P, B>
where
    P: Trace<ArcKey>,
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Face<G>> + Parametric<Data = G>,
    G: GraphData,
{
    type Entity = Face<G>;

    fn next(&mut self) -> Option<<Self::Entity as Entity>::Key> {
        while let Some(arc) = self.inner.next() {
            if let Some(face) = self
                .inner
                .storage
                .reborrow()
                .as_storage_of::<Arc<_>>()
                .get(&arc)
                .and_then(|arc| arc.face)
            {
                return Some(face);
            }
            else {
                // Skip arcs with no face. This can occur within non-enclosed
                // meshes.
                continue;
            }
        }
        None
    }
}

impl<P, B, M, G> Clone for FaceCirculator<P, B>
where
    P: Clone + Trace<ArcKey>,
    B: Clone + Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Face<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn clone(&self) -> Self {
        FaceCirculator {
            inner: self.inner.clone(),
        }
    }
}

impl<P, B, M, G> From<ArcCirculator<P, B>> for FaceCirculator<P, B>
where
    P: Trace<ArcKey>,
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Face<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn from(inner: ArcCirculator<P, B>) -> Self {
        FaceCirculator { inner }
    }
}

impl<'a, P, M> Iterator for FaceCirculator<P, &'a M>
where
    P: Trace<ArcKey>,
    M: AsStorage<Arc<M::Data>> + AsStorage<Face<M::Data>> + Parametric,
{
    type Item = FaceView<&'a M>;

    fn next(&mut self) -> Option<Self::Item> {
        self.bind_next_view()
    }
}

impl<'a, M> Iterator for FaceCirculator<TraceAny<ArcKey>, &'a mut M>
where
    M: AsStorage<Arc<M::Data>> + AsStorageMut<Face<M::Data>> + Parametric,
{
    type Item = FaceOrphan<'a, M::Data>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.bind_next_orphan() }
    }
}

impl<'a, M> Iterator for FaceCirculator<TraceFirst<ArcKey>, &'a mut M>
where
    M: AsStorage<Arc<M::Data>> + AsStorageMut<Face<M::Data>> + Consistent + Parametric,
{
    type Item = FaceOrphan<'a, M::Data>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.bind_next_orphan() }
    }
}

impl<'a, P, M> OrphanCirculator<'a, M> for FaceCirculator<P, &'a mut M>
where
    P: Trace<ArcKey>,
    M: AsStorage<Arc<M::Data>> + AsStorageMut<Face<M::Data>> + Parametric,
{
    fn target(&mut self) -> &mut M {
        self.inner.storage
    }
}

impl<'a, P, M> ViewCirculator<'a, M> for FaceCirculator<P, &'a M>
where
    P: Trace<ArcKey>,
    M: AsStorage<Arc<M::Data>> + AsStorage<Face<M::Data>> + Parametric,
{
    fn target(&self) -> &'a M {
        self.inner.storage
    }
}

#[cfg(test)]
mod tests {
    use decorum::R64;
    use nalgebra::{Point2, Point3};

    use crate::graph::{GraphError, MeshGraph};
    use crate::prelude::*;
    use crate::primitive::cube::Cube;
    use crate::primitive::generate::Position;
    use crate::primitive::sphere::UvSphere;
    use crate::primitive::Trigon;

    type E3 = Point3<R64>;

    #[test]
    fn circulate_over_arcs() {
        let graph: MeshGraph<E3> = UvSphere::new(4, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect();

        // All faces should be triangles and all vertices should have 4
        // (incoming) arcs.
        for vertex in graph.vertices() {
            assert_eq!(4, vertex.incoming_arcs().count());
        }
    }

    #[test]
    fn reachable_shortest_path() {
        let graph = MeshGraph::<Point2<f64>>::from_raw_buffers(
            vec![Trigon::new(0usize, 1, 2)],
            vec![(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)],
        )
        .unwrap();
        let from = graph.vertices().nth(0).unwrap();
        let to = from.outgoing_arc().destination_vertex().key();
        let path = from.shortest_path(to).unwrap();

        assert_eq!(path.back().key(), from.key());
        assert_eq!(path.front().key(), to);
        assert_eq!(path.arcs().count(), 1);
    }

    #[test]
    fn unreachable_shortest_path() {
        // Create a graph from two disjoint trigons.
        let graph = MeshGraph::<Point2<f64>>::from_raw_buffers(
            vec![Trigon::new(0usize, 1, 2), Trigon::new(3, 4, 5)],
            vec![
                (-2.0, 0.0),
                (-1.0, 1.0),
                (-1.0, 0.0),
                (0.0, 0.0),
                (1.0, 1.0),
                (1.0, 0.0),
            ],
        )
        .unwrap();
        let mut vertices = graph.disjoint_subgraph_vertices();
        let from = vertices.next().unwrap();
        let to = vertices.next().unwrap().key();

        assert!(matches!(
            from.into_shortest_path(to),
            Err(GraphError::TopologyUnreachable)
        ));
    }

    #[test]
    fn traverse_by_breadth() {
        let graph: MeshGraph<E3> = Cube::new()
            .polygons::<Position<E3>>() // 6 quadrilaterals, 24 vertices.
            .collect();

        let vertex = graph.vertices().nth(0).unwrap();
        assert_eq!(graph.vertex_count(), vertex.traverse_by_breadth().count());
    }

    #[test]
    fn traverse_by_depth() {
        let graph: MeshGraph<E3> = Cube::new()
            .polygons::<Position<E3>>() // 6 quadrilaterals, 24 vertices.
            .collect();

        let vertex = graph.vertices().nth(0).unwrap();
        assert_eq!(graph.vertex_count(), vertex.traverse_by_depth().count());
    }
}
