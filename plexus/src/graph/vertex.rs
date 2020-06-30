use derivative::Derivative;
use fool::BoolExt;
use slotmap::DefaultKey;
use smallvec::SmallVec;
use std::mem;
use std::ops::{Deref, DerefMut};
use theon::space::Vector;
use theon::AsPosition;

use crate::entity::borrow::{Reborrow, ReborrowInto, ReborrowMut};
use crate::entity::dijkstra::{self, Metric};
use crate::entity::storage::{AsStorage, AsStorageMut, AsStorageOf, OpaqueKey, SlotStorage};
use crate::entity::traverse::{Adjacency, Breadth, Depth, Traversal};
use crate::entity::view::{Bind, ClosedView, Orphan, Rebind, Unbind, View};
use crate::entity::Entity;
use crate::graph::edge::{Arc, ArcKey, ArcOrphan, ArcView, Edge};
use crate::graph::face::{Face, FaceKey, FaceOrphan, FaceView};
use crate::graph::geometry::{
    Geometric, Geometry, GraphGeometry, VertexCentroid, VertexNormal, VertexPosition,
};
use crate::graph::mutation::vertex::{self, VertexRemoveCache};
use crate::graph::mutation::{Consistent, Mutable, Mutation};
use crate::graph::path::Path;
use crate::graph::trace::{Trace, TraceAny, TraceFirst};
use crate::graph::{GraphError, OptionExt as _, ResultExt as _};
use crate::transact::{Mutate, Transact};
use crate::IteratorExt as _;

/// Vertex entity.
#[derivative(Clone, Copy, Debug, Hash)]
#[derive(Derivative)]
pub struct Vertex<G>
where
    G: GraphGeometry,
{
    /// Geometry.
    ///
    /// The type of this field is derived from `GraphGeometry`.
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Vertex,
    /// Required key into the leading arc.
    pub(in crate::graph) arc: Option<ArcKey>,
}

impl<G> Vertex<G>
where
    G: GraphGeometry,
{
    pub(in crate::graph) fn new(geometry: G::Vertex) -> Self {
        Vertex {
            geometry,
            arc: None,
        }
    }
}

impl<G> Entity for Vertex<G>
where
    G: GraphGeometry,
{
    type Key = VertexKey;
    type Storage = SlotStorage<Self>;
}

/// Vertex key.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct VertexKey(DefaultKey);

impl OpaqueKey for VertexKey {
    type Inner = DefaultKey;

    fn from_inner(key: Self::Inner) -> Self {
        VertexKey(key)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

/// View of a vertex entity.
///
/// Exposes references to a vertex entity and provides the primary vertex API.
///
/// See the module documentation for more information about views.
pub struct VertexView<B>
where
    B: Reborrow,
    B::Target: AsStorage<Vertex<Geometry<B>>> + Geometric,
{
    inner: View<B, Vertex<Geometry<B>>>,
}

impl<B, M> VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<Geometry<B>>> + Geometric,
{
    pub fn to_ref(&self) -> VertexView<&M> {
        self.inner.to_ref().into()
    }

    pub fn position<'a>(&'a self) -> &'a VertexPosition<Geometry<B>>
    where
        Geometry<B>: 'a,
        <Geometry<B> as GraphGeometry>::Vertex: AsPosition,
    {
        self.geometry.as_position()
    }
}

impl<B, M> VertexView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorage<Vertex<Geometry<B>>> + Geometric,
{
    fn to_mut(&mut self) -> VertexView<&mut M> {
        self.inner.to_mut().into()
    }
}

impl<'a, B, M, G> VertexView<B>
where
    B: ReborrowInto<'a, Target = M>,
    M: AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    // TODO: Relocate this documentation of `into_ref`.
    /// # Examples
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::N64;
    /// use nalgebra::Point3;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    ///
    /// let mut graph = Cube::new()
    ///     .polygons::<Position<Point3<N64>>>()
    ///     .collect::<MeshGraph<Point3<N64>>>();
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

/// Reachable API.
impl<B, M, G> VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    pub(in crate::graph) fn into_reachable_outgoing_arc(self) -> Option<ArcView<B>> {
        let key = self.arc;
        key.and_then(|key| self.rebind(key))
    }
}

impl<B, M, G> VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    /// Converts the vertex into its leading (outgoing) arc.
    pub fn into_outgoing_arc(self) -> ArcView<B> {
        self.into_reachable_outgoing_arc().expect_consistent()
    }

    /// Gets the leading (outgoing) arc of the vertex.
    pub fn outgoing_arc(&self) -> ArcView<&M> {
        self.to_ref().into_outgoing_arc()
    }

    pub fn shortest_path(&self, key: VertexKey) -> Result<Path<&M>, GraphError> {
        self.to_ref().into_shortest_path(key)
    }

    pub fn into_shortest_path(self, key: VertexKey) -> Result<Path<B>, GraphError> {
        self.into_shortest_path_with(key, |_, _| 1usize)
    }

    pub fn shortest_path_with<Q, F>(&self, key: VertexKey, f: F) -> Result<Path<&M>, GraphError>
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
    ) -> Result<Path<B>, GraphError>
    where
        Q: Copy + Metric,
        F: Fn(VertexView<&M>, VertexView<&M>) -> Q,
    {
        let metrics = dijkstra::metrics_with(self.to_ref(), Some(key), f);
        let mut keys = vec![key];
        while let Some((Some(previous), _)) = metrics.get(&key) {
            key = *previous;
            keys.push(key);
        }
        let (storage, _) = self.unbind();
        // TODO: This will fail if `key` is not found in the graph or is
        //       unreachable, but with a `TopologyMalformed` error. It would be
        //       better to emit `TopologyNotFound` and some kind of unreachable
        //       error in these cases.
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
        + Geometric<Geometry = G>,
    G: GraphGeometry,
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
    M: 'a + AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
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
    B::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
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
    M: 'a + AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
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
    B::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
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
        + Geometric<Geometry = G>,
    G: GraphGeometry,
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
        + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    /// Gets an iterator of views over the adjacent faces of the vertex.
    ///
    /// The ordering of faces is deterministic and is based on the leading arc
    /// of the vertex.
    pub fn adjacent_faces(&self) -> impl Clone + Iterator<Item = FaceView<&B::Target>> {
        self.to_ref().into_adjacent_faces()
    }
}

impl<'a, M, G> VertexView<&'a mut M>
where
    M: AsStorage<Arc<G>> + AsStorageMut<Vertex<G>> + Consistent + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    pub fn into_adjacent_vertex_orphans(self) -> impl Iterator<Item = VertexOrphan<'a, G>> {
        VertexCirculator::from(ArcCirculator::<TraceFirst<_>, _>::from(self))
    }
}

impl<B> VertexView<B>
where
    B: ReborrowMut,
    B::Target:
        AsStorage<Arc<Geometry<B>>> + AsStorageMut<Vertex<Geometry<B>>> + Consistent + Geometric,
{
    pub fn adjacent_vertex_orphans(&mut self) -> impl Iterator<Item = VertexOrphan<Geometry<B>>> {
        self.to_mut().into_adjacent_vertex_orphans()
    }
}

impl<'a, M, G> VertexView<&'a mut M>
where
    M: AsStorageMut<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    pub fn into_incoming_arc_orphans(self) -> impl Iterator<Item = ArcOrphan<'a, G>> {
        ArcCirculator::<TraceFirst<_>, _>::from(self)
    }
}

impl<B> VertexView<B>
where
    B: ReborrowMut,
    B::Target:
        AsStorageMut<Arc<Geometry<B>>> + AsStorage<Vertex<Geometry<B>>> + Consistent + Geometric,
{
    /// Gets an iterator of orphan views over the incoming arcs of the vertex.
    ///
    /// The ordering of arcs is deterministic and is based on the leading arc of
    /// the vertex.
    pub fn incoming_arc_orphans(&mut self) -> impl Iterator<Item = ArcOrphan<Geometry<B>>> {
        self.to_mut().into_incoming_arc_orphans()
    }
}

impl<'a, M, G> VertexView<&'a mut M>
where
    M: AsStorage<Arc<G>>
        + AsStorageMut<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    pub fn into_adjacent_face_orphans(self) -> impl Iterator<Item = FaceOrphan<'a, G>> {
        FaceCirculator::from(ArcCirculator::<TraceFirst<_>, _>::from(self))
    }
}

impl<B> VertexView<B>
where
    B: ReborrowMut,
    B::Target: AsStorage<Arc<Geometry<B>>>
        + AsStorageMut<Face<Geometry<B>>>
        + AsStorage<Vertex<Geometry<B>>>
        + Consistent
        + Geometric,
{
    /// Gets an iterator of orphan views over the adjacent faces of the vertex.
    ///
    /// The ordering of faces is deterministic and is based on the leading arc
    /// of the vertex.
    pub fn adjacent_face_orphans(&mut self) -> impl Iterator<Item = FaceOrphan<Geometry<B>>> {
        self.to_mut().into_adjacent_face_orphans()
    }
}

impl<'a, M, G> VertexView<&'a mut M>
where
    M: AsStorage<Arc<G>>
        + AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Default
        + Mutable<Geometry = G>,
    G: GraphGeometry,
{
    // TODO: This is not yet implemented, so examples use `no_run`. Run these
    //       examples in doc tests once this no longer intentionally panics.
    /// Removes the vertex.
    ///
    /// Any and all dependent topology is also removed, such as arcs and edges
    /// connected to the vertex, faces connected to such arcs, vertices with no
    /// remaining leading arc, etc.
    ///
    /// Vertex removal is the most destructive removal, because vertices are a
    /// dependency of all other topological structures.
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
    /// use decorum::N64;
    /// use nalgebra::Point3;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    ///
    /// let mut graph = Cube::new()
    ///     .polygons::<Position<Point3<N64>>>()
    ///     .collect::<MeshGraph<Point3<f64>>>();
    /// let key = graph.vertices().nth(0).unwrap().key();
    /// graph.vertex_mut(key).unwrap().remove();
    /// ```
    pub fn remove(self) {
        // This should never fail here.
        let cache = VertexRemoveCache::from_vertex(self.to_ref()).expect_consistent();
        let (storage, _) = self.unbind();
        Mutation::replace(storage, Default::default())
            .commit_with(|mutation| vertex::remove(mutation, cache))
            .map(|_| ())
            .expect_consistent()
    }
}

impl<B, M, G> Adjacency for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Output = SmallVec<[Self::Key; 8]>;

    fn adjacency(&self) -> Self::Output {
        self.adjacent_vertices().keys().collect()
    }
}

impl<B, M, G> ClosedView for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Key = VertexKey;
    type Entity = Vertex<G>;

    /// Gets the key for the vertex.
    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<B, M, G> Clone for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
    View<B, Vertex<G>>: Clone,
{
    fn clone(&self) -> Self {
        VertexView {
            inner: self.inner.clone(),
        }
    }
}

impl<B, M, G> Copy for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
    View<B, Vertex<G>>: Copy,
{
}

impl<B, M, G> Deref for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Target = Vertex<G>;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl<B, M, G> DerefMut for VertexView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorageMut<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

impl<B, M, G> From<View<B, Vertex<G>>> for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(view: View<B, Vertex<G>>) -> Self {
        VertexView { inner: view }
    }
}

impl<B, M, G> Into<View<B, Vertex<G>>> for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn into(self) -> View<B, Vertex<G>> {
        let VertexView { inner, .. } = self;
        inner
    }
}

impl<B, M, G> PartialEq for VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Orphan view of a vertex.
///
/// Provides mutable access to vertex's geometry. See the module documentation
/// for more information about topological views.
pub struct VertexOrphan<'a, G>
where
    G: GraphGeometry,
{
    inner: Orphan<'a, Vertex<G>>,
}

impl<'a, G> VertexOrphan<'a, G>
where
    G: GraphGeometry,
{
    pub fn position(&self) -> &VertexPosition<G>
    where
        G::Vertex: AsPosition,
    {
        self.inner.geometry.as_position()
    }
}

impl<'a, G> Deref for VertexOrphan<'a, G>
where
    G: GraphGeometry,
{
    type Target = Vertex<G>;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl<'a, G> DerefMut for VertexOrphan<'a, G>
where
    G: GraphGeometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

impl<'a, G> ClosedView for VertexOrphan<'a, G>
where
    G: GraphGeometry,
{
    type Key = VertexKey;
    type Entity = Vertex<G>;

    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<'a, G> From<Orphan<'a, Vertex<G>>> for VertexOrphan<'a, G>
where
    G: GraphGeometry,
{
    fn from(inner: Orphan<'a, Vertex<G>>) -> Self {
        VertexOrphan { inner }
    }
}

impl<'a, M, G> From<View<&'a mut M, Vertex<G>>> for VertexOrphan<'a, G>
where
    M: AsStorageMut<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(view: View<&'a mut M, Vertex<G>>) -> Self {
        VertexOrphan { inner: view.into() }
    }
}

impl<'a, M, G> From<VertexView<&'a mut M>> for VertexOrphan<'a, G>
where
    M: AsStorageMut<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(vertex: VertexView<&'a mut M>) -> Self {
        Orphan::from(vertex.inner).into()
    }
}

pub struct VertexCirculator<P, B>
where
    P: Trace<ArcKey>,
    B: Reborrow,
    B::Target: AsStorage<Arc<Geometry<B>>> + AsStorage<Vertex<Geometry<B>>> + Geometric,
{
    inner: ArcCirculator<P, B>,
}

impl<P, B, M, G> VertexCirculator<P, B>
where
    P: Trace<ArcKey>,
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn next(&mut self) -> Option<VertexKey> {
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
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
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
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(inner: ArcCirculator<P, B>) -> Self {
        VertexCirculator { inner }
    }
}

impl<'a, P, M, G> Iterator for VertexCirculator<P, &'a M>
where
    P: Trace<ArcKey>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Item = VertexView<&'a M>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).and_then(|key| Bind::bind(self.inner.storage, key))
    }
}

impl<'a, P, M, G> Iterator for VertexCirculator<P, &'a mut M>
where
    P: Trace<ArcKey>,
    M: AsStorage<Arc<G>> + AsStorageMut<Vertex<G>> + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    type Item = VertexOrphan<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).map(|key| {
            let vertex = self.inner.storage.as_storage_mut().get_mut(&key).unwrap();
            let vertex = unsafe { mem::transmute::<&'_ mut Vertex<G>, &'a mut Vertex<G>>(vertex) };
            Orphan::bind_unchecked(vertex, key).into()
        })
    }
}

pub struct ArcCirculator<P, B>
where
    P: Trace<ArcKey>,
    B: Reborrow,
    B::Target: AsStorage<Arc<Geometry<B>>> + Geometric,
{
    storage: B,
    outgoing: Option<ArcKey>,
    trace: P,
}

impl<P, B, M, G> ArcCirculator<P, B>
where
    P: Trace<ArcKey>,
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    #[allow(unstable_name_collisions)]
    fn next(&mut self) -> Option<ArcKey> {
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
    M: AsStorage<Arc<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
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
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
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

impl<'a, P, M, G> Iterator for ArcCirculator<P, &'a M>
where
    P: Trace<ArcKey>,
    M: AsStorage<Arc<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Item = ArcView<&'a M>;

    fn next(&mut self) -> Option<Self::Item> {
        ArcCirculator::next(self).and_then(|key| Bind::bind(self.storage, key))
    }
}

impl<'a, P, M, G> Iterator for ArcCirculator<P, &'a mut M>
where
    P: Trace<ArcKey>,
    M: AsStorageMut<Arc<G>> + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    type Item = ArcOrphan<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        ArcCirculator::next(self).map(|key| {
            let arc = self.storage.as_storage_mut().get_mut(&key).unwrap();
            let arc = unsafe { mem::transmute::<&'_ mut Arc<G>, &'a mut Arc<G>>(arc) };
            Orphan::bind_unchecked(arc, key).into()
        })
    }
}

pub struct FaceCirculator<P, B>
where
    P: Trace<ArcKey>,
    B: Reborrow,
    B::Target: AsStorage<Arc<Geometry<B>>> + AsStorage<Face<Geometry<B>>> + Geometric,
{
    inner: ArcCirculator<P, B>,
}

impl<P, B, M, G> FaceCirculator<P, B>
where
    P: Trace<ArcKey>,
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn next(&mut self) -> Option<FaceKey> {
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
    M: AsStorage<Arc<G>> + AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
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
    M: AsStorage<Arc<G>> + AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(inner: ArcCirculator<P, B>) -> Self {
        FaceCirculator { inner }
    }
}

impl<'a, P, M, G> Iterator for FaceCirculator<P, &'a M>
where
    P: Trace<ArcKey>,
    M: AsStorage<Arc<G>> + AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Item = FaceView<&'a M>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).and_then(|key| Bind::bind(self.inner.storage, key))
    }
}

impl<'a, P, M, G> Iterator for FaceCirculator<P, &'a mut M>
where
    P: Trace<ArcKey>,
    M: AsStorage<Arc<G>> + AsStorageMut<Face<G>> + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    type Item = FaceOrphan<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).map(|key| {
            let face = self.inner.storage.as_storage_mut().get_mut(&key).unwrap();
            let face = unsafe { mem::transmute::<&'_ mut Face<G>, &'a mut Face<G>>(face) };
            Orphan::bind_unchecked(face, key).into()
        })
    }
}

#[cfg(test)]
mod tests {
    use decorum::N64;
    use nalgebra::{Point2, Point3};

    use crate::graph::MeshGraph;
    use crate::prelude::*;
    use crate::primitive::cube::Cube;
    use crate::primitive::generate::Position;
    use crate::primitive::sphere::UvSphere;
    use crate::primitive::Trigon;

    type E3 = Point3<N64>;

    #[test]
    fn circulate_over_arcs() {
        let graph = UvSphere::new(4, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<E3>>();

        // All faces should be triangles and all vertices should have 4
        // (incoming) arcs.
        for vertex in graph.vertices() {
            assert_eq!(4, vertex.incoming_arcs().count());
        }
    }

    #[test]
    fn path() {
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
    fn traverse_by_breadth() {
        let graph = Cube::new()
            .polygons::<Position<E3>>() // 6 quadrilaterals, 24 vertices.
            .collect::<MeshGraph<E3>>();

        let vertex = graph.vertices().nth(0).unwrap();
        assert_eq!(graph.vertex_count(), vertex.traverse_by_breadth().count());
    }

    #[test]
    fn traverse_by_depth() {
        let graph = Cube::new()
            .polygons::<Position<E3>>() // 6 quadrilaterals, 24 vertices.
            .collect::<MeshGraph<E3>>();

        let vertex = graph.vertices().nth(0).unwrap();
        assert_eq!(graph.vertex_count(), vertex.traverse_by_depth().count());
    }
}
