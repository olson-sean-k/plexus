use derivative::Derivative;
use fool::BoolExt;
use slotmap::DefaultKey;
use smallvec::SmallVec;
use std::mem;
use std::ops::{Deref, DerefMut};
use theon::space::Vector;
use theon::AsPosition;

use crate::graph::edge::{Arc, ArcKey, ArcOrphan, ArcView, Edge};
use crate::graph::face::{Face, FaceKey, FaceOrphan, FaceView};
use crate::graph::geometry::{
    Geometric, Geometry, GraphGeometry, VertexCentroid, VertexNormal, VertexPosition,
};
use crate::graph::mutation::vertex::{self, VertexRemoveCache};
use crate::graph::mutation::{Consistent, Mutable, Mutation};
use crate::graph::storage::*;
use crate::graph::trace::{Trace, TraceAny, TraceFirst};
use crate::graph::{GraphError, OptionExt as _, ResultExt as _};
use crate::network::borrow::{Reborrow, ReborrowMut};
use crate::network::storage::{AsStorage, AsStorageMut, OpaqueKey, SlotStorage};
use crate::network::traverse::{Adjacency, BreadthTraversal, DepthTraversal};
use crate::network::view::{ClosedView, Orphan, View};
use crate::network::Entity;
use crate::transact::{Mutate, Transact};

/// Graph vertex.
#[derivative(Clone, Copy, Debug, Hash)]
#[derive(Derivative)]
pub struct Vertex<G>
where
    G: GraphGeometry,
{
    /// User geometry.
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

/// View of a vertex in a graph.
///
/// Provides traversals, queries, and mutations related to vertices in a graph.
/// See the module documentation for more information about topological views.
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
    fn into_inner(self) -> View<B, Vertex<Geometry<B>>> {
        self.into()
    }

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
    pub fn to_mut(&mut self) -> VertexView<&mut M> {
        self.inner.to_mut().into()
    }
}

impl<'a, M> VertexView<&'a mut M>
where
    M: AsStorageMut<Vertex<Geometry<M>>> + Geometric,
{
    /// Converts a mutable view into an immutable view.
    ///
    /// This is useful when mutations are not (or no longer) needed and mutual
    /// access is desired.
    ///
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
        self.into_inner().into_ref().into()
    }
}

/// Reachable API.
impl<B, M> VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Vertex<Geometry<B>>> + Geometric,
{
    pub(in crate::graph) fn into_reachable_outgoing_arc(self) -> Option<ArcView<B>> {
        let inner = self.into_inner();
        let key = inner.arc;
        key.and_then(move |key| inner.rebind_into(key))
    }

    pub(in crate::graph) fn reachable_outgoing_arc(&self) -> Option<ArcView<&M>> {
        self.arc
            .and_then(|key| self.inner.to_ref().rebind_into(key))
    }

    pub(in crate::graph) fn reachable_incoming_arcs<'a>(
        &'a self,
    ) -> impl Clone + Iterator<Item = ArcView<&'a M>>
    where
        M: 'a,
    {
        // This reachable circulator is needed for face insertions.
        ArcCirculator::<TraceAny<_>, _>::from(self.to_ref())
    }

    pub(in crate::graph) fn reachable_outgoing_arcs<'a>(
        &'a self,
    ) -> impl Clone + Iterator<Item = ArcView<&'a M>>
    where
        M: 'a,
    {
        // This reachable circulator is needed for face insertions.
        self.reachable_incoming_arcs()
            .flat_map(|arc| arc.into_reachable_opposite_arc())
    }
}

impl<B, M> VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Vertex<Geometry<B>>> + Consistent + Geometric,
{
    /// Converts the vertex into its leading (outgoing) arc.
    pub fn into_outgoing_arc(self) -> ArcView<B> {
        self.into_reachable_outgoing_arc().expect_consistent()
    }

    /// Gets the leading (outgoing) arc of the vertex.
    pub fn outgoing_arc(&self) -> ArcView<&M> {
        self.reachable_outgoing_arc().expect_consistent()
    }

    /// Gets an iterator that traverses the vertices of the graph in
    /// breadth-first order beginning with the vertex on which this function is
    /// called.
    ///
    /// The traversal moves from the vertex to its neighboring vertices and so
    /// on. If there are disjoint subgraphs in the graph, then a traversal will
    /// not reach every vertex in the graph.
    pub fn traverse_by_breadth<'a>(&'a self) -> impl Clone + Iterator<Item = VertexView<&'a M>>
    where
        M: 'a,
    {
        BreadthTraversal::from(self.to_ref())
    }

    /// Gets an iterator that traverses the vertices of the graph in depth-first
    /// order beginning with the vertex on which this function is called.
    ///
    /// The traversal moves from the vertex to its neighboring vertices and so
    /// on. If there are disjoint subgraphs in the graph, then a traversal will
    /// not reach every vertex in the graph.
    pub fn traverse_by_depth<'a>(&'a self) -> impl Clone + Iterator<Item = VertexView<&'a M>>
    where
        M: 'a,
    {
        DepthTraversal::from(self.to_ref())
    }

    pub fn neighboring_vertices<'a>(&'a self) -> impl Clone + Iterator<Item = VertexView<&'a M>>
    where
        M: 'a,
    {
        VertexCirculator::from(ArcCirculator::<TraceFirst<_>, _>::from(self.to_ref()))
    }

    /// Gets an iterator of views over the incoming arcs of the vertex.
    ///
    /// The ordering of arcs is deterministic and is based on the leading arc of
    /// the vertex.
    pub fn incoming_arcs<'a>(&'a self) -> impl Clone + Iterator<Item = ArcView<&'a M>>
    where
        M: 'a,
    {
        ArcCirculator::<TraceFirst<_>, _>::from(self.to_ref())
    }

    /// Gets an iterator of views over the outgoing arcs of the vertex.
    ///
    /// The ordering of arcs is deterministic and is based on the leading arc of
    /// the vertex.
    pub fn outgoing_arcs<'a>(&'a self) -> impl Clone + Iterator<Item = ArcView<&'a M>>
    where
        M: 'a,
    {
        ArcCirculator::<TraceFirst<_>, _>::from(self.to_ref()).map(|arc| arc.into_opposite_arc())
    }

    /// Gets the valence of the vertex.
    ///
    /// A vertex's _valence_ is the number of neighboring vertices to which it
    /// is connected by arcs. The valence of a vertex is the same as its
    /// _degree_, which is the number of edges to which the vertex is connected.
    pub fn valence(&self) -> usize {
        self.neighboring_vertices().count()
    }

    pub fn centroid(&self) -> VertexPosition<Geometry<B>>
    where
        Geometry<B>: VertexCentroid,
        <Geometry<B> as GraphGeometry>::Vertex: AsPosition,
    {
        <Geometry<B> as VertexCentroid>::centroid(self.to_ref()).expect_consistent()
    }
}

impl<B, M> VertexView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorageMut<Vertex<Geometry<B>>> + Consistent + Geometric,
{
    pub fn neighboring_vertex_orphans<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = VertexOrphan<'a, Geometry<B>>>
    where
        M: 'a,
    {
        VertexCirculator::from(ArcCirculator::<TraceFirst<_>, _>::from(self.to_mut()))
    }
}

impl<B, M> VertexView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorageMut<Arc<Geometry<B>>> + AsStorage<Vertex<Geometry<B>>> + Consistent + Geometric,
{
    /// Gets an iterator of orphan views over the incoming arcs of the vertex.
    ///
    /// The ordering of arcs is deterministic and is based on the leading arc of
    /// the vertex.
    pub fn incoming_arc_orphans<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = ArcOrphan<'a, Geometry<B>>>
    where
        M: 'a,
    {
        ArcCirculator::<TraceFirst<_>, _>::from(self.to_mut())
    }
}

impl<B, M> VertexView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>>
        + AsStorage<Face<Geometry<B>>>
        + AsStorage<Vertex<Geometry<B>>>
        + Consistent
        + Geometric,
{
    /// Gets an iterator of views over the neighboring faces of the vertex.
    ///
    /// The ordering of faces is deterministic and is based on the leading arc
    /// of the vertex.
    pub fn neighboring_faces<'a>(&'a self) -> impl Clone + Iterator<Item = FaceView<&'a M>>
    where
        M: 'a,
    {
        FaceCirculator::from(ArcCirculator::<TraceFirst<_>, _>::from(self.to_ref()))
    }

    pub fn normal(&self) -> Result<Vector<VertexPosition<Geometry<B>>>, GraphError>
    where
        Geometry<B>: VertexNormal,
        <Geometry<B> as GraphGeometry>::Vertex: AsPosition,
    {
        <Geometry<B> as VertexNormal>::normal(self.to_ref())
    }
}

impl<B, M> VertexView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorage<Arc<Geometry<B>>>
        + AsStorageMut<Face<Geometry<B>>>
        + AsStorage<Vertex<Geometry<B>>>
        + Consistent
        + Geometric,
{
    /// Gets an iterator of orphan views over the neighboring faces of the
    /// vertex.
    ///
    /// The ordering of faces is deterministic and is based on the leading arc
    /// of the vertex.
    pub fn neighboring_face_orphans<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = FaceOrphan<'a, Geometry<B>>>
    where
        M: 'a,
    {
        FaceCirculator::from(ArcCirculator::<TraceFirst<_>, _>::from(self.to_mut()))
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
        let (storage, a) = self.into_inner().unbind();
        let cache = VertexRemoveCache::snapshot(&storage, a).expect_consistent();
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| vertex::remove(mutation, cache))
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
        self.neighboring_vertices()
            .map(|vertex| vertex.key())
            .collect()
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
        Orphan::from(vertex.into_inner()).into()
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
    G: 'a + GraphGeometry,
{
    type Item = VertexView<&'a M>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).and_then(|key| View::bind_into(self.inner.storage, key))
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
    fn next(&mut self) -> Option<ArcKey> {
        self.outgoing
            .and_then(|outgoing| self.trace.insert(outgoing).some(outgoing))
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
        let inner = vertex.into_inner();
        let key = inner.arc;
        let (storage, _) = inner.unbind();
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
    G: 'a + GraphGeometry,
{
    type Item = ArcView<&'a M>;

    fn next(&mut self) -> Option<Self::Item> {
        ArcCirculator::next(self).and_then(|key| View::bind_into(self.storage, key))
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
                .as_arc_storage()
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
    G: 'a + GraphGeometry,
{
    type Item = FaceView<&'a M>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).and_then(|key| View::bind_into(self.inner.storage, key))
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
    use nalgebra::Point3;

    use crate::graph::MeshGraph;
    use crate::prelude::*;
    use crate::primitive::cube::Cube;
    use crate::primitive::generate::Position;
    use crate::primitive::sphere::UvSphere;

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
