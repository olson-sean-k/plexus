use fool::BoolExt;
use smallvec::SmallVec;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use theon::space::Vector;
use theon::AsPosition;

use crate::graph::borrow::{Reborrow, ReborrowMut};
use crate::graph::geometry::{GraphGeometry, VertexCentroid, VertexNormal, VertexPosition};
use crate::graph::mutation::vertex::{self, VertexRemoveCache};
use crate::graph::mutation::{Consistent, Mutable, Mutation};
use crate::graph::storage::alias::*;
use crate::graph::storage::key::{ArcKey, FaceKey, VertexKey};
use crate::graph::storage::payload::{Arc, Edge, Face, Vertex};
use crate::graph::storage::{AsStorage, AsStorageMut, StorageProxy};
use crate::graph::view::edge::{ArcOrphan, ArcView};
use crate::graph::view::face::{FaceOrphan, FaceView};
use crate::graph::view::traverse::{
    Adjacency, BreadthTraversal, DepthTraversal, Trace, TraceAny, TraceFirst,
};
use crate::graph::view::{ClosedView, Orphan, View};
use crate::graph::{GraphError, OptionExt as _, ResultExt as _};
use crate::transact::{Mutate, Transact};

/// View of a vertex in a graph.
///
/// Provides traversals, queries, and mutations related to vertices in a graph.
/// See the module documentation for more information about topological views.
pub struct VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: GraphGeometry,
{
    inner: View<M, Vertex<G>>,
}

impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: GraphGeometry,
{
    fn into_inner(self) -> View<M, Vertex<G>> {
        self.into()
    }

    fn interior_reborrow(&self) -> VertexView<&M::Target, G> {
        self.inner.interior_reborrow().into()
    }

    pub fn position(&self) -> &VertexPosition<G>
    where
        G::Vertex: AsPosition,
    {
        self.geometry.as_position()
    }
}

impl<M, G> VertexView<M, G>
where
    M: ReborrowMut,
    M::Target: AsStorage<Vertex<G>>,
    G: GraphGeometry,
{
    fn interior_reborrow_mut(&mut self) -> VertexView<&mut M::Target, G> {
        self.inner.interior_reborrow_mut().into()
    }
}

impl<'a, M, G> VertexView<&'a mut M, G>
where
    M: 'a + AsStorageMut<Vertex<G>>,
    G: 'a + GraphGeometry,
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
    pub fn into_ref(self) -> VertexView<&'a M, G> {
        self.into_inner().into_ref().into()
    }
}

/// Reachable API.
impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>>,
    G: GraphGeometry,
{
    pub(in crate::graph) fn into_reachable_outgoing_arc(self) -> Option<ArcView<M, G>> {
        let inner = self.into_inner();
        let key = inner.arc;
        key.and_then(move |key| inner.rebind_into(key))
    }

    pub(in crate::graph) fn reachable_outgoing_arc(&self) -> Option<ArcView<&M::Target, G>> {
        self.arc
            .and_then(|key| self.inner.interior_reborrow().rebind_into(key))
    }

    pub(in crate::graph) fn reachable_incoming_arcs(
        &self,
    ) -> impl Clone + Iterator<Item = ArcView<&M::Target, G>> {
        // This reachable circulator is needed for face insertions.
        ArcCirculator::from(self.interior_reborrow())
    }

    pub(in crate::graph) fn reachable_outgoing_arcs(
        &self,
    ) -> impl Clone + Iterator<Item = ArcView<&M::Target, G>> {
        // This reachable circulator is needed for face insertions.
        self.reachable_incoming_arcs()
            .flat_map(|arc| arc.into_reachable_opposite_arc())
    }
}

impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent,
    G: GraphGeometry,
{
    /// Converts the vertex into its leading (outgoing) arc.
    pub fn into_outgoing_arc(self) -> ArcView<M, G> {
        self.into_reachable_outgoing_arc().expect_consistent()
    }

    /// Gets the leading (outgoing) arc of the vertex.
    pub fn outgoing_arc(&self) -> ArcView<&M::Target, G> {
        self.reachable_outgoing_arc().expect_consistent()
    }

    /// Gets an iterator that traverses the vertices of the graph in
    /// breadth-first order beginning with the vertex on which this function is
    /// called.
    ///
    /// The traversal moves from the vertex to its neighboring vertices and so
    /// on. If there are disjoint subgraphs in the graph, then a traversal will
    /// not reach every vertex in the graph.
    pub fn traverse_by_breadth(&self) -> impl Clone + Iterator<Item = VertexView<&M::Target, G>> {
        BreadthTraversal::from(self.interior_reborrow())
    }

    /// Gets an iterator that traverses the vertices of the graph in depth-first
    /// order beginning with the vertex on which this function is called.
    ///
    /// The traversal moves from the vertex to its neighboring vertices and so
    /// on. If there are disjoint subgraphs in the graph, then a traversal will
    /// not reach every vertex in the graph.
    pub fn traverse_by_depth(&self) -> impl Clone + Iterator<Item = VertexView<&M::Target, G>> {
        DepthTraversal::from(self.interior_reborrow())
    }

    pub fn neighboring_vertices(&self) -> impl Clone + Iterator<Item = VertexView<&M::Target, G>> {
        VertexCirculator::from(ArcCirculator::<TraceFirst<_>, _, _>::from(
            self.interior_reborrow(),
        ))
    }

    /// Gets an iterator of views over the incoming arcs of the vertex.
    ///
    /// The ordering of arcs is deterministic and is based on the leading arc of
    /// the vertex.
    pub fn incoming_arcs(&self) -> impl Clone + Iterator<Item = ArcView<&M::Target, G>> {
        ArcCirculator::<TraceFirst<_>, _, _>::from(self.interior_reborrow())
    }

    /// Gets an iterator of views over the outgoing arcs of the vertex.
    ///
    /// The ordering of arcs is deterministic and is based on the leading arc of
    /// the vertex.
    pub fn outgoing_arcs(&self) -> impl Clone + Iterator<Item = ArcView<&M::Target, G>> {
        ArcCirculator::<TraceFirst<_>, _, _>::from(self.interior_reborrow())
            .map(|arc| arc.into_opposite_arc())
    }

    /// Gets the valence of the vertex.
    ///
    /// A vertex's _valence_ is the number of neighboring vertices to which it
    /// is connected by arcs. The valence of a vertex is the same as its
    /// _degree_, which is the number of edges to which the vertex is connected.
    pub fn valence(&self) -> usize {
        self.neighboring_vertices().count()
    }

    pub fn centroid(&self) -> VertexPosition<G>
    where
        G: VertexCentroid,
        G::Vertex: AsPosition,
    {
        G::centroid(self.interior_reborrow()).expect_consistent()
    }
}

impl<M, G> VertexView<M, G>
where
    M: ReborrowMut,
    M::Target: AsStorage<Arc<G>> + AsStorageMut<Vertex<G>> + Consistent,
    G: GraphGeometry,
{
    pub fn neighboring_vertex_orphans(&mut self) -> impl Iterator<Item = VertexOrphan<G>> {
        VertexCirculator::from(ArcCirculator::<TraceFirst<_>, _, _>::from(
            self.interior_reborrow_mut(),
        ))
    }
}

impl<M, G> VertexView<M, G>
where
    M: ReborrowMut,
    M::Target: AsStorageMut<Arc<G>> + AsStorage<Vertex<G>> + Consistent,
    G: GraphGeometry,
{
    /// Gets an iterator of orphan views over the incoming arcs of the vertex.
    ///
    /// The ordering of arcs is deterministic and is based on the leading arc of
    /// the vertex.
    pub fn incoming_arc_orphans(&mut self) -> impl Iterator<Item = ArcOrphan<G>> {
        ArcCirculator::<TraceFirst<_>, _, _>::from(self.interior_reborrow_mut())
    }
}

impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    G: GraphGeometry,
{
    /// Gets an iterator of views over the neighboring faces of the vertex.
    ///
    /// The ordering of faces is deterministic and is based on the leading arc
    /// of the vertex.
    pub fn neighboring_faces(&self) -> impl Clone + Iterator<Item = FaceView<&M::Target, G>> {
        FaceCirculator::from(ArcCirculator::<TraceFirst<_>, _, _>::from(
            self.interior_reborrow(),
        ))
    }

    pub fn normal(&self) -> Result<Vector<VertexPosition<G>>, GraphError>
    where
        G: VertexNormal,
        G::Vertex: AsPosition,
    {
        <G as VertexNormal>::normal(self.interior_reborrow())
    }
}

impl<M, G> VertexView<M, G>
where
    M: ReborrowMut,
    M::Target: AsStorage<Arc<G>> + AsStorageMut<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    G: GraphGeometry,
{
    /// Gets an iterator of orphan views over the neighboring faces of the
    /// vertex.
    ///
    /// The ordering of faces is deterministic and is based on the leading arc
    /// of the vertex.
    pub fn neighboring_face_orphans(&mut self) -> impl Iterator<Item = FaceOrphan<G>> {
        FaceCirculator::from(ArcCirculator::<TraceFirst<_>, _, _>::from(
            self.interior_reborrow_mut(),
        ))
    }
}

impl<'a, M, G> VertexView<&'a mut M, G>
where
    M: AsStorage<Arc<G>>
        + AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Default
        + Mutable<G>,
    G: 'a + GraphGeometry,
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
            .commit_with(move |mutation| vertex::remove_with_cache(mutation, cache))
            .map(|_| ())
            .expect_consistent()
    }
}

impl<M, G> Adjacency for VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent,
    G: GraphGeometry,
{
    type Output = SmallVec<[Self::Key; 8]>;

    fn adjacency(&self) -> Self::Output {
        self.neighboring_vertices()
            .map(|vertex| vertex.key())
            .collect()
    }
}

impl<M, G> ClosedView for VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: GraphGeometry,
{
    type Key = VertexKey;
    type Entity = Vertex<G>;

    /// Gets the key for the vertex.
    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<M, G> Clone for VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: GraphGeometry,
    View<M, Vertex<G>>: Clone,
{
    fn clone(&self) -> Self {
        VertexView {
            inner: self.inner.clone(),
        }
    }
}

impl<M, G> Copy for VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: GraphGeometry,
    View<M, Vertex<G>>: Copy,
{
}

impl<M, G> Deref for VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: GraphGeometry,
{
    type Target = Vertex<G>;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl<M, G> DerefMut for VertexView<M, G>
where
    M: ReborrowMut,
    M::Target: AsStorageMut<Vertex<G>>,
    G: GraphGeometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

impl<M, G> From<View<M, Vertex<G>>> for VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: GraphGeometry,
{
    fn from(view: View<M, Vertex<G>>) -> Self {
        VertexView { inner: view }
    }
}

impl<M, G> Into<View<M, Vertex<G>>> for VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: GraphGeometry,
{
    fn into(self) -> View<M, Vertex<G>> {
        let VertexView { inner, .. } = self;
        inner
    }
}

impl<M, G> PartialEq for VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>> + Consistent,
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
    G: 'a + GraphGeometry,
{
    inner: Orphan<'a, Vertex<G>>,
}

impl<'a, G> VertexOrphan<'a, G>
where
    G: 'a + GraphGeometry,
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
    G: 'a + GraphGeometry,
{
    type Target = Vertex<G>;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl<'a, G> DerefMut for VertexOrphan<'a, G>
where
    G: 'a + GraphGeometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

impl<'a, G> ClosedView for VertexOrphan<'a, G>
where
    G: 'a + GraphGeometry,
{
    type Key = VertexKey;
    type Entity = Vertex<G>;

    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<'a, G> From<Orphan<'a, Vertex<G>>> for VertexOrphan<'a, G>
where
    G: 'a + GraphGeometry,
{
    fn from(inner: Orphan<'a, Vertex<G>>) -> Self {
        VertexOrphan { inner }
    }
}

impl<'a, M, G> From<VertexView<&'a mut M, G>> for VertexOrphan<'a, G>
where
    M: AsStorageMut<Vertex<G>>,
    G: 'a + GraphGeometry,
{
    fn from(vertex: VertexView<&'a mut M, G>) -> Self {
        Orphan::from(vertex.into_inner()).into()
    }
}

pub struct VertexCirculator<P, M, G>
where
    P: Trace<ArcKey>,
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>>,
    G: GraphGeometry,
{
    inner: ArcCirculator<P, M, G>,
}

impl<P, M, G> VertexCirculator<P, M, G>
where
    P: Trace<ArcKey>,
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>>,
    G: GraphGeometry,
{
    fn next(&mut self) -> Option<VertexKey> {
        self.inner.next().map(|arc| {
            let (source, _) = arc.into();
            source
        })
    }
}

impl<P, M, G> Clone for VertexCirculator<P, M, G>
where
    P: Clone + Trace<ArcKey>,
    M: Clone + Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>>,
    G: GraphGeometry,
{
    fn clone(&self) -> Self {
        VertexCirculator {
            inner: self.inner.clone(),
        }
    }
}

impl<P, M, G> From<ArcCirculator<P, M, G>> for VertexCirculator<P, M, G>
where
    P: Trace<ArcKey>,
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>>,
    G: GraphGeometry,
{
    fn from(inner: ArcCirculator<P, M, G>) -> Self {
        VertexCirculator { inner }
    }
}

impl<'a, P, M, G> Iterator for VertexCirculator<P, &'a M, G>
where
    P: 'a + Trace<ArcKey>,
    M: 'a + AsStorage<Arc<G>> + AsStorage<Vertex<G>>,
    G: 'a + GraphGeometry,
{
    type Item = VertexView<&'a M, G>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).and_then(|key| View::bind_into(self.inner.storage, key))
    }
}

impl<'a, P, M, G> Iterator for VertexCirculator<P, &'a mut M, G>
where
    P: 'a + Trace<ArcKey>,
    M: 'a + AsStorage<Arc<G>> + AsStorageMut<Vertex<G>>,
    G: 'a + GraphGeometry,
{
    type Item = VertexOrphan<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).and_then(|key| {
            let storage = unsafe {
                mem::transmute::<&'_ mut StorageProxy<Vertex<G>>, &'a mut StorageProxy<Vertex<G>>>(
                    self.inner.storage.as_storage_mut(),
                )
            };
            Orphan::bind_into(storage, key)
        })
    }
}

pub struct ArcCirculator<P, M, G>
where
    P: Trace<ArcKey>,
    M: Reborrow,
    M::Target: AsStorage<Arc<G>>,
    G: GraphGeometry,
{
    storage: M,
    outgoing: Option<ArcKey>,
    trace: P,
    phantom: PhantomData<G>,
}

impl<P, M, G> ArcCirculator<P, M, G>
where
    P: Trace<ArcKey>,
    M: Reborrow,
    M::Target: AsStorage<Arc<G>>,
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

impl<P, M, G> Clone for ArcCirculator<P, M, G>
where
    P: Clone + Trace<ArcKey>,
    M: Clone + Reborrow,
    M::Target: AsStorage<Arc<G>>,
    G: GraphGeometry,
{
    fn clone(&self) -> Self {
        ArcCirculator {
            storage: self.storage.clone(),
            outgoing: self.outgoing,
            trace: self.trace.clone(),
            phantom: PhantomData,
        }
    }
}

impl<M, G> From<VertexView<M, G>> for ArcCirculator<TraceAny<ArcKey>, M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>>,
    G: GraphGeometry,
{
    fn from(vertex: VertexView<M, G>) -> Self {
        let inner = vertex.into_inner();
        let key = inner.arc;
        let (storage, _) = inner.unbind();
        ArcCirculator {
            storage,
            outgoing: key,
            trace: Default::default(),
            phantom: PhantomData,
        }
    }
}

impl<M, G> From<VertexView<M, G>> for ArcCirculator<TraceFirst<ArcKey>, M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent,
    G: GraphGeometry,
{
    fn from(vertex: VertexView<M, G>) -> Self {
        let inner = vertex.into_inner();
        let key = inner.arc;
        let (storage, _) = inner.unbind();
        ArcCirculator {
            storage,
            outgoing: key,
            trace: Default::default(),
            phantom: PhantomData,
        }
    }
}

impl<'a, P, M, G> Iterator for ArcCirculator<P, &'a M, G>
where
    P: 'a + Trace<ArcKey>,
    M: 'a + AsStorage<Arc<G>>,
    G: 'a + GraphGeometry,
{
    type Item = ArcView<&'a M, G>;

    fn next(&mut self) -> Option<Self::Item> {
        ArcCirculator::next(self).and_then(|key| View::bind_into(self.storage, key))
    }
}

impl<'a, P, M, G> Iterator for ArcCirculator<P, &'a mut M, G>
where
    P: 'a + Trace<ArcKey>,
    M: 'a + AsStorageMut<Arc<G>>,
    G: 'a + GraphGeometry,
{
    type Item = ArcOrphan<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        ArcCirculator::next(self).and_then(|key| {
            let storage = unsafe {
                mem::transmute::<&'_ mut StorageProxy<Arc<G>>, &'a mut StorageProxy<Arc<G>>>(
                    self.storage.as_storage_mut(),
                )
            };
            Orphan::bind_into(storage, key)
        })
    }
}

pub struct FaceCirculator<P, M, G>
where
    P: Trace<ArcKey>,
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>>,
    G: GraphGeometry,
{
    inner: ArcCirculator<P, M, G>,
}

impl<P, M, G> FaceCirculator<P, M, G>
where
    P: Trace<ArcKey>,
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>>,
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

impl<P, M, G> Clone for FaceCirculator<P, M, G>
where
    P: Clone + Trace<ArcKey>,
    M: Clone + Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>>,
    G: GraphGeometry,
{
    fn clone(&self) -> Self {
        FaceCirculator {
            inner: self.inner.clone(),
        }
    }
}

impl<P, M, G> From<ArcCirculator<P, M, G>> for FaceCirculator<P, M, G>
where
    P: Trace<ArcKey>,
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>>,
    G: GraphGeometry,
{
    fn from(inner: ArcCirculator<P, M, G>) -> Self {
        FaceCirculator { inner }
    }
}

impl<'a, P, M, G> Iterator for FaceCirculator<P, &'a M, G>
where
    P: 'a + Trace<ArcKey>,
    M: 'a + AsStorage<Arc<G>> + AsStorage<Face<G>>,
    G: 'a + GraphGeometry,
{
    type Item = FaceView<&'a M, G>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).and_then(|key| View::bind_into(self.inner.storage, key))
    }
}

impl<'a, P, M, G> Iterator for FaceCirculator<P, &'a mut M, G>
where
    P: 'a + Trace<ArcKey>,
    M: 'a + AsStorage<Arc<G>> + AsStorageMut<Face<G>>,
    G: 'a + GraphGeometry,
{
    type Item = FaceOrphan<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).and_then(|key| {
            let storage = unsafe {
                mem::transmute::<&'_ mut StorageProxy<Face<G>>, &'a mut StorageProxy<Face<G>>>(
                    self.inner.storage.as_storage_mut(),
                )
            };
            Orphan::bind_into(storage, key)
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
