use arrayvec::ArrayVec;
use derivative::Derivative;
use fool::BoolExt;
use slotmap::DefaultKey;
use std::mem;
use std::ops::{Deref, DerefMut};
use theon::space::{EuclideanSpace, Scalar, Vector};
use theon::AsPosition;

use crate::graph::face::{Face, FaceKey, FaceOrphan, FaceView, Ring};
use crate::graph::geometry::{
    ArcNormal, EdgeMidpoint, Geometric, Geometry, GraphGeometry, VertexPosition,
};
use crate::graph::mutation::edge::{
    self, ArcBridgeCache, ArcExtrudeCache, EdgeRemoveCache, EdgeSplitCache,
};
use crate::graph::mutation::{Consistent, Mutable, Mutation};
use crate::graph::path::Path;
use crate::graph::vertex::{Vertex, VertexKey, VertexOrphan, VertexView};
use crate::graph::{GraphError, OptionExt as _, ResultExt as _, Selector};
use crate::network::borrow::{Reborrow, ReborrowMut};
use crate::network::storage::{AsStorage, AsStorageMut, HashStorage, OpaqueKey, SlotStorage};
use crate::network::view::{Bind, ClosedView, Orphan, Rebind, Unbind, View};
use crate::network::Entity;
use crate::transact::{Mutate, Transact};

/// Edge-like structure. Abstracts arcs and edges.
///
/// Types implementing this trait participate in a composite edge and can be
/// converted into an arc or edge that is a part of that composite edge. This
/// trait allows edge structures to be abstracted.
pub trait Edgoid<B>
where
    B: Reborrow,
    B::Target: AsStorage<Arc<Geometry<B>>> + AsStorage<Edge<Geometry<B>>> + Consistent + Geometric,
{
    fn into_arc(self) -> ArcView<B>;

    fn into_edge(self) -> EdgeView<B>;
}

// Unlike other graph structures, the vertex connectivity of an arc is immutable
// and encoded within its key. This provides fast and reliable lookups even when
// a graph is in an inconsistent state. However, it also complicates certain
// topological mutations and sometimes requires that arcs be rekeyed. For this
// reason, `Arc` has no fields representing its source and destination vertices
// nor its opposite arc; such fields would be redundant.
/// Graph arc.
#[derivative(Clone, Copy, Debug, Hash)]
#[derive(Derivative)]
pub struct Arc<G>
where
    G: GraphGeometry,
{
    /// User geometry.
    ///
    /// The type of this field is derived from `GraphGeometry`.
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Arc,
    /// Required key into the next arc.
    pub(in crate::graph) next: Option<ArcKey>,
    /// Required key into the previous arc.
    pub(in crate::graph) previous: Option<ArcKey>,
    /// Required key into the edge.
    pub(in crate::graph) edge: Option<EdgeKey>,
    /// Optional key into the face.
    pub(in crate::graph) face: Option<FaceKey>,
}

impl<G> Arc<G>
where
    G: GraphGeometry,
{
    pub(in crate::graph) fn new(geometry: G::Arc) -> Self {
        Arc {
            geometry,
            next: None,
            previous: None,
            edge: None,
            face: None,
        }
    }
}

impl<G> Entity for Arc<G>
where
    G: GraphGeometry,
{
    type Key = ArcKey;
    type Storage = HashStorage<Self>;
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ArcKey(VertexKey, VertexKey);

impl ArcKey {
    pub(in crate::graph) fn into_opposite(self) -> ArcKey {
        let (a, b) = self.into();
        (b, a).into()
    }
}

impl From<(VertexKey, VertexKey)> for ArcKey {
    fn from(key: (VertexKey, VertexKey)) -> Self {
        ArcKey(key.0, key.1)
    }
}

impl Into<(VertexKey, VertexKey)> for ArcKey {
    fn into(self) -> (VertexKey, VertexKey) {
        (self.0, self.1)
    }
}

impl OpaqueKey for ArcKey {
    type Inner = (VertexKey, VertexKey);

    fn from_inner(key: Self::Inner) -> Self {
        ArcKey(key.0, key.1)
    }

    fn into_inner(self) -> Self::Inner {
        (self.0, self.1)
    }
}

/// View of an arc in a graph.
///
/// Provides traversals, queries, and mutations related to arcs in a graph. See
/// the module documentation for more information about topological views.
///
/// Arcs provide the connectivity information within a `MeshGraph` and are the
/// primary mechanism for traversing its topology. Moreover, most edge-like
/// operations are exposed by arcs, because they are directed and therefore can
/// emit deterministic results (this is not true of edges).
///
/// An arc from a vertex $A$ to a vertex $B$ is notated $\overrightarrow{AB}$.
/// This is shorthand for the path notation $\overrightarrow{(A,B)}$.
///
/// # Examples
///
/// Traversing a graph of a cube via its arcs to find an opposing face:
///
/// ```rust
/// # extern crate decorum;
/// # extern crate nalgebra;
/// # extern crate plexus;
/// #
/// use decorum::N64;
/// use nalgebra::Point3;
/// use plexus::graph::MeshGraph;
/// use plexus::index::HashIndexer;
/// use plexus::prelude::*;
/// use plexus::primitive::cube::Cube;
/// use plexus::primitive::generate::Position;
///
/// let mut graph = Cube::new()
///     .polygons::<Position<Point3<N64>>>()
///     .collect_with_indexer::<MeshGraph<Point3<N64>>, _>(HashIndexer::default())
///     .unwrap();
///
/// let face = graph.faces().nth(0).unwrap();
/// let opposite = face
///     .into_arc()
///     .into_opposite_arc()
///     .into_next_arc()
///     .into_next_arc()
///     .into_opposite_arc()
///     .into_face()
///     .unwrap();
/// ```
pub struct ArcView<B>
where
    B: Reborrow,
    B::Target: AsStorage<Arc<Geometry<B>>> + Geometric,
{
    inner: View<B, Arc<Geometry<B>>>,
}

impl<B, M> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + Geometric,
{
    fn into_inner(self) -> View<B, Arc<Geometry<B>>> {
        self.into()
    }

    pub fn to_ref(&self) -> ArcView<&M> {
        self.inner.to_ref().into()
    }

    /// Returns `true` if this is a boundary arc.
    ///
    /// A boundary arc has no associated face.
    pub fn is_boundary_arc(&self) -> bool {
        self.face.is_none()
    }
}

impl<B, M> ArcView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + Geometric,
{
    pub fn to_mut(&mut self) -> ArcView<&mut M> {
        self.inner.to_mut().into()
    }
}

impl<'a, M> ArcView<&'a mut M>
where
    M: AsStorageMut<Arc<Geometry<M>>> + Geometric,
{
    /// Converts a mutable view into an immutable view.
    ///
    /// This is useful when mutations are not (or no longer) needed and mutual
    /// access is desired.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point2;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    ///
    /// let mut graph = MeshGraph::<Point2<f64>>::from_raw_buffers_with_arity(
    ///     vec![0u32, 1, 2, 3],
    ///     vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    ///     4,
    /// )
    /// .unwrap();
    /// let key = graph
    ///     .arcs()
    ///     .find(|arc| arc.is_boundary_arc())
    ///     .unwrap()
    ///     .key();
    /// let arc = graph.arc_mut(key).unwrap().extrude(1.0).unwrap().into_ref();
    ///
    /// // This would not be possible without conversion into an immutable view.
    /// let _ = arc.into_next_arc().into_next_arc().into_face();
    /// let _ = arc.into_opposite_arc().into_face();
    /// ```
    pub fn into_ref(self) -> ArcView<&'a M> {
        self.into_inner().into_ref().into()
    }
}

/// Reachable API.
impl<B, M> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + Geometric,
{
    pub(in crate::graph) fn into_reachable_boundary_arc(self) -> Option<Self> {
        if self.is_boundary_arc() {
            Some(self)
        }
        else {
            self.into_reachable_opposite_arc()
                .and_then(|opposite| opposite.is_boundary_arc().some(opposite))
        }
    }

    pub(in crate::graph) fn into_reachable_opposite_arc(self) -> Option<Self> {
        let key = self.key().into_opposite();
        self.rebind(key)
    }

    pub(in crate::graph) fn into_reachable_next_arc(self) -> Option<Self> {
        let key = self.next;
        key.and_then(|key| self.rebind(key))
    }

    pub(in crate::graph) fn into_reachable_previous_arc(self) -> Option<Self> {
        let key = self.previous;
        key.and_then(|key| self.rebind(key))
    }

    pub(in crate::graph) fn reachable_boundary_arc(&self) -> Option<ArcView<&M>> {
        if self.is_boundary_arc() {
            Some(self.to_ref())
        }
        else {
            self.reachable_opposite_arc()
                .and_then(|opposite| opposite.is_boundary_arc().some_with(|| opposite))
        }
    }

    pub(in crate::graph) fn reachable_opposite_arc(&self) -> Option<ArcView<&M>> {
        let key = self.key().into_opposite();
        self.to_ref().rebind(key)
    }

    pub(in crate::graph) fn reachable_next_arc(&self) -> Option<ArcView<&M>> {
        self.next.and_then(|key| self.to_ref().rebind(key))
    }

    pub(in crate::graph) fn reachable_previous_arc(&self) -> Option<ArcView<&M>> {
        self.previous.and_then(|key| self.to_ref().rebind(key))
    }
}

impl<B, M> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + Consistent + Geometric,
{
    /// Converts the arc into its ring.
    pub fn into_ring(self) -> Ring<B> {
        self.into()
    }

    /// Returns the arc if it is a boundary arc, otherwise `None`.
    pub fn into_boundary_arc(self) -> Option<Self> {
        self.into_reachable_boundary_arc()
    }

    /// Converts the arc into its opposite arc.
    pub fn into_opposite_arc(self) -> Self {
        self.into_reachable_opposite_arc().expect_consistent()
    }

    /// Converts the arc into its next arc.
    pub fn into_next_arc(self) -> Self {
        self.into_reachable_next_arc().expect_consistent()
    }

    /// Converts the arc into its previous arc.
    pub fn into_previous_arc(self) -> Self {
        self.into_reachable_previous_arc().expect_consistent()
    }

    /// Gets the ring of the arc.
    pub fn ring(&self) -> Ring<&M> {
        self.to_ref().into()
    }

    /// Returns the same arc if it is a boundary arc, otherwise `None`.
    pub fn boundary_arc(&self) -> Option<ArcView<&M>> {
        self.reachable_boundary_arc()
    }

    /// Gets the opposite arc.
    pub fn opposite_arc(&self) -> ArcView<&M> {
        self.reachable_opposite_arc().expect_consistent()
    }

    /// Gets the next arc.
    pub fn next_arc(&self) -> ArcView<&M> {
        self.reachable_next_arc().expect_consistent()
    }

    /// Gets the previous arc.
    pub fn previous_arc(&self) -> ArcView<&M> {
        self.reachable_previous_arc().expect_consistent()
    }
}

/// Reachable API.
impl<B, M> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Vertex<Geometry<B>>> + Geometric,
{
    pub(in crate::graph) fn into_reachable_source_vertex(self) -> Option<VertexView<B>> {
        let (key, _) = self.key().into();
        self.rebind(key)
    }

    pub(in crate::graph) fn into_reachable_destination_vertex(self) -> Option<VertexView<B>> {
        let (_, key) = self.key().into();
        self.rebind(key)
    }

    pub(in crate::graph) fn reachable_source_vertex(&self) -> Option<VertexView<&M>> {
        let (key, _) = self.key().into();
        self.to_ref().rebind(key)
    }

    pub(in crate::graph) fn reachable_destination_vertex(&self) -> Option<VertexView<&M>> {
        let (_, key) = self.key().into();
        self.to_ref().rebind(key)
    }
}

impl<B, M> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Vertex<Geometry<B>>> + Consistent + Geometric,
{
    pub fn into_path(self) -> Path<B> {
        let (storage, ab) = self.unbind();
        let (a, b) = ab.into();
        Path::bind(storage, &[a, b]).unwrap()
    }

    pub fn path(&self) -> Path<&M> {
        self.to_ref().into_path()
    }

    /// Converts the arc into its source vertex.
    pub fn into_source_vertex(self) -> VertexView<B> {
        self.into_reachable_source_vertex().expect_consistent()
    }

    /// Converts the arc into its destination vertex.
    pub fn into_destination_vertex(self) -> VertexView<B> {
        self.into_reachable_destination_vertex().expect_consistent()
    }

    /// Gets the source vertex of the arc.
    pub fn source_vertex(&self) -> VertexView<&M> {
        self.reachable_source_vertex().expect_consistent()
    }

    /// Gets the destination vertex of the arc.
    pub fn destination_vertex(&self) -> VertexView<&M> {
        self.reachable_destination_vertex().expect_consistent()
    }
}

/// Reachable API.
impl<B, M> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Edge<Geometry<B>>> + Geometric,
{
    pub(in crate::graph) fn into_reachable_edge(self) -> Option<EdgeView<B>> {
        let key = self.edge;
        key.and_then(|key| self.rebind(key))
    }

    pub(in crate::graph) fn reachable_edge(&self) -> Option<EdgeView<&M>> {
        self.edge.and_then(|key| self.to_ref().rebind(key))
    }
}

impl<B, M> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Edge<Geometry<B>>> + Consistent + Geometric,
{
    /// Converts the arc into its edge.
    pub fn into_edge(self) -> EdgeView<B> {
        self.into_reachable_edge().expect_consistent()
    }

    /// Gets the edge of the arc.
    pub fn edge(&self) -> EdgeView<&M> {
        self.reachable_edge().expect_consistent()
    }
}

/// Reachable API.
impl<B, M> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Face<Geometry<B>>> + Geometric,
{
    pub(in crate::graph) fn into_reachable_face(self) -> Option<FaceView<B>> {
        let key = self.face;
        key.and_then(|key| self.rebind(key))
    }

    pub(in crate::graph) fn reachable_face(&self) -> Option<FaceView<&M>> {
        self.face.and_then(|key| self.to_ref().rebind(key))
    }
}

impl<B, M> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Face<Geometry<B>>> + Consistent + Geometric,
{
    /// Converts the arc into its face.
    ///
    /// If this is a boundary arc, then `None` is returned.
    pub fn into_face(self) -> Option<FaceView<B>> {
        self.into_reachable_face()
    }

    /// Gets the face of this arc.
    ///
    /// If this is a boundary arc, then `None` is returned.
    pub fn face(&self) -> Option<FaceView<&M>> {
        self.reachable_face()
    }
}

impl<B, M> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Vertex<Geometry<B>>> + Consistent + Geometric,
{
    /// Gets an iterator of views over the vertices connected by the arc.
    pub fn vertices<'a>(&'a self) -> impl Clone + ExactSizeIterator<Item = VertexView<&'a M>>
    where
        M: 'a,
    {
        VertexCirculator::from(self.to_ref())
    }
}

impl<B, M> ArcView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorageMut<Vertex<Geometry<B>>> + Consistent + Geometric,
{
    /// Gets an iterator of orphan views over the vertices connected by the arc.
    pub fn vertex_orphans<'a>(
        &'a mut self,
    ) -> impl ExactSizeIterator<Item = VertexOrphan<Geometry<B>>>
    where
        M: 'a,
    {
        VertexCirculator::from(self.to_mut())
    }
}

impl<B, M> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Face<Geometry<B>>> + Consistent + Geometric,
{
    /// Gets an iterator of views over the faces connected to the arc.
    pub fn faces<'a>(&'a self) -> impl Clone + ExactSizeIterator<Item = FaceView<&'a M>>
    where
        M: 'a,
    {
        FaceCirculator::from(self.to_ref())
    }
}

impl<B, M> ArcView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorageMut<Face<Geometry<B>>> + Consistent + Geometric,
{
    /// Gets an iterator of orphan views over the faces connected to the arc.
    pub fn face_orphans<'a>(&'a mut self) -> impl ExactSizeIterator<Item = FaceOrphan<Geometry<B>>>
    where
        M: 'a,
    {
        FaceCirculator::from(self.to_mut())
    }
}

impl<B, M> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Vertex<Geometry<B>>> + Consistent + Geometric,
{
    pub fn normal(&self) -> Vector<VertexPosition<Geometry<B>>>
    where
        Geometry<B>: ArcNormal,
        <Geometry<B> as GraphGeometry>::Vertex: AsPosition,
    {
        <Geometry<B> as ArcNormal>::normal(self.to_ref()).expect_consistent()
    }
}

impl<B, M, G> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>>
        + AsStorage<Edge<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    pub fn midpoint(&self) -> VertexPosition<G>
    where
        G: EdgeMidpoint,
        G::Vertex: AsPosition,
    {
        G::midpoint(self.to_ref()).expect_consistent()
    }
}

impl<'a, M, G> ArcView<&'a mut M>
where
    M: AsStorage<Arc<G>>
        + AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Default
        + Mutable<Geometry = G>,
    G: GraphGeometry,
{
    /// Splits a composite edge into two neighboring edges that share a vertex.
    ///
    /// Splitting inserts a new vertex with the geometry provided by the given
    /// function. Splitting an arc $\overrightarrow{AB}$ returns a vertex $M$
    /// that subdivides the composite edge. The leading arc of $M$ is
    /// $\overrightarrow{MB}$ and is a part of the same ring as the initiating
    /// arc.
    ///
    /// Returns the inserted vertex.
    ///
    /// # Examples
    ///
    /// Split an edge in a graph with weighted vertices:
    ///
    /// ```rust
    /// use plexus::graph::{GraphGeometry, MeshGraph};
    /// use plexus::prelude::*;
    /// use plexus::primitive::NGon;
    ///
    /// pub enum Weight {}
    ///
    /// impl GraphGeometry for Weight {
    ///     type Vertex = f64;
    ///     type Arc = ();
    ///     type Edge = ();
    ///     type Face = ();
    /// }
    ///
    /// let mut graph =
    ///     MeshGraph::<Weight>::from_raw_buffers(vec![NGon([0usize, 1, 2])], vec![1.0, 2.0, 0.5])
    ///         .unwrap();
    /// let key = graph.arcs().nth(0).unwrap().key();
    /// let vertex = graph.arc_mut(key).unwrap().split_with(|| 0.1);
    /// ```
    pub fn split_with<F>(self, f: F) -> VertexView<&'a mut M>
    where
        F: FnOnce() -> G::Vertex,
    {
        let (storage, ab) = self.unbind();
        let cache = EdgeSplitCache::snapshot(&storage, ab).expect_consistent();
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| edge::split_with(mutation, cache, f))
            .map(|(storage, m)| Bind::bind(storage, m).expect_consistent())
            .expect_consistent()
    }

    /// Splits an edge (and its arcs) at the midpoint of the arc's vertices.
    ///
    /// Splitting inserts a new vertex with the geometry of the arc's source
    /// vertex but modified such that the positional data of the vertex is the
    /// computed midpoint of both of the arc's vertices.
    ///
    /// Splitting inserts a new vertex with the geometry provided by the given
    /// function. Splitting an arc $\overrightarrow{AB}$ returns a vertex $M$
    /// that subdivides the composite edge. The leading arc of $M$ is
    /// $\overrightarrow{MB}$ and is a part of the same ring as the initiating
    /// arc.
    ///
    /// This function is only available if a graph's geometry exposes positional
    /// data in its vertices and that data supports interpolation. See the
    /// `geometry` module.
    ///
    /// Returns the inserted vertex.
    ///
    /// # Examples
    ///
    /// Split an edge in a triangle at its midpoint:
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point2;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::Trigon;
    ///
    /// let mut graph = MeshGraph::<Point2<f64>>::from_raw_buffers(
    ///     vec![Trigon::new(0usize, 1, 2)],
    ///     vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    /// )
    /// .unwrap();
    /// let key = graph.arcs().nth(0).unwrap().key();
    /// let vertex = graph.arc_mut(key).unwrap().split_at_midpoint();
    /// ```
    pub fn split_at_midpoint(self) -> VertexView<&'a mut M>
    where
        G: EdgeMidpoint,
        G::Vertex: AsPosition,
    {
        let mut geometry = self.source_vertex().geometry;
        let midpoint = self.midpoint();
        self.split_with(move || {
            *geometry.as_position_mut() = midpoint;
            geometry
        })
    }

    // TODO: What if an edge in the bridging quadrilateral is collapsed, such as
    //       bridging arcs within a triangular ring? Document these edge cases
    //       (no pun intended).
    /// Connects a boundary arc to another boundary arc with a face.
    ///
    /// Bridging arcs inserts a new face and, as needed, new arcs and edges.
    /// The inserted face is always a quadrilateral. The bridged arcs must be
    /// boundary arcs with an orientation that allows them to form a ring.
    ///
    /// Bridging two compatible arcs $\overrightarrow{AB}$ and
    /// $\overrightarrow{CD}$ will result in a ring $\overrightarrow{\\{A,B,
    /// C,D\\}}$.
    ///
    /// Arcs can be bridged within a ring. The destination arc can be chosen by
    /// key or index, where an index selects the $n^\text{th}$ arc from the
    /// source arc within the ring.
    ///
    /// Returns the inserted face if successful.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination arc cannot be found, either arc is
    /// not a boundary arc, or the orientation of the destination arc is
    /// incompatible with the initiating arc.
    ///
    /// # Examples
    ///
    /// Bridging two disjoint quadrilaterals together:
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point2;
    /// use plexus::graph::{GraphGeometry, MeshGraph, VertexKey, VertexView};
    /// use plexus::prelude::*;
    /// use plexus::primitive::NGon;
    /// use plexus::IntoGeometry;
    ///
    /// fn find<'a, I, T, G>(input: I, geometry: T) -> Option<VertexKey>
    /// where
    ///     I: IntoIterator<Item = VertexView<&'a MeshGraph<G>>>,
    ///     T: Copy + IntoGeometry<G::Vertex>,
    ///     G: 'a + GraphGeometry,
    ///     G::Vertex: PartialEq,
    /// {
    ///     input
    ///         .into_iter()
    ///         .find(|vertex| vertex.geometry == geometry.into_geometry())
    ///         .map(|vertex| vertex.key())
    /// }
    ///
    /// let mut graph = MeshGraph::<Point2<f64>>::from_raw_buffers(
    ///     vec![NGon([0usize, 1, 2, 3]), NGon([4, 5, 6, 7])],
    ///     vec![
    ///         (-2.0, 0.0),
    ///         (-1.0, 0.0), // b
    ///         (-1.0, 1.0), // a
    ///         (-2.0, 1.0),
    ///         (1.0, 0.0), // c
    ///         (2.0, 0.0),
    ///         (2.0, 1.0),
    ///         (1.0, 1.0), // d
    ///     ],
    /// )
    /// .unwrap();
    /// let a = find(graph.vertices(), (-1.0, 1.0)).unwrap();
    /// let b = find(graph.vertices(), (-1.0, 0.0)).unwrap();
    /// let c = find(graph.vertices(), (1.0, 0.0)).unwrap();
    /// let d = find(graph.vertices(), (1.0, 1.0)).unwrap();
    /// let face = graph
    ///     .arc_mut((a, b).into())
    ///     .unwrap()
    ///     .bridge(ByKey((c, d).into()))
    ///     .unwrap();
    /// ```
    pub fn bridge(self, destination: Selector<ArcKey>) -> Result<FaceView<&'a mut M>, GraphError> {
        let destination = destination.key_or_else(|index| {
            self.ring()
                .interior_arcs()
                .nth(index)
                .ok_or_else(|| GraphError::TopologyNotFound)
                .map(|arc| arc.key())
        })?;
        let (storage, source) = self.unbind();
        // Errors can easily be caused by inputs to this function. Allow errors
        // from the snapshot to propagate.
        let cache = ArcBridgeCache::snapshot(&storage, source, destination)?;
        Ok(Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| edge::bridge(mutation, cache))
            .map(|(storage, face)| Bind::bind(storage, face).expect_consistent())
            .expect_consistent())
    }

    /// Extrudes a boundary arc along its normal into a composite edge.
    ///
    /// Extrusion inserts a new composite edge with the same geometry as the
    /// initiating arc and its composite edge, but modifies the positional
    /// geometry of the new edge's vertices such that they extend geometrically
    /// along the normal of the originating arc. The originating arc is then
    /// bridged with an arc in the opposing edge. This inserts a quadrilateral
    /// face. See `bridge`.
    ///
    /// An arc's normal is perpendicular to the arc and also coplanar with the
    /// arc and one of its neighbors. This is computed via a projection and
    /// supports both 2D and 3D geometries.
    ///
    /// Returns the opposing arc. This is the arc in the destination edge that
    /// is within the same ring as the initiating arc.
    ///
    /// # Errors
    ///
    /// Returns an error if the arc is not a boundary arc.
    ///
    /// # Examples
    ///
    /// Extrude an exterior arc of a quadrilateral.
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point2;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    ///
    /// let mut graph = MeshGraph::<Point2<f64>>::from_raw_buffers_with_arity(
    ///     vec![0usize, 1, 2, 3],
    ///     vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    ///     4,
    /// )
    /// .unwrap();
    /// let key = graph
    ///     .arcs()
    ///     .find(|arc| arc.is_boundary_arc())
    ///     .map(|arc| arc.key())
    ///     .unwrap();
    /// graph.arc_mut(key).unwrap().extrude(1.0).unwrap();
    /// ```
    pub fn extrude<T>(self, offset: T) -> Result<ArcView<&'a mut M>, GraphError>
    where
        T: Into<Scalar<VertexPosition<G>>>,
        G: ArcNormal,
        G::Vertex: AsPosition,
        VertexPosition<G>: EuclideanSpace,
    {
        let normal = self.normal();
        let (storage, ab) = self.unbind();
        let cache = ArcExtrudeCache::snapshot(&storage, ab).expect_consistent();
        Ok(Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| {
                edge::extrude_with(mutation, cache, || normal * offset.into())
            })
            .map(|(storage, arc)| Bind::bind(storage, arc).expect_consistent())
            .expect_consistent())
    }

    /// Removes the arc and its composite edge.
    ///
    /// Any and all dependent topology is also removed, such as connected faces,
    /// disjoint vertices, etc.
    ///
    /// Returns the source vertex of the initiating arc or `None` if that vertex
    /// becomes disjoint and is also removed. If an arc $\overrightarrow{AB}$ is
    /// removed and its source vertex is not disjoint, then $A$ is returned.
    pub fn remove(self) -> Option<VertexView<&'a mut M>> {
        let a = self.source_vertex().key();
        let (storage, ab) = self.unbind();
        let cache = EdgeRemoveCache::snapshot(&storage, ab).expect_consistent();
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| edge::remove(mutation, cache))
            .map(|(storage, _)| Bind::bind(storage, a))
            .expect_consistent()
    }
}

impl<B, M, G> ClosedView for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Key = ArcKey;
    type Entity = Arc<G>;

    /// Gets the key for the arc.
    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<B, M, G> Clone for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
    View<B, Arc<G>>: Clone,
{
    fn clone(&self) -> Self {
        ArcView {
            inner: self.inner.clone(),
        }
    }
}

impl<B, M, G> Edgoid<B> for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Edge<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn into_arc(self) -> ArcView<B> {
        self
    }

    fn into_edge(self) -> EdgeView<B> {
        self.into_edge()
    }
}

impl<B, M, G> Copy for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
    View<B, Arc<G>>: Copy,
{
}

impl<B, M, G> Deref for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Target = Arc<G>;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl<B, M, G> DerefMut for ArcView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorageMut<Arc<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

impl<B, M, G> From<Ring<B>> for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(ring: Ring<B>) -> Self {
        ring.into_arc()
    }
}

impl<B, M, G> From<View<B, Arc<G>>> for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(view: View<B, Arc<G>>) -> Self {
        ArcView { inner: view }
    }
}

impl<B, M, G> Into<View<B, Arc<G>>> for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn into(self) -> View<B, Arc<G>> {
        let ArcView { inner, .. } = self;
        inner
    }
}

impl<B, M, G> PartialEq for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Orphan view of an arc.
///
/// Provides mutable access to an arc's geometry. See the module documentation
/// for more information about topological views.
pub struct ArcOrphan<'a, G>
where
    G: GraphGeometry,
{
    inner: Orphan<'a, Arc<G>>,
}

impl<'a, G> ClosedView for ArcOrphan<'a, G>
where
    G: GraphGeometry,
{
    type Key = ArcKey;
    type Entity = Arc<G>;

    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<'a, G> Deref for ArcOrphan<'a, G>
where
    G: GraphGeometry,
{
    type Target = Arc<G>;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl<'a, G> DerefMut for ArcOrphan<'a, G>
where
    G: GraphGeometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

impl<'a, M, G> From<ArcView<&'a mut M>> for ArcOrphan<'a, G>
where
    M: AsStorageMut<Arc<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(arc: ArcView<&'a mut M>) -> Self {
        Orphan::from(arc.into_inner()).into()
    }
}

impl<'a, G> From<Orphan<'a, Arc<G>>> for ArcOrphan<'a, G>
where
    G: GraphGeometry,
{
    fn from(inner: Orphan<'a, Arc<G>>) -> Self {
        ArcOrphan { inner }
    }
}

impl<'a, M, G> From<View<&'a mut M, Arc<G>>> for ArcOrphan<'a, G>
where
    M: AsStorageMut<Arc<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(view: View<&'a mut M, Arc<G>>) -> Self {
        ArcOrphan { inner: view.into() }
    }
}

/// Graph edge.
#[derivative(Clone, Copy, Debug, Hash)]
#[derive(Derivative)]
pub struct Edge<G>
where
    G: GraphGeometry,
{
    /// User geometry.
    ///
    /// The type of this field is derived from `GraphGeometry`.
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Edge,
    /// Required key into the leading arc.
    pub(in crate::graph) arc: ArcKey,
}

impl<G> Edge<G>
where
    G: GraphGeometry,
{
    pub(in crate::graph) fn new(arc: ArcKey, geometry: G::Edge) -> Self {
        Edge { geometry, arc }
    }
}

impl<G> Entity for Edge<G>
where
    G: GraphGeometry,
{
    type Key = EdgeKey;
    type Storage = SlotStorage<Self>;
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EdgeKey(DefaultKey);

impl OpaqueKey for EdgeKey {
    type Inner = DefaultKey;

    fn from_inner(key: Self::Inner) -> Self {
        EdgeKey(key)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

/// View of an edge in a graph.
///
/// Provides traversals, queries, and mutations related to edges in a graph.
/// See the module documentation for more information about topological views.
///
/// An edge connecting a vertex $A$ and a vertex $B$ is notated
/// $\overleftrightarrow{AB}$ or $\overleftrightarrow{BA}$ (both representing
/// the same edge). Typically, edges are described by one of their arcs (e.g.,
/// "the edge of $\overrightarrow{AB}$").
pub struct EdgeView<B>
where
    B: Reborrow,
    B::Target: AsStorage<Edge<<B::Target as Geometric>::Geometry>> + Geometric,
{
    inner: View<B, Edge<<B::Target as Geometric>::Geometry>>,
}

impl<B, M> EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<Geometry<B>>> + Geometric,
{
    fn into_inner(self) -> View<B, Edge<Geometry<B>>> {
        self.into()
    }

    pub fn to_ref(&self) -> EdgeView<&M> {
        self.inner.to_ref().into()
    }
}

impl<B, M> EdgeView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorageMut<Edge<Geometry<B>>> + Geometric,
{
    pub fn to_mut(&mut self) -> EdgeView<&mut M> {
        self.inner.to_mut().into()
    }
}

impl<'a, M> EdgeView<&'a mut M>
where
    M: AsStorageMut<Edge<Geometry<M>>> + Geometric,
{
    /// Converts a mutable view into an immutable view.
    ///
    /// This is useful when mutations are not (or no longer) needed and mutual
    /// access is desired.
    pub fn into_ref(self) -> EdgeView<&'a M> {
        self.into_inner().into_ref().into()
    }
}

/// Reachable API.
impl<B, M> EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Edge<Geometry<B>>> + Geometric,
{
    pub(in crate::graph) fn into_reachable_arc(self) -> Option<ArcView<B>> {
        let key = self.arc;
        self.rebind(key)
    }

    pub(in crate::graph) fn reachable_arc(&self) -> Option<ArcView<&M>> {
        let key = self.arc;
        self.to_ref().rebind(key)
    }
}

impl<B, M> EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Edge<Geometry<B>>> + Consistent + Geometric,
{
    pub fn into_arc(self) -> ArcView<B> {
        self.into_reachable_arc().expect_consistent()
    }

    pub fn arc(&self) -> ArcView<&M> {
        self.reachable_arc().expect_consistent()
    }

    pub fn is_boundary_edge(&self) -> bool {
        let arc = self.arc();
        arc.is_boundary_arc() || arc.opposite_arc().is_boundary_arc()
    }
}

impl<B, M, G> EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>>
        + AsStorage<Edge<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    pub fn midpoint(&self) -> VertexPosition<G>
    where
        G: EdgeMidpoint,
        G::Vertex: AsPosition,
    {
        G::midpoint(self.to_ref()).expect_consistent()
    }
}

impl<B, M, G> Clone for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
    View<B, Edge<G>>: Clone,
{
    fn clone(&self) -> Self {
        EdgeView {
            inner: self.inner.clone(),
        }
    }
}

impl<B, M, G> Edgoid<B> for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Edge<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn into_arc(self) -> ArcView<B> {
        self.into_arc()
    }

    fn into_edge(self) -> EdgeView<B> {
        self
    }
}

impl<B, M, G> Copy for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
    View<B, Edge<G>>: Copy,
{
}

impl<B, M, G> Deref for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Target = Edge<G>;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl<B, M, G> DerefMut for EdgeView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorageMut<Edge<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

impl<B, M, G> ClosedView for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Key = EdgeKey;
    type Entity = Edge<G>;

    /// Gets the key for the edge.
    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<B, M, G> From<View<B, Edge<G>>> for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(view: View<B, Edge<G>>) -> Self {
        EdgeView { inner: view }
    }
}

impl<B, M, G> Into<View<B, Edge<G>>> for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn into(self) -> View<B, Edge<G>> {
        let EdgeView { inner, .. } = self;
        inner
    }
}

impl<B, M, G> PartialEq for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Orphan view of an edge.
///
/// Provides mutable access to an edge's geometry. See the module documentation
/// for more information about topological views.
pub struct EdgeOrphan<'a, G>
where
    G: GraphGeometry,
{
    inner: Orphan<'a, Edge<G>>,
}

impl<'a, G> ClosedView for EdgeOrphan<'a, G>
where
    G: GraphGeometry,
{
    type Key = EdgeKey;
    type Entity = Edge<G>;

    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<'a, G> Deref for EdgeOrphan<'a, G>
where
    G: GraphGeometry,
{
    type Target = Edge<G>;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl<'a, G> DerefMut for EdgeOrphan<'a, G>
where
    G: GraphGeometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

impl<'a, M, G> From<EdgeView<&'a mut M>> for EdgeOrphan<'a, G>
where
    M: AsStorageMut<Edge<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(edge: EdgeView<&'a mut M>) -> Self {
        Orphan::from(edge.into_inner()).into()
    }
}

impl<'a, G> From<Orphan<'a, Edge<G>>> for EdgeOrphan<'a, G>
where
    G: GraphGeometry,
{
    fn from(inner: Orphan<'a, Edge<G>>) -> Self {
        EdgeOrphan { inner }
    }
}

impl<'a, M, G> From<View<&'a mut M, Edge<G>>> for EdgeOrphan<'a, G>
where
    M: AsStorageMut<Edge<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(view: View<&'a mut M, Edge<G>>) -> Self {
        EdgeOrphan { inner: view.into() }
    }
}

pub struct VertexCirculator<B>
where
    B: Reborrow,
    B::Target: AsStorage<Vertex<<B::Target as Geometric>::Geometry>> + Geometric,
{
    storage: B,
    inner: <ArrayVec<[VertexKey; 2]> as IntoIterator>::IntoIter,
}

impl<B, M, G> VertexCirculator<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn next(&mut self) -> Option<VertexKey> {
        self.inner.next()
    }
}

impl<B, M, G> From<ArcView<B>> for VertexCirculator<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(arc: ArcView<B>) -> Self {
        let (a, b) = arc.key().into();
        let (storage, _) = arc.unbind();
        VertexCirculator {
            storage,
            inner: ArrayVec::from([a, b]).into_iter(),
        }
    }
}

impl<B, M, G> Clone for VertexCirculator<B>
where
    B: Clone + Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn clone(&self) -> Self {
        VertexCirculator {
            storage: self.storage.clone(),
            inner: self.inner.clone(),
        }
    }
}

impl<B, M, G> ExactSizeIterator for VertexCirculator<B>
where
    Self: Iterator,
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
}

impl<'a, M, G> Iterator for VertexCirculator<&'a M>
where
    M: AsStorage<Vertex<G>> + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    type Item = VertexView<&'a M>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).and_then(|key| Bind::bind(self.storage, key))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.inner.len()))
    }
}

impl<'a, M, G> Iterator for VertexCirculator<&'a mut M>
where
    M: AsStorageMut<Vertex<G>> + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    type Item = VertexOrphan<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).map(|key| {
            let vertex = self.storage.as_storage_mut().get_mut(&key).unwrap();
            let vertex = unsafe { mem::transmute::<&'_ mut Vertex<G>, &'a mut Vertex<G>>(vertex) };
            Orphan::bind_unchecked(vertex, key).into()
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.inner.len()))
    }
}

pub struct FaceCirculator<B>
where
    B: Reborrow,
    B::Target: AsStorage<Face<<B::Target as Geometric>::Geometry>> + Geometric,
{
    storage: B,
    inner: <ArrayVec<[FaceKey; 2]> as IntoIterator>::IntoIter,
}

impl<B, M, G> FaceCirculator<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn next(&mut self) -> Option<FaceKey> {
        self.inner.next()
    }
}

impl<B, M, G> Clone for FaceCirculator<B>
where
    B: Clone + Reborrow<Target = M>,
    M: AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn clone(&self) -> Self {
        FaceCirculator {
            storage: self.storage.clone(),
            inner: self.inner.clone(),
        }
    }
}

impl<B, M, G> ExactSizeIterator for FaceCirculator<B>
where
    Self: Iterator,
    B: Reborrow<Target = M>,
    M: AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
}

impl<B, M, G> From<ArcView<B>> for FaceCirculator<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(arc: ArcView<B>) -> Self {
        let inner = arc
            .face
            .into_iter()
            .chain(
                arc.reachable_opposite_arc()
                    .and_then(|opposite| opposite.face)
                    .into_iter(),
            )
            .collect::<ArrayVec<_>>()
            .into_iter();
        let (storage, _) = arc.unbind();
        FaceCirculator { storage, inner }
    }
}

impl<'a, M, G> Iterator for FaceCirculator<&'a M>
where
    M: AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    type Item = FaceView<&'a M>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).and_then(|key| Bind::bind(self.storage, key))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.inner.len()))
    }
}

impl<'a, M, G> Iterator for FaceCirculator<&'a mut M>
where
    M: AsStorageMut<Face<G>> + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    type Item = FaceOrphan<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).map(|key| {
            let face = self.storage.as_storage_mut().get_mut(&key).unwrap();
            let face = unsafe { mem::transmute::<&'_ mut Face<G>, &'a mut Face<G>>(face) };
            Orphan::bind_unchecked(face, key).into()
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.inner.len()))
    }
}

#[cfg(test)]
mod tests {
    use decorum::N64;
    use nalgebra::{Point2, Point3};

    use crate::graph::{ArcKey, GraphGeometry, MeshGraph, VertexView};
    use crate::index::HashIndexer;
    use crate::prelude::*;
    use crate::primitive::cube::Cube;
    use crate::primitive::generate::Position;
    use crate::primitive::Tetragon;
    use crate::IntoGeometry;

    fn find_arc_with_vertex_geometry<G, T>(graph: &MeshGraph<G>, geometry: (T, T)) -> Option<ArcKey>
    where
        G: GraphGeometry,
        G::Vertex: PartialEq,
        T: IntoGeometry<G::Vertex>,
    {
        fn find_vertex_with_geometry<G, T>(
            graph: &MeshGraph<G>,
            geometry: T,
        ) -> Option<VertexView<&MeshGraph<G>>>
        where
            G: GraphGeometry,
            G::Vertex: PartialEq,
            T: IntoGeometry<G::Vertex>,
        {
            let geometry = geometry.into_geometry();
            graph.vertices().find(|vertex| vertex.geometry == geometry)
        }

        let (source, destination) = geometry;
        find_vertex_with_geometry(graph, source)
            .and_then(|source| {
                find_vertex_with_geometry(graph, destination)
                    .map(move |destination| (source, destination))
            })
            .and_then(|(source, destination)| {
                destination
                    .incoming_arcs()
                    .find(|arc| arc.source_vertex().key() == source.key())
                    .map(|arc| arc.key())
            })
    }

    #[test]
    fn extrude_arc() {
        let mut graph = MeshGraph::<Point2<f32>>::from_raw_buffers_with_arity(
            vec![0u32, 1, 2, 3],
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            4,
        )
        .unwrap();
        let source = find_arc_with_vertex_geometry(&graph, ((1.0, 1.0), (1.0, 0.0))).unwrap();
        graph.arc_mut(source).unwrap().extrude(1.0).unwrap();

        assert_eq!(14, graph.arc_count());
        assert_eq!(2, graph.face_count());
    }

    #[test]
    fn bridge_arcs() {
        // Construct a mesh with two disjoint quadrilaterals.
        let mut graph = MeshGraph::<Point3<f32>>::from_raw_buffers_with_arity(
            vec![0u32, 1, 2, 3, 4, 5, 6, 7],
            vec![
                (-2.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0), // 1
                (-1.0, 1.0, 0.0), // 2
                (-2.0, 1.0, 0.0),
                (1.0, 0.0, 0.0), // 4
                (2.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
                (1.0, 1.0, 0.0), // 7
            ],
            4,
        )
        .unwrap();
        let source =
            find_arc_with_vertex_geometry(&graph, ((-1.0, 1.0, 0.0), (-1.0, 0.0, 0.0))).unwrap();
        let destination =
            find_arc_with_vertex_geometry(&graph, ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0))).unwrap();
        graph
            .arc_mut(source)
            .unwrap()
            .bridge(ByKey(destination))
            .unwrap();

        assert_eq!(20, graph.arc_count());
        assert_eq!(3, graph.face_count());
    }

    #[test]
    fn split_edge() {
        let (indices, vertices) = Cube::new()
            .polygons::<Position<Point3<N64>>>() // 6 quadrilaterals, 24 vertices.
            .index_vertices::<Tetragon<usize>, _>(HashIndexer::default());
        let mut graph = MeshGraph::<Point3<f64>>::from_raw_buffers(indices, vertices).unwrap();
        let key = graph.arcs().nth(0).unwrap().key();
        let vertex = graph.arc_mut(key).unwrap().split_at_midpoint().into_ref();

        assert_eq!(5, vertex.into_outgoing_arc().into_face().unwrap().arity());
        assert_eq!(
            5,
            vertex
                .into_outgoing_arc()
                .into_opposite_arc()
                .into_face()
                .unwrap()
                .arity()
        );
    }

    #[test]
    fn remove_edge() {
        // Construct a graph with two connected quadrilaterals.
        let mut graph = MeshGraph::<Point2<f32>>::from_raw_buffers_with_arity(
            vec![0u32, 1, 2, 3, 0, 3, 4, 5],
            vec![
                (0.0, 0.0),  // 0
                (1.0, 0.0),  // 1
                (1.0, 1.0),  // 2
                (0.0, 1.0),  // 3
                (-1.0, 1.0), // 4
                (-1.0, 0.0), // 5
            ],
            4,
        )
        .unwrap();

        // The graph should begin with 2 faces.
        assert_eq!(2, graph.face_count());

        // Remove the edge joining the quadrilaterals from the graph.
        let ab = find_arc_with_vertex_geometry(&graph, ((0.0, 0.0), (0.0, 1.0))).unwrap();
        {
            let vertex = graph.arc_mut(ab).unwrap().remove().unwrap().into_ref();

            // The ring should be formed from 6 edges.
            assert_eq!(6, vertex.into_outgoing_arc().into_ring().arity());
        }

        // After the removal, the graph should have no faces.
        assert_eq!(0, graph.face_count());
    }
}
