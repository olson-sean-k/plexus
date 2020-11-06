use arrayvec::ArrayVec;
use derivative::Derivative;
use fool::BoolExt;
use slotmap::DefaultKey;
use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::{Deref, DerefMut};
use theon::space::{EuclideanSpace, Scalar, Vector};
use theon::{AsPosition, AsPositionMut};

use crate::entity::borrow::{Reborrow, ReborrowInto, ReborrowMut};
use crate::entity::storage::{AsStorage, AsStorageMut, FnvEntityMap, Key, SlotEntityMap};
use crate::entity::view::{Bind, ClosedView, Orphan, Rebind, Unbind, View};
use crate::entity::{Entity, Payload};
use crate::graph::data::{Data, GraphData, Parametric};
use crate::graph::face::{Face, FaceKey, FaceOrphan, FaceView, Ring};
use crate::graph::geometry::{ArcNormal, EdgeMidpoint, VertexPosition};
use crate::graph::mutation::edge::{
    self, ArcBridgeCache, ArcExtrudeCache, EdgeRemoveCache, EdgeSplitCache,
};
use crate::graph::mutation::{self, Consistent, Immediate, Mutable};
use crate::graph::path::Path;
use crate::graph::vertex::{Vertex, VertexKey, VertexOrphan, VertexView};
use crate::graph::{GraphError, OptionExt as _, ResultExt as _, Selector};
use crate::transact::{BypassOrCommit, Mutate};

type Mutation<M> = mutation::Mutation<Immediate<M>>;

pub trait ToArc<B>: Sized
where
    B: Reborrow,
    B::Target: AsStorage<Arc<Data<B>>> + AsStorage<Edge<Data<B>>> + Consistent + Parametric,
{
    fn into_arc(self) -> ArcView<B>;

    fn arc(&self) -> ArcView<&B::Target>;
}

// Unlike other graph structures, the vertex connectivity of an arc is immutable
// and encoded within its key. This provides fast and reliable lookups even when
// a graph is in an inconsistent state. However, it also complicates certain
// topological mutations and sometimes requires that arcs be rekeyed. For this
// reason, `Arc` has no fields representing its source and destination vertices
// nor its opposite arc; such fields would be redundant.
/// Arc entity.
#[derivative(Clone, Copy, Debug, Hash)]
#[derive(Derivative)]
pub struct Arc<G>
where
    G: GraphData,
{
    /// User data.
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub(in crate) data: G::Arc,
    /// Required key into the next arc.
    pub(in crate) next: Option<ArcKey>,
    /// Required key into the previous arc.
    pub(in crate) previous: Option<ArcKey>,
    /// Required key into the edge.
    pub(in crate) edge: Option<EdgeKey>,
    /// Optional key into the face.
    pub(in crate) face: Option<FaceKey>,
}

impl<G> Arc<G>
where
    G: GraphData,
{
    pub fn new(geometry: G::Arc) -> Self {
        Arc {
            data: geometry,
            next: None,
            previous: None,
            edge: None,
            face: None,
        }
    }
}

impl<G> Entity for Arc<G>
where
    G: GraphData,
{
    type Key = ArcKey;
    type Storage = FnvEntityMap<Self>;
}

impl<G> Payload for Arc<G>
where
    G: GraphData,
{
    type Data = G::Arc;

    fn get(&self) -> &Self::Data {
        &self.data
    }

    fn get_mut(&mut self) -> &mut Self::Data {
        &mut self.data
    }
}

/// Arc key.
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

impl Key for ArcKey {
    type Inner = (VertexKey, VertexKey);

    fn from_inner(key: Self::Inner) -> Self {
        ArcKey(key.0, key.1)
    }

    fn into_inner(self) -> Self::Inner {
        (self.0, self.1)
    }
}

/// View of an arc entity.
///
/// An arc from a vertex $A$ to a vertex $B$ is notated $\overrightarrow{AB}$.
/// This is shorthand for the path notation $\overrightarrow{(A,B)}$.
///
/// Arcs provide the connectivity information within a [`MeshGraph`] and are the
/// primary mechanism for traversing its topology. Moreover, most edge-like
/// operations are exposed by arcs.
///
/// See the [`graph`] module documentation for more information about views.
///
/// # Examples
///
/// Traversing a [`MeshGraph`] of a cube via its arcs to find an opposing face:
///
/// ```rust
/// # extern crate decorum;
/// # extern crate nalgebra;
/// # extern crate plexus;
/// #
/// use decorum::R64;
/// use nalgebra::Point3;
/// use plexus::graph::MeshGraph;
/// use plexus::index::HashIndexer;
/// use plexus::prelude::*;
/// use plexus::primitive::cube::Cube;
/// use plexus::primitive::generate::Position;
///
/// type E3 = Point3<R64>;
///
/// let mut graph: MeshGraph<E3> = Cube::new()
///     .polygons::<Position<E3>>()
///     .collect_with_indexer(HashIndexer::default())
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
///
/// [`MeshGraph`]: crate::graph::MeshGraph
/// [`graph`]: crate::graph
pub struct ArcView<B>
where
    B: Reborrow,
    B::Target: AsStorage<Arc<Data<B>>> + Parametric,
{
    inner: View<B, Arc<Data<B>>>,
}

impl<B, M> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Data<B>>> + Parametric,
{
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
    M: AsStorage<Arc<Data<B>>> + Parametric,
{
    // This function is also used to implement `Ring::to_mut_unchecked`.
    #[allow(clippy::wrong_self_convention)]
    pub(in crate::graph) fn to_mut_unchecked(&mut self) -> ArcView<&mut M> {
        self.inner.to_mut_unchecked().into()
    }
}

impl<'a, B, M, G> ArcView<B>
where
    B: ReborrowInto<'a, Target = M>,
    M: AsStorage<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
{
    // TODO: Relocate this documentation of `into_ref`.
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
    /// let arc = graph
    ///     .arc_mut(key)
    ///     .unwrap()
    ///     .extrude_with_offset(1.0)
    ///     .unwrap()
    ///     .into_ref();
    ///
    /// // This would not be possible without conversion into an immutable view.
    /// let _ = arc.into_next_arc().into_next_arc().into_face();
    /// let _ = arc.into_opposite_arc().into_face();
    /// ```
    pub fn into_ref(self) -> ArcView<&'a M> {
        self.inner.into_ref().into()
    }
}

impl<B, M, G> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
{
    pub fn get<'a>(&'a self) -> &'a G::Arc
    where
        G: 'a,
    {
        self.inner.get()
    }
}

impl<B, M, G> ArcView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorageMut<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
{
    pub fn get_mut<'a>(&'a mut self) -> &'a mut G::Arc
    where
        G: 'a,
    {
        self.inner.get_mut()
    }
}

/// Reachable API.
impl<B, M, G> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
{
    pub(in crate::graph) fn into_reachable_boundary_arc(self) -> Option<Self> {
        if self.is_boundary_arc() {
            Some(self)
        }
        else {
            self.into_reachable_opposite_arc()
                .and_then(|opposite| opposite.is_boundary_arc().then_some_ext(opposite))
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
}

impl<B, M, G> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
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
        self.to_ref().into_boundary_arc()
    }

    /// Gets the opposite arc.
    pub fn opposite_arc(&self) -> ArcView<&M> {
        self.to_ref().into_opposite_arc()
    }

    /// Gets the next arc.
    pub fn next_arc(&self) -> ArcView<&M> {
        self.to_ref().into_next_arc()
    }

    /// Gets the previous arc.
    pub fn previous_arc(&self) -> ArcView<&M> {
        self.to_ref().into_previous_arc()
    }
}

/// Reachable API.
impl<B, M, G> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    pub(in crate::graph) fn into_reachable_source_vertex(self) -> Option<VertexView<B>> {
        let (key, _) = self.key().into();
        self.rebind(key)
    }

    pub(in crate::graph) fn into_reachable_destination_vertex(self) -> Option<VertexView<B>> {
        let (_, key) = self.key().into();
        self.rebind(key)
    }
}

impl<B, M, G> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
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
        self.to_ref().into_source_vertex()
    }

    /// Gets the destination vertex of the arc.
    pub fn destination_vertex(&self) -> VertexView<&M> {
        self.to_ref().into_destination_vertex()
    }
}

/// Reachable API.
impl<B, M, G> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Edge<G>> + Parametric<Data = G>,
    G: GraphData,
{
    pub(in crate::graph) fn into_reachable_edge(self) -> Option<EdgeView<B>> {
        let key = self.edge;
        key.and_then(|key| self.rebind(key))
    }
}

impl<B, M, G> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Edge<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    /// Converts the arc into its edge.
    pub fn into_edge(self) -> EdgeView<B> {
        self.into_reachable_edge().expect_consistent()
    }

    /// Gets the edge of the arc.
    pub fn edge(&self) -> EdgeView<&M> {
        self.to_ref().into_edge()
    }
}

/// Reachable API.
impl<B, M, G> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Face<G>> + Parametric<Data = G>,
    G: GraphData,
{
    pub(in crate::graph) fn into_reachable_face(self) -> Option<FaceView<B>> {
        let key = self.face;
        key.and_then(|key| self.rebind(key))
    }
}

impl<B, M, G> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Face<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
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
        self.to_ref().into_face()
    }
}

impl<B, M, G> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    pub fn normal(&self) -> Vector<VertexPosition<G>>
    where
        G: ArcNormal,
        G::Vertex: AsPosition,
    {
        <G as ArcNormal>::normal(self.to_ref()).expect_consistent()
    }
}

impl<B, M, G> ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>>
        + AsStorage<Edge<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Parametric<Data = G>,
    G: GraphData,
{
    pub fn midpoint(&self) -> VertexPosition<G>
    where
        G: EdgeMidpoint,
        G::Vertex: AsPosition,
    {
        G::midpoint(self.to_ref()).expect_consistent()
    }
}

impl<'a, B, M, G> ArcView<B>
where
    B: ReborrowInto<'a, Target = M>,
    M: 'a + AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    pub fn into_adjacent_vertices(
        self,
    ) -> impl Clone + ExactSizeIterator<Item = VertexView<&'a M>> {
        VertexCirculator::from(self.into_ref())
    }
}

impl<B, G> ArcView<B>
where
    B: Reborrow,
    B::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    /// Gets an iterator of views over the vertices connected by the arc.
    pub fn adjacent_vertices(
        &self,
    ) -> impl Clone + ExactSizeIterator<Item = VertexView<&B::Target>> {
        self.to_ref().into_adjacent_vertices()
    }
}

impl<'a, B, M, G> ArcView<B>
where
    B: ReborrowInto<'a, Target = M>,
    M: 'a + AsStorage<Arc<G>> + AsStorage<Face<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    pub fn into_adjacent_faces(self) -> impl Clone + ExactSizeIterator<Item = FaceView<&'a M>> {
        FaceCirculator::from(self.into_ref())
    }
}

impl<B, G> ArcView<B>
where
    B: Reborrow,
    B::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    /// Gets an iterator of views over the faces connected to the arc.
    pub fn adjacent_faces(&self) -> impl Clone + ExactSizeIterator<Item = FaceView<&B::Target>> {
        self.to_ref().into_adjacent_faces()
    }
}

impl<'a, M, G> ArcView<&'a mut M>
where
    M: AsStorage<Arc<G>> + AsStorageMut<Vertex<G>> + Consistent + Parametric<Data = G>,
    G: 'a + GraphData,
{
    pub fn into_adjacent_vertex_orphans(
        self,
    ) -> impl ExactSizeIterator<Item = VertexOrphan<'a, G>> {
        VertexCirculator::from(self)
    }
}

impl<B> ArcView<B>
where
    B: ReborrowMut,
    B::Target: AsStorage<Arc<Data<B>>> + AsStorageMut<Vertex<Data<B>>> + Consistent + Parametric,
{
    /// Gets an iterator of orphan views over the vertices connected by the arc.
    pub fn adjacent_vertex_orphans(
        &mut self,
    ) -> impl ExactSizeIterator<Item = VertexOrphan<Data<B>>> {
        self.to_mut_unchecked().into_adjacent_vertex_orphans()
    }
}

impl<'a, M, G> ArcView<&'a mut M>
where
    M: AsStorage<Arc<G>> + AsStorageMut<Face<G>> + Consistent + Parametric<Data = G>,
    G: 'a + GraphData,
{
    pub fn into_adjacent_face_orphans(self) -> impl ExactSizeIterator<Item = FaceOrphan<'a, G>> {
        FaceCirculator::from(self)
    }
}

impl<B> ArcView<B>
where
    B: ReborrowMut,
    B::Target: AsStorage<Arc<Data<B>>> + AsStorageMut<Face<Data<B>>> + Consistent + Parametric,
{
    /// Gets an iterator of orphan views over the faces connected to the arc.
    pub fn adjacent_face_orphans(&mut self) -> impl ExactSizeIterator<Item = FaceOrphan<Data<B>>> {
        self.to_mut_unchecked().into_adjacent_face_orphans()
    }
}

impl<'a, M, G> ArcView<&'a mut M>
where
    M: AsStorage<Arc<G>>
        + AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Default
        + Mutable<Data = G>,
    G: GraphData,
{
    /// Splits the composite edge of the arc into two adjacent edges that share
    /// a vertex.
    ///
    /// Splitting inserts a new vertex with data provided by the given function.
    /// Splitting an arc $\overrightarrow{AB}$ returns a vertex $M$ that
    /// subdivides the composite edge. The leading arc of $M$ is
    /// $\overrightarrow{MB}$ and is a part of the same ring as the initiating
    /// arc.
    ///
    /// Returns the inserted vertex.
    ///
    /// # Examples
    ///
    /// Splitting an edge in a [`MeshGraph`] with weighted vertices:
    ///
    /// ```rust
    /// use plexus::graph::{GraphData, MeshGraph};
    /// use plexus::prelude::*;
    /// use plexus::primitive::NGon;
    ///
    /// pub enum Weight {}
    ///
    /// impl GraphData for Weight {
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
    ///
    /// [`MeshGraph`]: crate::graph::MeshGraph
    pub fn split_with<F>(self, f: F) -> VertexView<&'a mut M>
    where
        F: FnOnce() -> G::Vertex,
    {
        // This should never fail here.
        let cache = EdgeSplitCache::from_arc(self.to_ref()).expect_consistent();
        let (storage, _) = self.unbind();
        Mutation::replace(storage, Default::default())
            .bypass_or_commit_with(|mutation| edge::split_with(mutation, cache, f))
            .map(|(storage, m)| Bind::bind(storage, m).expect_consistent())
            .map_err(|(_, error)| error)
            .expect_consistent()
    }

    /// Splits the composite edge of the arc at its midpoint.
    ///
    /// Splitting inserts a new vertex with the data of the arc's source vertex
    /// but modified such that the position of the vertex is the computed
    /// midpoint of both of the arc's vertices.
    ///
    /// Splitting inserts a new vertex with data provided by the given function.
    /// Splitting an arc $\overrightarrow{AB}$ returns a vertex $M$ that
    /// subdivides the composite edge. The leading arc of $M$ is
    /// $\overrightarrow{MB}$ and is a part of the same ring as the initiating
    /// arc.
    ///
    /// This function is only available if a [`MeshGraph`] exposes positional
    /// data in its vertices and that data supports interpolation. See the
    /// [`EdgeMidpoint`] trait.
    ///
    /// Returns the inserted vertex.
    ///
    /// # Examples
    ///
    /// Splitting an edge in a triangle at its midpoint:
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
    ///
    /// [`EdgeMidpoint`]: crate::graph::EdgeMidpoint
    /// [`MeshGraph`]: crate::graph::MeshGraph
    pub fn split_at_midpoint(self) -> VertexView<&'a mut M>
    where
        G: EdgeMidpoint,
        G::Vertex: AsPositionMut,
    {
        let mut geometry = self.source_vertex().data;
        let midpoint = self.midpoint();
        self.split_with(move || {
            *geometry.as_position_mut() = midpoint;
            geometry
        })
    }

    // TODO: What if an edge in the bridging quadrilateral is collapsed, such as
    //       bridging arcs within a triangular ring? Document these edge cases
    //       (no pun intended).
    /// Connects the arc to another arc by inserting a face.
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
    /// Returns the inserted face.
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
    /// use plexus::geometry::FromGeometry;
    /// use plexus::graph::{GraphData, MeshGraph, VertexKey, VertexView};
    /// use plexus::prelude::*;
    /// use plexus::primitive::NGon;
    ///
    /// fn find<'a, I, T, G>(input: I, data: T) -> Option<VertexKey>
    /// where
    ///     I: IntoIterator<Item = VertexView<&'a MeshGraph<G>>>,
    ///     G: 'a + GraphData,
    ///     G::Vertex: FromGeometry<T> + PartialEq,
    /// {
    ///     let data = data.into_geometry();
    ///     input
    ///         .into_iter()
    ///         .find(|vertex| *vertex.get() == data)
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
                .arcs()
                .nth(index)
                .ok_or_else(|| GraphError::TopologyNotFound)
                .map(|arc| arc.key())
        })?;
        let cache = ArcBridgeCache::from_arc(self.to_ref(), destination)?;
        let (storage, _) = self.unbind();
        Ok(Mutation::replace(storage, Default::default())
            .bypass_or_commit_with(|mutation| edge::bridge(mutation, cache))
            .map(|(storage, face)| Bind::bind(storage, face).expect_consistent())
            .map_err(|(_, error)| error)
            .expect_consistent())
    }

    /// Extrudes the arc along its normal.
    ///
    /// The positions of each extruded vertex are translated along the arc's
    /// normal by the given offset.
    ///
    /// An arc's normal is perpendicular to the arc and also coplanar with the
    /// arc and one of its adjacent arcs. This is computed via a projection and
    /// supports both 2D and 3D geometries.
    ///
    /// Returns the extruded arc, which is in the same ring as the initiating
    /// arc.
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
    /// graph
    ///     .arc_mut(key)
    ///     .unwrap()
    ///     .extrude_with_offset(1.0)
    ///     .unwrap();
    /// ```
    pub fn extrude_with_offset<T>(self, offset: T) -> Result<ArcView<&'a mut M>, GraphError>
    where
        T: Into<Scalar<VertexPosition<G>>>,
        G: ArcNormal,
        G::Vertex: AsPositionMut,
        VertexPosition<G>: EuclideanSpace,
    {
        let translation = self.normal() * offset.into();
        self.extrude_with_translation(translation)
    }

    /// Extrudes the arc along a translation.
    ///
    /// The positions of each extruded vertex are translated by the given
    /// vector.
    ///
    /// Returns the extruded arc, which is in the same ring as the initiating
    /// arc.
    ///
    /// # Errors
    ///
    /// Returns an error if the arc is not a boundary arc.
    pub fn extrude_with_translation(
        self,
        translation: Vector<VertexPosition<G>>,
    ) -> Result<ArcView<&'a mut M>, GraphError>
    where
        G::Vertex: AsPositionMut,
        VertexPosition<G>: EuclideanSpace,
    {
        self.extrude_with(|geometry| geometry.map_position(|position| *position + translation))
    }

    /// Extrudes the arc using the given vertex data.
    ///
    /// The data of each extruded vertex is determined by the given function,
    /// which maps the data of each source vertex into the data of the
    /// corresponding destination vertex.
    ///
    /// Returns the extruded arc, which is in the same ring as the initiating
    /// arc.
    ///
    /// # Errors
    ///
    /// Returns an error if the arc is not a boundary arc.
    pub fn extrude_with<F>(self, f: F) -> Result<ArcView<&'a mut M>, GraphError>
    where
        F: Fn(G::Vertex) -> G::Vertex,
    {
        let cache = ArcExtrudeCache::from_arc(self.to_ref())?;
        let (storage, _) = self.unbind();
        Ok(Mutation::replace(storage, Default::default())
            .bypass_or_commit_with(|mutation| edge::extrude_with(mutation, cache, f))
            .map(|(storage, arc)| Bind::bind(storage, arc).expect_consistent())
            .map_err(|(_, error)| error)
            .expect_consistent())
    }

    /// Removes the arc and its composite edge.
    ///
    /// Any and all dependent entities are also removed, such as connected
    /// faces, disjoint vertices, etc.
    ///
    /// Returns the source vertex of the initiating arc or `None` if that vertex
    /// becomes disjoint and is also removed. If an arc $\overrightarrow{AB}$ is
    /// removed and its source vertex is not disjoint, then $A$ is returned.
    pub fn remove(self) -> Option<VertexView<&'a mut M>> {
        let a = self.source_vertex().key();
        // This should never fail here.
        let cache = EdgeRemoveCache::from_arc(self.to_ref()).expect_consistent();
        let (storage, _) = self.unbind();
        Mutation::replace(storage, Default::default())
            .bypass_or_commit_with(|mutation| edge::remove(mutation, cache))
            .map(|(storage, _)| Bind::bind(storage, a))
            .map_err(|(_, error)| error)
            .expect_consistent()
    }
}

impl<B> Borrow<ArcKey> for ArcView<B>
where
    B: Reborrow,
    B::Target: AsStorage<Arc<Data<B>>> + Parametric,
{
    fn borrow(&self) -> &ArcKey {
        self.inner.as_ref()
    }
}

impl<B, M, G> Clone for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
    View<B, Arc<G>>: Clone,
{
    fn clone(&self) -> Self {
        ArcView {
            inner: self.inner.clone(),
        }
    }
}

impl<B, M, G> ClosedView for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
{
    type Key = ArcKey;
    type Entity = Arc<G>;

    /// Gets the key for the arc.
    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<B, M, G> Copy for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
    View<B, Arc<G>>: Copy,
{
}

impl<B, M, G> Deref for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
{
    type Target = Arc<G>;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl<B, M, G> DerefMut for ArcView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorageMut<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

impl<B, M, G> Eq for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
{
}

impl<B, M, G> From<Ring<B>> for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    fn from(ring: Ring<B>) -> Self {
        ring.into_arc()
    }
}

impl<B, M, G> From<View<B, Arc<G>>> for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn from(view: View<B, Arc<G>>) -> Self {
        ArcView { inner: view }
    }
}

impl<B, M, G> Hash for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.inner.hash(state);
    }
}

impl<B, M, G> Into<View<B, Arc<G>>> for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn into(self) -> View<B, Arc<G>> {
        let ArcView { inner, .. } = self;
        inner
    }
}

impl<B, M, G> PartialEq for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<B, M, G> ToArc<B> for ArcView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Edge<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    fn into_arc(self) -> ArcView<B> {
        self
    }

    fn arc(&self) -> ArcView<&M> {
        self.to_ref()
    }
}

/// Orphan view of an arc entity.
pub struct ArcOrphan<'a, G>
where
    G: GraphData,
{
    inner: Orphan<'a, Arc<G>>,
}

impl<'a, G> ArcOrphan<'a, G>
where
    G: 'a + GraphData,
{
    pub fn get(&self) -> &G::Arc {
        self.inner.get()
    }

    pub fn get_mut(&mut self) -> &mut G::Arc {
        self.inner.get_mut()
    }
}

impl<'a, G> Borrow<ArcKey> for ArcOrphan<'a, G>
where
    G: GraphData,
{
    fn borrow(&self) -> &ArcKey {
        self.inner.as_ref()
    }
}

impl<'a, G> ClosedView for ArcOrphan<'a, G>
where
    G: GraphData,
{
    type Key = ArcKey;
    type Entity = Arc<G>;

    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<'a, G> Eq for ArcOrphan<'a, G> where G: GraphData {}

impl<'a, M, G> From<ArcView<&'a mut M>> for ArcOrphan<'a, G>
where
    M: AsStorageMut<Arc<G>> + Parametric<Data = G>,
    G: 'a + GraphData,
{
    fn from(arc: ArcView<&'a mut M>) -> Self {
        Orphan::from(arc.inner).into()
    }
}

impl<'a, G> From<Orphan<'a, Arc<G>>> for ArcOrphan<'a, G>
where
    G: GraphData,
{
    fn from(inner: Orphan<'a, Arc<G>>) -> Self {
        ArcOrphan { inner }
    }
}

impl<'a, M, G> From<View<&'a mut M, Arc<G>>> for ArcOrphan<'a, G>
where
    M: AsStorageMut<Arc<G>> + Parametric<Data = G>,
    G: 'a + GraphData,
{
    fn from(view: View<&'a mut M, Arc<G>>) -> Self {
        ArcOrphan { inner: view.into() }
    }
}

impl<'a, G> Hash for ArcOrphan<'a, G>
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

impl<'a, G> PartialEq for ArcOrphan<'a, G>
where
    G: GraphData,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Edge entity.
#[derivative(Clone, Copy, Debug, Hash)]
#[derive(Derivative)]
pub struct Edge<G>
where
    G: GraphData,
{
    /// User data.
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub(in crate) data: G::Edge,
    /// Required key into the leading arc.
    pub(in crate) arc: ArcKey,
}

impl<G> Edge<G>
where
    G: GraphData,
{
    pub fn new(arc: ArcKey, geometry: G::Edge) -> Self {
        Edge {
            data: geometry,
            arc,
        }
    }
}

impl<G> Entity for Edge<G>
where
    G: GraphData,
{
    type Key = EdgeKey;
    type Storage = SlotEntityMap<Self>;
}

impl<G> Payload for Edge<G>
where
    G: GraphData,
{
    type Data = G::Edge;

    fn get(&self) -> &Self::Data {
        &self.data
    }

    fn get_mut(&mut self) -> &mut Self::Data {
        &mut self.data
    }
}

/// Edge key.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EdgeKey(DefaultKey);

impl Key for EdgeKey {
    type Inner = DefaultKey;

    fn from_inner(key: Self::Inner) -> Self {
        EdgeKey(key)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

/// View of an edge entity.
///
/// An edge connecting a vertex $A$ and a vertex $B$ is notated
/// $\overleftrightarrow{AB}$ or $\overleftrightarrow{BA}$.
///
/// See the [`graph`] module documentation for more information about views.
///
/// [`graph`]: crate::graph
pub struct EdgeView<B>
where
    B: Reborrow,
    B::Target: AsStorage<Edge<<B::Target as Parametric>::Data>> + Parametric,
{
    inner: View<B, Edge<<B::Target as Parametric>::Data>>,
}

impl<B, M> EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<Data<B>>> + Parametric,
{
    pub fn to_ref(&self) -> EdgeView<&M> {
        self.inner.to_ref().into()
    }
}

impl<'a, B, M, G> EdgeView<B>
where
    B: ReborrowInto<'a, Target = M>,
    M: AsStorage<Edge<G>> + Parametric<Data = G>,
    G: GraphData,
{
    pub fn into_ref(self) -> EdgeView<&'a M> {
        self.inner.into_ref().into()
    }
}

impl<B, M, G> EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Parametric<Data = G>,
    G: GraphData,
{
    pub fn get<'a>(&'a self) -> &'a G::Edge
    where
        G: 'a,
    {
        self.inner.get()
    }
}

impl<B, M, G> EdgeView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorageMut<Edge<G>> + Parametric<Data = G>,
    G: GraphData,
{
    pub fn get_mut<'a>(&'a mut self) -> &'a mut G::Edge
    where
        G: 'a,
    {
        self.inner.get_mut()
    }
}

/// Reachable API.
impl<B, M, G> EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Edge<G>> + Parametric<Data = G>,
    G: GraphData,
{
    pub(in crate::graph) fn into_reachable_arc(self) -> Option<ArcView<B>> {
        let key = self.arc;
        self.rebind(key)
    }
}

impl<B, M, G> EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Edge<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    pub fn into_arc(self) -> ArcView<B> {
        self.into_reachable_arc().expect_consistent()
    }

    pub fn arc(&self) -> ArcView<&M> {
        self.to_ref().into_arc()
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
        + Parametric<Data = G>,
    G: GraphData,
{
    pub fn midpoint(&self) -> VertexPosition<G>
    where
        G: EdgeMidpoint,
        G::Vertex: AsPosition,
    {
        G::midpoint(self.to_ref()).expect_consistent()
    }
}

impl<B> Borrow<EdgeKey> for EdgeView<B>
where
    B: Reborrow,
    B::Target: AsStorage<Edge<Data<B>>> + Parametric,
{
    fn borrow(&self) -> &EdgeKey {
        self.inner.as_ref()
    }
}

impl<B, M, G> Clone for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Parametric<Data = G>,
    G: GraphData,
    View<B, Edge<G>>: Clone,
{
    fn clone(&self) -> Self {
        EdgeView {
            inner: self.inner.clone(),
        }
    }
}

impl<B, M, G> ClosedView for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Parametric<Data = G>,
    G: GraphData,
{
    type Key = EdgeKey;
    type Entity = Edge<G>;

    /// Gets the key for the edge.
    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<B, M, G> Copy for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Parametric<Data = G>,
    G: GraphData,
    View<B, Edge<G>>: Copy,
{
}

impl<B, M, G> Deref for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Parametric<Data = G>,
    G: GraphData,
{
    type Target = Edge<G>;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl<B, M, G> DerefMut for EdgeView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorageMut<Edge<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

impl<B, M, G> Eq for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Parametric<Data = G>,
    G: GraphData,
{
}

impl<B, M, G> From<View<B, Edge<G>>> for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn from(view: View<B, Edge<G>>) -> Self {
        EdgeView { inner: view }
    }
}

impl<B, M, G> Hash for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.inner.hash(state);
    }
}

impl<B, M, G> Into<View<B, Edge<G>>> for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn into(self) -> View<B, Edge<G>> {
        let EdgeView { inner, .. } = self;
        inner
    }
}

impl<B, M, G> PartialEq for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Edge<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<B, M, G> ToArc<B> for EdgeView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Edge<G>> + Consistent + Parametric<Data = G>,
    G: GraphData,
{
    fn into_arc(self) -> ArcView<B> {
        EdgeView::into_arc(self)
    }

    fn arc(&self) -> ArcView<&M> {
        EdgeView::arc(self)
    }
}

/// Orphan view of an edge entity.
pub struct EdgeOrphan<'a, G>
where
    G: GraphData,
{
    inner: Orphan<'a, Edge<G>>,
}

impl<'a, G> EdgeOrphan<'a, G>
where
    G: 'a + GraphData,
{
    pub fn get(&self) -> &G::Edge {
        self.inner.get()
    }

    pub fn get_mut(&mut self) -> &mut G::Edge {
        self.inner.get_mut()
    }
}

impl<'a, G> Borrow<EdgeKey> for EdgeOrphan<'a, G>
where
    G: GraphData,
{
    fn borrow(&self) -> &EdgeKey {
        self.inner.as_ref()
    }
}

impl<'a, G> ClosedView for EdgeOrphan<'a, G>
where
    G: GraphData,
{
    type Key = EdgeKey;
    type Entity = Edge<G>;

    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<'a, G> Eq for EdgeOrphan<'a, G> where G: GraphData {}

impl<'a, M, G> From<EdgeView<&'a mut M>> for EdgeOrphan<'a, G>
where
    M: AsStorageMut<Edge<G>> + Parametric<Data = G>,
    G: 'a + GraphData,
{
    fn from(edge: EdgeView<&'a mut M>) -> Self {
        Orphan::from(edge.inner).into()
    }
}

impl<'a, G> From<Orphan<'a, Edge<G>>> for EdgeOrphan<'a, G>
where
    G: GraphData,
{
    fn from(inner: Orphan<'a, Edge<G>>) -> Self {
        EdgeOrphan { inner }
    }
}

impl<'a, M, G> From<View<&'a mut M, Edge<G>>> for EdgeOrphan<'a, G>
where
    M: AsStorageMut<Edge<G>> + Parametric<Data = G>,
    G: 'a + GraphData,
{
    fn from(view: View<&'a mut M, Edge<G>>) -> Self {
        EdgeOrphan { inner: view.into() }
    }
}

impl<'a, G> Hash for EdgeOrphan<'a, G>
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

impl<'a, G> PartialEq for EdgeOrphan<'a, G>
where
    G: GraphData,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

pub struct VertexCirculator<B>
where
    B: Reborrow,
    B::Target: AsStorage<Vertex<<B::Target as Parametric>::Data>> + Parametric,
{
    storage: B,
    inner: <ArrayVec<[VertexKey; 2]> as IntoIterator>::IntoIter,
}

impl<B, M, G> VertexCirculator<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn next(&mut self) -> Option<VertexKey> {
        self.inner.next()
    }
}

impl<B, M, G> From<ArcView<B>> for VertexCirculator<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
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
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
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
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
{
}

impl<'a, M, G> Iterator for VertexCirculator<&'a M>
where
    M: AsStorage<Vertex<G>> + Parametric<Data = G>,
    G: GraphData,
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
    M: AsStorageMut<Vertex<G>> + Parametric<Data = G>,
    G: 'a + GraphData,
{
    type Item = VertexOrphan<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).map(|key| {
            let vertex = self.storage.as_storage_mut().get_mut(&key).unwrap();
            let data = &mut vertex.data;
            let data = unsafe { mem::transmute::<&'_ mut G::Vertex, &'a mut G::Vertex>(data) };
            Orphan::bind_unchecked(data, key).into()
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.inner.len()))
    }
}

pub struct FaceCirculator<B>
where
    B: Reborrow,
    B::Target: AsStorage<Face<<B::Target as Parametric>::Data>> + Parametric,
{
    storage: B,
    inner: <ArrayVec<[FaceKey; 2]> as IntoIterator>::IntoIter,
}

impl<B, M, G> FaceCirculator<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Face<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn next(&mut self) -> Option<FaceKey> {
        self.inner.next()
    }
}

impl<B, M, G> Clone for FaceCirculator<B>
where
    B: Clone + Reborrow<Target = M>,
    M: AsStorage<Face<G>> + Parametric<Data = G>,
    G: GraphData,
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
    M: AsStorage<Face<G>> + Parametric<Data = G>,
    G: GraphData,
{
}

impl<B, M, G> From<ArcView<B>> for FaceCirculator<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Face<G>> + Parametric<Data = G>,
    G: GraphData,
{
    fn from(arc: ArcView<B>) -> Self {
        let inner = arc
            .face
            .into_iter()
            .chain(
                arc.to_ref()
                    .into_reachable_opposite_arc()
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
    M: AsStorage<Face<G>> + Parametric<Data = G>,
    G: GraphData,
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
    M: AsStorageMut<Face<G>> + Parametric<Data = G>,
    G: 'a + GraphData,
{
    type Item = FaceOrphan<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).map(|key| {
            let face = self.storage.as_storage_mut().get_mut(&key).unwrap();
            let data = &mut face.data;
            let data = unsafe { mem::transmute::<&'_ mut G::Face, &'a mut G::Face>(data) };
            Orphan::bind_unchecked(data, key).into()
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.inner.len()))
    }
}

#[cfg(test)]
mod tests {
    use decorum::R64;
    use nalgebra::{Point2, Point3};

    use crate::geometry::FromGeometry;
    use crate::graph::{ArcKey, GraphData, MeshGraph};
    use crate::index::HashIndexer;
    use crate::prelude::*;
    use crate::primitive::cube::Cube;
    use crate::primitive::generate::Position;
    use crate::primitive::Tetragon;

    type E2 = Point2<R64>;
    type E3 = Point3<R64>;

    fn find_arc<G, T>(graph: &MeshGraph<G>, data: (T, T)) -> Option<ArcKey>
    where
        G: GraphData,
        G::Vertex: FromGeometry<T> + PartialEq,
    {
        let (source, destination) = data;
        let source = source.into_geometry();
        let destination = destination.into_geometry();
        graph
            .vertices()
            .filter(|vertex| vertex.data == source)
            .flat_map(|source| {
                source
                    .adjacent_vertices()
                    .find(|vertex| vertex.data == destination)
                    .map(|destination| (source.key(), destination.key()))
            })
            .map(|(a, b)| (a, b).into())
            .next()
    }

    #[test]
    fn extrude_arc() {
        let mut graph = MeshGraph::<E2>::from_raw_buffers_with_arity(
            vec![0u32, 1, 2, 3],
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            4,
        )
        .unwrap();
        let source = find_arc(&graph, ((1.0, 1.0), (1.0, 0.0))).unwrap();
        graph
            .arc_mut(source)
            .unwrap()
            .extrude_with_offset(1.0)
            .unwrap();

        assert_eq!(14, graph.arc_count());
        assert_eq!(2, graph.face_count());
    }

    #[test]
    fn bridge_arcs() {
        // Construct a mesh with two disjoint quadrilaterals.
        let mut graph = MeshGraph::<E3>::from_raw_buffers_with_arity(
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
        let source = find_arc(&graph, ((-1.0, 1.0, 0.0), (-1.0, 0.0, 0.0))).unwrap();
        let destination = find_arc(&graph, ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0))).unwrap();
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
            .polygons::<Position<E3>>() // 6 quadrilaterals, 24 vertices.
            .index_vertices::<Tetragon<usize>, _>(HashIndexer::default());
        let mut graph = MeshGraph::<E3>::from_raw_buffers(indices, vertices).unwrap();
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
        let mut graph = MeshGraph::<E2>::from_raw_buffers_with_arity(
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
        let ab = find_arc(&graph, ((0.0, 0.0), (0.0, 1.0))).unwrap();
        {
            let vertex = graph.arc_mut(ab).unwrap().remove().unwrap().into_ref();

            // The ring should be formed from 6 edges.
            assert_eq!(6, vertex.into_outgoing_arc().into_ring().arity());
        }

        // After the removal, the graph should have no faces.
        assert_eq!(0, graph.face_count());
    }
}
