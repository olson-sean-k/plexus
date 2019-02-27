use arrayvec::ArrayVec;
use either::Either;
use fool::prelude::*;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Add, Deref, DerefMut, Mul};

use crate::geometry::alias::{ScaledArcNormal, VertexPosition};
use crate::geometry::convert::AsPosition;
use crate::geometry::Geometry;
use crate::graph::container::{Bind, Consistent, Reborrow, ReborrowMut};
use crate::graph::geometry::{ArcNormal, EdgeMidpoint};
use crate::graph::mutation::alias::Mutable;
use crate::graph::mutation::edge::{
    self, ArcBridgeCache, ArcExtrudeCache, EdgeRemoveCache, EdgeSplitCache,
};
use crate::graph::mutation::{Mutate, Mutation};
use crate::graph::payload::{ArcPayload, EdgePayload, FacePayload, Payload, VertexPayload};
use crate::graph::storage::convert::{AsStorage, AsStorageMut};
use crate::graph::storage::{ArcKey, EdgeKey, FaceKey, Storage, VertexKey};
use crate::graph::view::convert::{FromKeyedSource, IntoKeyedSource, IntoView};
use crate::graph::view::{
    FaceView, InteriorPathView, OrphanFaceView, OrphanVertexView, VertexView,
};
use crate::graph::{GraphError, OptionExt, ResultExt, Selector};

/// View of an arc.
///
/// Provides traversals, queries, and mutations related to arcs in a graph. See
/// the module documentation for more information about topological views.
///
/// Arcs provide the connectivity information within a `MeshGraph` and are the
/// primary mechanism for traversing its topology. Moreover, most edge-like
/// operations are exposed by arcs, because they are directed and therefore can
/// emit deterministic results (unlike true edges).
///
/// # Examples
///
/// Traversing a graph of a cube via its arcs to find an opposing face:
///
/// ```rust
/// use plexus::graph::MeshGraph;
/// use plexus::prelude::*;
/// use plexus::primitive::cube::Cube;
/// use plexus::primitive::index::HashIndexer;
///
/// let mut graph = Cube::new()
///     .polygons_with_position()
///     .collect_with_indexer::<MeshGraph<Triplet<_>>, _>(HashIndexer::default())
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
pub struct ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>>,
    G: Geometry,
{
    key: ArcKey,
    storage: M,
    phantom: PhantomData<G>,
}

/// Storage.
impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>>,
    G: Geometry,
{
    // TODO: This may become useful as the `mutation` module is developed. It
    //       may also be necessary to expose this API to user code.
    #[allow(dead_code)]
    pub(in crate::graph) fn bind<T, N>(self, storage: N) -> ArcView<<M as Bind<T, N>>::Output, G>
    where
        T: Payload,
        M: Bind<T, N>,
        M::Output: Reborrow,
        <M::Output as Reborrow>::Target: AsStorage<ArcPayload<G>>,
        N: AsStorage<T>,
    {
        let (key, origin) = self.into_keyed_source();
        ArcView::from_keyed_source_unchecked((key, origin.bind(storage)))
    }
}

impl<'a, M, G> ArcView<&'a mut M, G>
where
    M: 'a + AsStorage<ArcPayload<G>> + AsStorageMut<ArcPayload<G>>,
    G: 'a + Geometry,
{
    /// Converts a mutable view into an orphan view.
    pub fn into_orphan(self) -> OrphanArcView<'a, G> {
        let (key, storage) = self.into_keyed_source();
        (key, storage).into_view().unwrap()
    }

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
    /// use nalgebra::Point2;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    ///
    /// # fn main() {
    /// let mut graph = MeshGraph::<Point2<f32>>::from_raw_buffers_with_arity(
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
    /// # }
    /// ```
    pub fn into_ref(self) -> ArcView<&'a M, G> {
        let (key, storage) = self.into_keyed_source();
        ArcView::from_keyed_source_unchecked((key, &*storage))
    }

    /// Reborrows the view and constructs another mutable view from a given
    /// key.
    ///
    /// This allows for fallible traversals from a mutable view without the
    /// need for direct access to the source `MeshGraph`. If the given function
    /// emits a key, then that key will be used to convert this view into
    /// another. If no key is emitted, then the original mutable view is
    /// returned.
    pub fn with_ref<T, K, F>(self, f: F) -> Either<Result<T, GraphError>, Self>
    where
        T: FromKeyedSource<(K, &'a mut M)>,
        F: FnOnce(ArcView<&M, G>) -> Option<K>,
    {
        if let Some(key) = f(self.interior_reborrow()) {
            let (_, storage) = self.into_keyed_source();
            Either::Left(
                T::from_keyed_source((key, storage)).ok_or_else(|| GraphError::TopologyNotFound),
            )
        }
        else {
            Either::Right(self)
        }
    }
}

impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>>,
    G: Geometry,
{
    /// Gets the key for the arc.
    pub fn key(&self) -> ArcKey {
        self.key
    }

    /// Returns true if this is a boundary arc.
    ///
    /// A boundary arc has no associated face.
    pub fn is_boundary_arc(&self) -> bool {
        self.face.is_none()
    }

    fn from_keyed_source_unchecked(source: (ArcKey, M)) -> Self {
        let (key, storage) = source;
        ArcView {
            key,
            storage,
            phantom: PhantomData,
        }
    }

    fn interior_reborrow(&self) -> ArcView<&M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow();
        ArcView::from_keyed_source_unchecked((key, storage))
    }
}

impl<M, G> ArcView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<ArcPayload<G>>,
    G: Geometry,
{
    fn interior_reborrow_mut(&mut self) -> ArcView<&mut M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow_mut();
        ArcView::from_keyed_source_unchecked((key, storage))
    }
}

/// Reachable API.
impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>>,
    G: Geometry,
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
        let (key, storage) = self.into_keyed_source();
        (key.opposite(), storage).into_view()
    }

    pub(in crate::graph) fn into_reachable_next_arc(self) -> Option<Self> {
        let key = self.next;
        key.and_then(move |key| {
            let (_, storage) = self.into_keyed_source();
            (key, storage).into_view()
        })
    }

    pub(in crate::graph) fn into_reachable_previous_arc(self) -> Option<Self> {
        let key = self.previous;
        key.and_then(move |key| {
            let (_, storage) = self.into_keyed_source();
            (key, storage).into_view()
        })
    }

    pub(in crate::graph) fn reachable_boundary_arc(&self) -> Option<ArcView<&M::Target, G>> {
        if self.is_boundary_arc() {
            Some(self.interior_reborrow())
        }
        else {
            self.reachable_opposite_arc()
                .and_then(|opposite| opposite.is_boundary_arc().some_with(|| opposite))
        }
    }

    pub(in crate::graph) fn reachable_opposite_arc(&self) -> Option<ArcView<&M::Target, G>> {
        let ba = self.key.opposite();
        let storage = self.storage.reborrow();
        (ba, storage).into_view()
    }

    pub(in crate::graph) fn reachable_next_arc(&self) -> Option<ArcView<&M::Target, G>> {
        self.next.and_then(|key| {
            let storage = self.storage.reborrow();
            (key, storage).into_view()
        })
    }

    pub(in crate::graph) fn reachable_previous_arc(&self) -> Option<ArcView<&M::Target, G>> {
        self.previous.and_then(|key| {
            let storage = self.storage.reborrow();
            (key, storage).into_view()
        })
    }
}

impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + Consistent,
    G: Geometry,
{
    /// Converts the arc into its interior path.
    pub fn into_interior_path(self) -> InteriorPathView<M, G> {
        let (key, storage) = self.into_keyed_source();
        (key, storage).into_view().expect_consistent()
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

    /// Gets the interior path of the arc.
    pub fn interior_path(&self) -> InteriorPathView<&M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow();
        (key, storage).into_view().expect_consistent()
    }

    /// Returns the same arc if it is a boundary arc, otherwise `None`.
    pub fn boundary_arc(&self) -> Option<ArcView<&M::Target, G>> {
        self.reachable_boundary_arc()
    }

    /// Gets the opposite arc.
    pub fn opposite_arc(&self) -> ArcView<&M::Target, G> {
        self.reachable_opposite_arc().expect_consistent()
    }

    /// Gets the next arc.
    pub fn next_arc(&self) -> ArcView<&M::Target, G> {
        self.reachable_next_arc().expect_consistent()
    }

    /// Gets the previous arc.
    pub fn previous_arc(&self) -> ArcView<&M::Target, G> {
        self.reachable_previous_arc().expect_consistent()
    }
}

/// Reachable API.
impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<VertexPayload<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn into_reachable_source_vertex(self) -> Option<VertexView<M, G>> {
        let (a, _) = self.key.into();
        let (_, storage) = self.into_keyed_source();
        (a, storage).into_view()
    }

    pub(in crate::graph) fn into_reachable_destination_vertex(self) -> Option<VertexView<M, G>> {
        let (_, b) = self.key.into();
        let (_, storage) = self.into_keyed_source();
        (b, storage).into_view()
    }

    pub(in crate::graph) fn reachable_source_vertex(&self) -> Option<VertexView<&M::Target, G>> {
        let (a, _) = self.key.into();
        let storage = self.storage.reborrow();
        (a, storage).into_view()
    }

    pub(in crate::graph) fn reachable_destination_vertex(
        &self,
    ) -> Option<VertexView<&M::Target, G>> {
        let (_, b) = self.key.into();
        let storage = self.storage.reborrow();
        (b, storage).into_view()
    }
}

impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<VertexPayload<G>> + Consistent,
    G: Geometry,
{
    /// Gets the nieghborhood of the arc.
    pub fn neighborhood(&self) -> ArcNeighborhood {
        ArcNeighborhood::from(self.interior_reborrow())
    }

    /// Converts the arc into its source vertex.
    pub fn into_source_vertex(self) -> VertexView<M, G> {
        self.into_reachable_source_vertex().expect_consistent()
    }

    /// Converts the arc into its destination vertex.
    pub fn into_destination_vertex(self) -> VertexView<M, G> {
        self.into_reachable_destination_vertex().expect_consistent()
    }

    /// Gets the source vertex of the arc.
    pub fn source_vertex(&self) -> VertexView<&M::Target, G> {
        self.reachable_source_vertex().expect_consistent()
    }

    /// Gets the destination vertex of the arc.
    pub fn destination_vertex(&self) -> VertexView<&M::Target, G> {
        self.reachable_destination_vertex().expect_consistent()
    }
}

/// Reachable API.
impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<EdgePayload<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn into_reachable_edge(self) -> Option<EdgeView<M, G>> {
        let key = self.edge;
        key.and_then(move |key| {
            let (_, storage) = self.into_keyed_source();
            (key, storage).into_view()
        })
    }

    pub(in crate::graph) fn reachable_edge(&self) -> Option<EdgeView<&M::Target, G>> {
        self.edge.and_then(|key| {
            let storage = self.storage.reborrow();
            (key, storage).into_view()
        })
    }
}

impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<EdgePayload<G>> + Consistent,
    G: Geometry,
{
    /// Converts the arc into its edge.
    pub fn into_edge(self) -> EdgeView<M, G> {
        self.into_reachable_edge().expect_consistent()
    }

    /// Gets the edge of the arc.
    pub fn edge(&self) -> EdgeView<&M::Target, G> {
        self.reachable_edge().expect_consistent()
    }
}

/// Reachable API.
impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<FacePayload<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn into_reachable_face(self) -> Option<FaceView<M, G>> {
        let key = self.face;
        key.and_then(move |key| {
            let (_, storage) = self.into_keyed_source();
            (key, storage).into_view()
        })
    }

    pub(in crate::graph) fn reachable_face(&self) -> Option<FaceView<&M::Target, G>> {
        self.face.and_then(|key| {
            let storage = self.storage.reborrow();
            (key, storage).into_view()
        })
    }
}

impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<FacePayload<G>> + Consistent,
    G: Geometry,
{
    /// Converts the arc into its face.
    ///
    /// If this is a boundary arc, then `None` is returned.
    pub fn into_face(self) -> Option<FaceView<M, G>> {
        self.into_reachable_face()
    }

    /// Gets the face of this arc.
    ///
    /// If this is a boundary arc, then `None` is returned.
    pub fn face(&self) -> Option<FaceView<&M::Target, G>> {
        self.reachable_face()
    }
}

/// Reachable API.
impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<VertexPayload<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_vertices(
        &self,
    ) -> impl Clone + Iterator<Item = VertexView<&M::Target, G>> {
        VertexCirculator::from(self.interior_reborrow())
    }
}

impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<VertexPayload<G>> + Consistent,
    G: Geometry,
{
    /// Gets an iterator of views over the vertices connected by the arc.
    pub fn vertices(&self) -> impl Clone + Iterator<Item = VertexView<&M::Target, G>> {
        self.reachable_vertices()
    }
}

/// Reachable API.
impl<M, G> ArcView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target:
        AsStorage<ArcPayload<G>> + AsStorage<VertexPayload<G>> + AsStorageMut<VertexPayload<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_orphan_vertices(
        &mut self,
    ) -> impl Iterator<Item = OrphanVertexView<G>> {
        VertexCirculator::from(self.interior_reborrow_mut())
    }
}

impl<M, G> ArcView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<ArcPayload<G>>
        + AsStorage<VertexPayload<G>>
        + AsStorageMut<VertexPayload<G>>
        + Consistent,
    G: Geometry,
{
    /// Gets an iterator of orphan views over the vertices connected by the
    /// arc.
    pub fn orphan_vertices(&mut self) -> impl Iterator<Item = OrphanVertexView<G>> {
        self.reachable_orphan_vertices()
    }
}

/// Reachable API.
impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<FacePayload<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_faces(
        &self,
    ) -> impl Clone + Iterator<Item = FaceView<&M::Target, G>> {
        FaceCirculator::from(self.interior_reborrow())
    }
}

impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<FacePayload<G>> + Consistent,
    G: Geometry,
{
    /// Gets an iterator of views over the faces connected to the arc.
    pub fn faces(&self) -> impl Clone + Iterator<Item = FaceView<&M::Target, G>> {
        self.reachable_faces()
    }
}

/// Reachable API.
impl<M, G> ArcView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<FacePayload<G>> + AsStorageMut<FacePayload<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_orphan_faces(
        &mut self,
    ) -> impl Iterator<Item = OrphanFaceView<G>> {
        FaceCirculator::from(self.interior_reborrow_mut())
    }
}

impl<M, G> ArcView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<ArcPayload<G>>
        + AsStorage<FacePayload<G>>
        + AsStorageMut<FacePayload<G>>
        + Consistent,
    G: Geometry,
{
    /// Gets an iterator of orphan views over the faces connected to the arc.
    pub fn orphan_faces(&mut self) -> impl Iterator<Item = OrphanFaceView<G>> {
        self.reachable_orphan_faces()
    }
}

impl<'a, M, G> ArcView<&'a mut M, G>
where
    M: AsStorage<ArcPayload<G>>
        + AsStorage<EdgePayload<G>>
        + AsStorage<FacePayload<G>>
        + AsStorage<VertexPayload<G>>
        + Default
        + Mutable<G>,
    G: 'a + Geometry,
{
    /// Splits an edge (and its arcs) into two neighboring edges that share a
    /// vertex.
    ///
    /// Splitting inserts a new vertex with the geometry provided by the given
    /// function. The leading arc of the inserted vertex will be one half of
    /// the arc that initiated the split and therefore will be part of the same
    /// interior path.
    ///
    /// Returns the inserted vertex.
    ///
    /// # Examples
    ///
    /// Split an edge in a graph with weighted vertices:
    ///
    /// ```rust
    /// use plexus::geometry::Geometry;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::Triangle;
    ///
    /// pub enum Weight {}
    ///
    /// impl Geometry for Weight {
    ///     type Vertex = f64;
    ///     type Arc = ();
    ///     type Edge = ();
    ///     type Face = ();
    /// }
    ///
    /// let mut graph = MeshGraph::<Weight>::from_raw_buffers(
    ///     vec![Triangle::new(0usize, 1, 2)],
    ///     vec![1.0, 2.0, 0.5],
    /// )
    /// .unwrap();
    /// let key = graph.arcs().nth(0).unwrap().key();
    /// let vertex = graph.arc_mut(key).unwrap().split_with(|| 0.1);
    /// ```
    pub fn split_with<F>(self, f: F) -> VertexView<&'a mut M, G>
    where
        F: FnOnce() -> G::Vertex,
    {
        (move || {
            let (ab, storage) = self.into_keyed_source();
            let cache = EdgeSplitCache::snapshot(&storage, ab, f())?;
            Mutation::replace(storage, Default::default())
                .commit_with(move |mutation| edge::split_with_cache(mutation, cache))
                .map(|(storage, m)| (m, storage).into_view().expect_consistent())
        })()
        .expect_consistent()
    }

    /// Splits an edge (and its arcs) at the midpoint of the arc's vertices.
    ///
    /// Splitting inserts a new vertex with the geometry of the arc's source
    /// vertex but modified such that the positional data of the vertex is the
    /// computed midpoint of both of the arc's vertices. The leading arc of the
    /// inserted vertex will be one half of the arc that initiated the split
    /// and therefore will be part of the same interior path.
    ///
    /// This function is only available if a graph's geometry exposes
    /// positional data in its vertices and that data supports interpolation.
    /// See the `geometry` module.
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
    /// use nalgebra::Point2;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::Triangle;
    ///
    /// # fn main() {
    /// let mut graph = MeshGraph::<Point2<f64>>::from_raw_buffers(
    ///     vec![Triangle::new(0usize, 1, 2)],
    ///     vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
    /// )
    /// .unwrap();
    /// let key = graph.arcs().nth(0).unwrap().key();
    /// let vertex = graph.arc_mut(key).unwrap().split_at_midpoint();
    /// # }
    /// ```
    pub fn split_at_midpoint(self) -> VertexView<&'a mut M, G>
    where
        G: EdgeMidpoint<Midpoint = VertexPosition<G>>,
        G::Vertex: AsPosition,
    {
        let mut geometry = self.source_vertex().geometry.clone();
        let midpoint = self.edge().midpoint();
        self.split_with(move || {
            *geometry.as_position_mut() = midpoint;
            geometry
        })
    }

    // TODO: What if an edge in the bridging quadrilateral is collapsed, such
    //       as bridging arcs within a triangular interior path? Document these
    //       edge cases (no pun intended).
    /// Connects a boundary arc to another boundary arc with a face.
    ///
    /// Bridging arcs inserts a new face and, as needed, new arcs and edges.
    /// The inserted face is always a quadrilateral. The bridged arcs must be
    /// boundary arcs with an orientation that allows them to form an interior
    /// path.
    ///
    /// Arcs can be bridged within an interior path. The destination arc can be
    /// chosen by key or index, where an index selects the nth arc from the
    /// source arc within the interior path.
    ///
    /// Returns the inserted face if successful.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination arc cannot be found, either arc is
    /// not a boundary arc, or the orientation of the destination arc is
    /// incompatible.
    ///
    /// # Examples
    ///
    /// Bridging two disjoint quadrilaterals together:
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// use nalgebra::Point2;
    /// use plexus::geometry::convert::IntoGeometry;
    /// use plexus::geometry::Geometry;
    /// use plexus::graph::{MeshGraph, VertexKey, VertexView};
    /// use plexus::prelude::*;
    /// use plexus::primitive::Quad;
    ///
    /// # fn main() {
    /// fn find<'a, I, T, G>(input: I, geometry: T) -> Option<VertexKey>
    /// where
    ///     I: IntoIterator<Item = VertexView<&'a MeshGraph<G>, G>>,
    ///     T: Copy + IntoGeometry<G::Vertex>,
    ///     G: 'a + Geometry,
    ///     G::Vertex: PartialEq,
    /// {
    ///     input
    ///         .into_iter()
    ///         .find(|vertex| vertex.geometry == geometry.into_geometry())
    ///         .map(|vertex| vertex.key())
    /// }
    ///
    /// let mut graph = MeshGraph::<Point2<f64>>::from_raw_buffers(
    ///     vec![Quad::new(0usize, 1, 2, 3), Quad::new(4, 5, 6, 7)],
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
    /// # }
    /// ```
    pub fn bridge(
        self,
        destination: Selector<ArcKey>,
    ) -> Result<FaceView<&'a mut M, G>, GraphError> {
        let destination = destination.key_or_else(|index| {
            self.interior_path()
                .arcs()
                .nth(index)
                .ok_or_else(|| GraphError::TopologyNotFound)
                .map(|arc| arc.key())
        })?;
        let (source, storage) = self.into_keyed_source();
        let cache = ArcBridgeCache::snapshot(&storage, source, destination)?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| edge::bridge_with_cache(mutation, cache))
            .map(|(storage, face)| (face, storage).into_view().expect_consistent())
    }

    /// Extrudes a boundary arc along its normal.
    ///
    /// Extrusion inserts a new edge (and arcs) with the same geometry as the
    /// initiating arc, but modifies the positional geometry of the new edge's
    /// vertices such that they extend geometrically along the normal of the
    /// originating arc. The originating arc is then bridged with an arc in the
    /// opposing edge. This inserts a quadrilateral face. See `bridge`.
    ///
    /// An arc's normal is perpendicular to the arc and also coplanar with the
    /// arc and one of its neighbors. This is computed via a projection.
    ///
    /// Returns the opposing arc. This is the arc in the destination edge that
    /// is within the same interior path as the initiating arc.
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
    /// use nalgebra::Point2;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    ///
    /// # fn main() {
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
    /// # }
    /// ```
    pub fn extrude<T>(self, distance: T) -> Result<ArcView<&'a mut M, G>, GraphError>
    where
        G: ArcNormal,
        G::Normal: Mul<T>,
        G::Vertex: AsPosition,
        ScaledArcNormal<G, T>: Clone,
        VertexPosition<G>: Add<ScaledArcNormal<G, T>, Output = VertexPosition<G>> + Clone,
    {
        let (ab, storage) = self.into_keyed_source();
        let cache = ArcExtrudeCache::snapshot(&storage, ab, distance)?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| edge::extrude_with_cache(mutation, cache))
            .map(|(storage, arc)| (arc, storage).into_view().expect_consistent())
    }

    /// Removes the arc, its opposite arc, and edge.
    ///
    /// Any and all dependent topology is also removed, such as connected
    /// faces, vertices with no remaining leading arc, etc.
    ///
    /// Returns the source vertex of the initiating arc.
    pub fn remove(self) -> VertexView<&'a mut M, G> {
        (move || {
            let a = self.source_vertex().key();
            let (ab, storage) = self.into_keyed_source();
            let cache = EdgeRemoveCache::snapshot(&storage, ab)?;
            Mutation::replace(storage, Default::default())
                .commit_with(move |mutation| edge::remove_with_cache(mutation, cache))
                .map(|(storage, _)| (a, storage).into_view().expect_consistent())
        })()
        .expect_consistent()
    }
}

impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>>
        + AsStorage<FacePayload<G>>
        + AsStorage<VertexPayload<G>>
        + Consistent,
    G: Geometry + ArcNormal,
{
    pub fn normal(&self) -> Result<G::Normal, GraphError> {
        G::normal(self.interior_reborrow())
    }
}

impl<M, G> Clone for ArcView<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<ArcPayload<G>>,
    G: Geometry,
{
    fn clone(&self) -> Self {
        ArcView {
            key: self.key,
            storage: self.storage.clone(),
            phantom: PhantomData,
        }
    }
}

impl<M, G> Copy for ArcView<M, G>
where
    M: Copy + Reborrow,
    M::Target: AsStorage<ArcPayload<G>>,
    G: Geometry,
{
}

impl<M, G> Deref for ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>>,
    G: Geometry,
{
    type Target = ArcPayload<G>;

    fn deref(&self) -> &Self::Target {
        self.storage.reborrow().as_storage().get(&self.key).unwrap()
    }
}

impl<M, G> DerefMut for ArcView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<ArcPayload<G>> + AsStorageMut<ArcPayload<G>>,
    G: Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.storage
            .reborrow_mut()
            .as_storage_mut()
            .get_mut(&self.key)
            .unwrap()
    }
}

impl<M, G> FromKeyedSource<(ArcKey, M)> for ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (ArcKey, M)) -> Option<Self> {
        let (key, storage) = source;
        storage
            .reborrow()
            .as_storage()
            .contains_key(&key)
            .some(ArcView::from_keyed_source_unchecked((key, storage)))
    }
}

impl<M, G> IntoKeyedSource<(ArcKey, M)> for ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>>,
    G: Geometry,
{
    fn into_keyed_source(self) -> (ArcKey, M) {
        let ArcView { key, storage, .. } = self;
        (key, storage)
    }
}

/// Orphan view of an arc.
///
/// Provides mutable access to an arc's geometry. See the module documentation
/// for more information about topological views.
pub struct OrphanArcView<'a, G>
where
    G: 'a + Geometry,
{
    key: ArcKey,
    arc: &'a mut ArcPayload<G>,
}

impl<'a, G> OrphanArcView<'a, G>
where
    G: 'a + Geometry,
{
    pub fn key(&self) -> ArcKey {
        self.key
    }
}

impl<'a, G> Deref for OrphanArcView<'a, G>
where
    G: 'a + Geometry,
{
    type Target = ArcPayload<G>;

    fn deref(&self) -> &Self::Target {
        &*self.arc
    }
}

impl<'a, G> DerefMut for OrphanArcView<'a, G>
where
    G: 'a + Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.arc
    }
}

impl<'a, M, G> FromKeyedSource<(ArcKey, &'a mut M)> for OrphanArcView<'a, G>
where
    M: AsStorage<ArcPayload<G>> + AsStorageMut<ArcPayload<G>>,
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (ArcKey, &'a mut M)) -> Option<Self> {
        let (key, storage) = source;
        storage
            .as_storage_mut()
            .get_mut(&key)
            .map(|arc| OrphanArcView { key, arc })
    }
}

impl<'a, G> FromKeyedSource<(ArcKey, &'a mut ArcPayload<G>)> for OrphanArcView<'a, G>
where
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (ArcKey, &'a mut ArcPayload<G>)) -> Option<Self> {
        let (key, arc) = source;
        Some(OrphanArcView { key, arc })
    }
}

/// View of an edge.
///
/// Provides traversals, queries, and mutations related to edges in a graph.
/// See the module documentation for more information about topological views.
pub struct EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<EdgePayload<G>>,
    G: Geometry,
{
    key: EdgeKey,
    storage: M,
    phantom: PhantomData<G>,
}

/// Storage.
impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<EdgePayload<G>>,
    G: Geometry,
{
    // TODO: This may become useful as the `mutation` module is developed. It
    //       may also be necessary to expose this API to user code.
    #[allow(dead_code)]
    pub(in crate::graph) fn bind<T, N>(self, storage: N) -> EdgeView<<M as Bind<T, N>>::Output, G>
    where
        T: Payload,
        M: Bind<T, N>,
        M::Output: Reborrow,
        <M::Output as Reborrow>::Target: AsStorage<EdgePayload<G>>,
        N: AsStorage<T>,
    {
        let (key, origin) = self.into_keyed_source();
        EdgeView::from_keyed_source_unchecked((key, origin.bind(storage)))
    }
}

impl<'a, M, G> EdgeView<&'a mut M, G>
where
    M: 'a + AsStorage<EdgePayload<G>> + AsStorageMut<EdgePayload<G>>,
    G: 'a + Geometry,
{
    /// Converts a mutable view into an orphan view.
    pub fn into_orphan(self) -> OrphanEdgeView<'a, G> {
        let (key, storage) = self.into_keyed_source();
        (key, storage).into_view().unwrap()
    }

    /// Converts a mutable view into an immutable view.
    ///
    /// This is useful when mutations are not (or no longer) needed and mutual
    /// access is desired.
    pub fn into_ref(self) -> EdgeView<&'a M, G> {
        let (key, storage) = self.into_keyed_source();
        EdgeView::from_keyed_source_unchecked((key, &*storage))
    }

    /// Reborrows the view and constructs another mutable view from a given
    /// key.
    ///
    /// This allows for fallible traversals from a mutable view without the
    /// need for direct access to the source `MeshGraph`. If the given function
    /// emits a key, then that key will be used to convert this view into
    /// another. If no key is emitted, then the original mutable view is
    /// returned.
    pub fn with_ref<T, K, F>(self, f: F) -> Either<Result<T, GraphError>, Self>
    where
        T: FromKeyedSource<(K, &'a mut M)>,
        F: FnOnce(EdgeView<&M, G>) -> Option<K>,
    {
        if let Some(key) = f(self.interior_reborrow()) {
            let (_, storage) = self.into_keyed_source();
            Either::Left(
                T::from_keyed_source((key, storage)).ok_or_else(|| GraphError::TopologyNotFound),
            )
        }
        else {
            Either::Right(self)
        }
    }
}

impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<EdgePayload<G>>,
    G: Geometry,
{
    /// Gets the key for the edge.
    pub fn key(&self) -> EdgeKey {
        self.key
    }

    fn from_keyed_source_unchecked(source: (EdgeKey, M)) -> Self {
        let (key, storage) = source;
        EdgeView {
            key,
            storage,
            phantom: PhantomData,
        }
    }

    fn interior_reborrow(&self) -> EdgeView<&M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow();
        EdgeView::from_keyed_source_unchecked((key, storage))
    }
}

impl<M, G> EdgeView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<EdgePayload<G>>,
    G: Geometry,
{
    fn interior_reborrow_mut(&mut self) -> EdgeView<&mut M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow_mut();
        EdgeView::from_keyed_source_unchecked((key, storage))
    }
}

/// Reachable API.
impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<EdgePayload<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn into_reachable_arc(self) -> Option<ArcView<M, G>> {
        let key = self.arc;
        let (_, storage) = self.into_keyed_source();
        (key, storage).into_view()
    }

    pub(in crate::graph) fn reachable_arc(&self) -> Option<ArcView<&M::Target, G>> {
        let key = self.arc;
        let storage = self.storage.reborrow();
        (key, storage).into_view()
    }
}

impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<EdgePayload<G>> + Consistent,
    G: Geometry,
{
    pub fn into_arc(self) -> ArcView<M, G> {
        self.into_reachable_arc().expect_consistent()
    }

    pub fn arc(&self) -> ArcView<&M::Target, G> {
        self.reachable_arc().expect_consistent()
    }

    pub fn is_disjoint_edge(&self) -> bool {
        let arc = self.arc();
        arc.is_boundary_arc() && arc.opposite_arc().is_boundary_arc()
    }
}

impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>>
        + AsStorage<EdgePayload<G>>
        + AsStorage<FacePayload<G>>
        + AsStorage<VertexPayload<G>>
        + Consistent,
    G: EdgeMidpoint + Geometry,
{
    pub fn midpoint(&self) -> G::Midpoint {
        G::midpoint(self.interior_reborrow()).expect_consistent()
    }
}

impl<M, G> Clone for EdgeView<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<EdgePayload<G>>,
    G: Geometry,
{
    fn clone(&self) -> Self {
        EdgeView {
            key: self.key,
            storage: self.storage.clone(),
            phantom: PhantomData,
        }
    }
}

impl<M, G> Copy for EdgeView<M, G>
where
    M: Copy + Reborrow,
    M::Target: AsStorage<EdgePayload<G>>,
    G: Geometry,
{
}

impl<M, G> Deref for EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<EdgePayload<G>>,
    G: Geometry,
{
    type Target = EdgePayload<G>;

    fn deref(&self) -> &Self::Target {
        self.storage.reborrow().as_storage().get(&self.key).unwrap()
    }
}

impl<M, G> DerefMut for EdgeView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<EdgePayload<G>> + AsStorageMut<EdgePayload<G>>,
    G: Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.storage
            .reborrow_mut()
            .as_storage_mut()
            .get_mut(&self.key)
            .unwrap()
    }
}

impl<M, G> FromKeyedSource<(EdgeKey, M)> for EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<EdgePayload<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (EdgeKey, M)) -> Option<Self> {
        let (key, storage) = source;
        storage
            .reborrow()
            .as_storage()
            .contains_key(&key)
            .some(EdgeView::from_keyed_source_unchecked((key, storage)))
    }
}

impl<M, G> IntoKeyedSource<(EdgeKey, M)> for EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<EdgePayload<G>>,
    G: Geometry,
{
    fn into_keyed_source(self) -> (EdgeKey, M) {
        let EdgeView { key, storage, .. } = self;
        (key, storage)
    }
}

/// Orphan view of an edge.
///
/// Provides mutable access to an edge's geometry. See the module documentation
/// for more information about topological views.
pub struct OrphanEdgeView<'a, G>
where
    G: 'a + Geometry,
{
    key: EdgeKey,
    edge: &'a mut EdgePayload<G>,
}

impl<'a, G> OrphanEdgeView<'a, G>
where
    G: 'a + Geometry,
{
    pub fn key(&self) -> EdgeKey {
        self.key
    }
}

impl<'a, G> Deref for OrphanEdgeView<'a, G>
where
    G: 'a + Geometry,
{
    type Target = EdgePayload<G>;

    fn deref(&self) -> &Self::Target {
        &*self.edge
    }
}

impl<'a, G> DerefMut for OrphanEdgeView<'a, G>
where
    G: 'a + Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.edge
    }
}

impl<'a, M, G> FromKeyedSource<(EdgeKey, &'a mut M)> for OrphanEdgeView<'a, G>
where
    M: AsStorage<EdgePayload<G>> + AsStorageMut<EdgePayload<G>>,
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (EdgeKey, &'a mut M)) -> Option<Self> {
        let (key, storage) = source;
        storage
            .as_storage_mut()
            .get_mut(&key)
            .map(|edge| OrphanEdgeView { key, edge })
    }
}

impl<'a, G> FromKeyedSource<(EdgeKey, &'a mut EdgePayload<G>)> for OrphanEdgeView<'a, G>
where
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (EdgeKey, &'a mut EdgePayload<G>)) -> Option<Self> {
        let (key, edge) = source;
        Some(OrphanEdgeView { key, edge })
    }
}

/// Keys for topology forming an arc.
///
/// An arc neighborhood describes the keys for the arc and its vertices.
#[derive(Clone, Debug)]
pub struct ArcNeighborhood {
    key: ArcKey,
    vertices: (VertexKey, VertexKey),
}

impl ArcNeighborhood {
    pub fn key(&self) -> ArcKey {
        self.key
    }

    pub fn vertices(&self) -> (VertexKey, VertexKey) {
        self.vertices
    }
}

impl<M, G> From<ArcView<M, G>> for ArcNeighborhood
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<VertexPayload<G>> + Consistent,
    G: Geometry,
{
    fn from(arc: ArcView<M, G>) -> Self {
        let a = arc.source_vertex().key();
        let b = arc.destination_vertex().key();
        ArcNeighborhood {
            key: arc.key,
            vertices: (a, b),
        }
    }
}

struct VertexCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<VertexPayload<G>>,
    G: Geometry,
{
    storage: M,
    input: <ArrayVec<[VertexKey; 2]> as IntoIterator>::IntoIter,
    phantom: PhantomData<G>,
}

impl<M, G> VertexCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<VertexPayload<G>>,
    G: Geometry,
{
    fn next(&mut self) -> Option<VertexKey> {
        self.input.next()
    }
}

impl<M, G> From<ArcView<M, G>> for VertexCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<VertexPayload<G>>,
    G: Geometry,
{
    fn from(arc: ArcView<M, G>) -> Self {
        let (a, b) = arc.key().into();
        let (_, storage) = arc.into_keyed_source();
        VertexCirculator {
            storage,
            input: ArrayVec::<_>::from([a, b]).into_iter(),
            phantom: PhantomData,
        }
    }
}

impl<M, G> Clone for VertexCirculator<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<VertexPayload<G>>,
    G: Geometry,
{
    fn clone(&self) -> Self {
        VertexCirculator {
            storage: self.storage.clone(),
            input: self.input.clone(),
            phantom: PhantomData,
        }
    }
}

impl<'a, M, G> Iterator for VertexCirculator<&'a M, G>
where
    M: 'a + AsStorage<VertexPayload<G>>,
    G: 'a + Geometry,
{
    type Item = VertexView<&'a M, G>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).and_then(|key| (key, self.storage).into_view())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(2))
    }
}

impl<'a, M, G> Iterator for VertexCirculator<&'a mut M, G>
where
    M: 'a + AsStorage<VertexPayload<G>> + AsStorageMut<VertexPayload<G>>,
    G: 'a + Geometry,
{
    type Item = OrphanVertexView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).and_then(|key| {
            (key, unsafe {
                mem::transmute::<&'_ mut Storage<VertexPayload<G>>, &'a mut Storage<VertexPayload<G>>>(
                    self.storage.as_storage_mut(),
                )
            })
                .into_view()
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(2))
    }
}

struct FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<FacePayload<G>>,
    G: Geometry,
{
    storage: M,
    input: <ArrayVec<[FaceKey; 2]> as IntoIterator>::IntoIter,
    phantom: PhantomData<G>,
}

impl<M, G> FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<FacePayload<G>>,
    G: Geometry,
{
    fn next(&mut self) -> Option<FaceKey> {
        self.input.next()
    }
}

impl<M, G> Clone for FaceCirculator<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<FacePayload<G>>,
    G: Geometry,
{
    fn clone(&self) -> Self {
        FaceCirculator {
            storage: self.storage.clone(),
            input: self.input.clone(),
            phantom: PhantomData,
        }
    }
}

impl<M, G> From<ArcView<M, G>> for FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<FacePayload<G>>,
    G: Geometry,
{
    fn from(arc: ArcView<M, G>) -> Self {
        let input = arc
            .face
            .into_iter()
            .chain(
                arc.reachable_opposite_arc()
                    .and_then(|opposite| opposite.face)
                    .into_iter(),
            )
            .collect::<ArrayVec<_>>()
            .into_iter();
        let (_, storage) = arc.into_keyed_source();
        FaceCirculator {
            storage,
            input,
            phantom: PhantomData,
        }
    }
}

impl<'a, M, G> Iterator for FaceCirculator<&'a M, G>
where
    M: 'a + AsStorage<FacePayload<G>>,
    G: 'a + Geometry,
{
    type Item = FaceView<&'a M, G>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).and_then(|key| (key, self.storage).into_view())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(2))
    }
}

impl<'a, M, G> Iterator for FaceCirculator<&'a mut M, G>
where
    M: 'a + AsStorage<FacePayload<G>> + AsStorageMut<FacePayload<G>>,
    G: 'a + Geometry,
{
    type Item = OrphanFaceView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).and_then(|key| {
            (key, unsafe {
                mem::transmute::<&'_ mut Storage<FacePayload<G>>, &'a mut Storage<FacePayload<G>>>(
                    self.storage.as_storage_mut(),
                )
            })
                .into_view()
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(2))
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point2, Point3};

    use crate::geometry::convert::IntoGeometry;
    use crate::geometry::*;
    use crate::graph::*;
    use crate::primitive::cube::Cube;
    use crate::primitive::generate::*;
    use crate::primitive::index::*;
    use crate::*;

    fn find_arc_with_vertex_geometry<G, T>(graph: &MeshGraph<G>, geometry: (T, T)) -> Option<ArcKey>
    where
        G: Geometry,
        G::Vertex: PartialEq,
        T: IntoGeometry<G::Vertex>,
    {
        fn find_vertex_with_geometry<G, T>(
            graph: &MeshGraph<G>,
            geometry: T,
        ) -> Option<VertexView<&MeshGraph<G>, G>>
        where
            G: Geometry,
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
        // Construct a mesh with two independent quads.
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
            .polygons_with_position() // 6 quads, 24 vertices.
            .index_vertices(HashIndexer::default());
        let mut graph = MeshGraph::<Point3<f32>>::from_raw_buffers(indices, vertices).unwrap();
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
        // Construct a graph with two connected quads.
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

        // Remove the edge joining the quads from the graph.
        let ab = find_arc_with_vertex_geometry(&graph, ((0.0, 0.0), (0.0, 1.0))).unwrap();
        {
            let vertex = graph.arc_mut(ab).unwrap().remove().into_ref();

            // The path should be formed from 6 edges.
            assert_eq!(6, vertex.into_outgoing_arc().into_interior_path().arity());
        }

        // After the removal, the graph should have no faces.
        assert_eq!(0, graph.face_count());
    }
}
