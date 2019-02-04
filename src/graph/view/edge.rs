use arrayvec::ArrayVec;
use either::Either;
use fool::prelude::*;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Add, Deref, DerefMut, Mul};

use crate::geometry::alias::{ScaledEdgeLateral, VertexPosition};
use crate::geometry::convert::AsPosition;
use crate::geometry::Geometry;
use crate::graph::container::alias::OwnedCore;
use crate::graph::container::{Bind, Consistent, Reborrow, ReborrowMut};
use crate::graph::geometry::{EdgeLateral, EdgeMidpoint};
use crate::graph::mutation::edge::{
    self, ArcBridgeCache, ArcExtrudeCache, EdgeRemoveCache, EdgeSplitCache,
};
use crate::graph::mutation::{Mutate, Mutation};
use crate::graph::storage::convert::{AsStorage, AsStorageMut};
use crate::graph::storage::{ArcKey, FaceKey, Storage, VertexKey};
use crate::graph::topology::{Arc, Face, Topological, Vertex};
use crate::graph::view::convert::{FromKeyedSource, IntoKeyedSource, IntoView};
use crate::graph::view::{
    FaceView, InteriorPathView, OrphanFaceView, OrphanVertexView, Selector, VertexView,
};
use crate::graph::{GraphError, OptionExt};

/// Reference to a arc.
///
/// Provides traversals, queries, and mutations related to arcs in a
/// mesh. See the module documentation for more information about topological
/// views.
pub struct ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>>,
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
    M::Target: AsStorage<Arc<G>>,
    G: Geometry,
{
    // TODO: This may become useful as the `mutation` module is developed. It
    //       may also be necessary to expose this API to user code.
    #[allow(dead_code)]
    pub(in crate::graph) fn bind<T, N>(self, storage: N) -> ArcView<<M as Bind<T, N>>::Output, G>
    where
        T: Topological,
        M: Bind<T, N>,
        M::Output: Reborrow,
        <M::Output as Reborrow>::Target: AsStorage<Arc<G>>,
        N: AsStorage<T>,
    {
        let (key, origin) = self.into_keyed_source();
        ArcView::from_keyed_source_unchecked((key, origin.bind(storage)))
    }
}

impl<'a, M, G> ArcView<&'a mut M, G>
where
    M: 'a + AsStorage<Arc<G>> + AsStorageMut<Arc<G>>,
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
    M::Target: AsStorage<Arc<G>>,
    G: Geometry,
{
    /// Gets the key for this arc.
    pub fn key(&self) -> ArcKey {
        self.key
    }

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
    M::Target: AsStorage<Arc<G>>,
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
    M::Target: AsStorage<Arc<G>>,
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
    M::Target: AsStorage<Arc<G>> + Consistent,
    G: Geometry,
{
    pub fn into_interior_path(self) -> InteriorPathView<M, G> {
        let (key, storage) = self.into_keyed_source();
        (key, storage).into_view().expect_consistent()
    }

    pub fn into_boundary_arc(self) -> Option<Self> {
        self.into_reachable_boundary_arc()
    }

    pub fn into_opposite_arc(self) -> Self {
        self.into_reachable_opposite_arc().expect_consistent()
    }

    pub fn into_next_arc(self) -> Self {
        self.into_reachable_next_arc().expect_consistent()
    }

    pub fn into_previous_arc(self) -> Self {
        self.into_reachable_previous_arc().expect_consistent()
    }

    pub fn interior_path(&self) -> InteriorPathView<&M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow();
        (key, storage).into_view().expect_consistent()
    }

    pub fn boundary_arc(&self) -> Option<ArcView<&M::Target, G>> {
        self.reachable_boundary_arc()
    }

    pub fn opposite_arc(&self) -> ArcView<&M::Target, G> {
        self.reachable_opposite_arc().expect_consistent()
    }

    pub fn next_arc(&self) -> ArcView<&M::Target, G> {
        self.reachable_next_arc().expect_consistent()
    }

    pub fn previous_arc(&self) -> ArcView<&M::Target, G> {
        self.reachable_previous_arc().expect_consistent()
    }

    // TODO: Move this into a composite-edge type.
    pub fn is_disjoint_arc(&self) -> bool {
        self.is_boundary_arc() && self.opposite_arc().is_boundary_arc()
    }
}

/// Reachable API.
impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>>,
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
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent,
    G: Geometry,
{
    pub fn neighborhood(&self) -> ArcNeighborhood {
        ArcNeighborhood::from(self.interior_reborrow())
    }

    pub fn into_source_vertex(self) -> VertexView<M, G> {
        self.into_reachable_source_vertex().expect_consistent()
    }

    pub fn into_destination_vertex(self) -> VertexView<M, G> {
        self.into_reachable_destination_vertex().expect_consistent()
    }

    pub fn source_vertex(&self) -> VertexView<&M::Target, G> {
        self.reachable_source_vertex().expect_consistent()
    }

    pub fn destination_vertex(&self) -> VertexView<&M::Target, G> {
        self.reachable_destination_vertex().expect_consistent()
    }
}

/// Reachable API.
impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>>,
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
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + Consistent,
    G: Geometry,
{
    pub fn into_face(self) -> Option<FaceView<M, G>> {
        self.into_reachable_face()
    }

    pub fn face(&self) -> Option<FaceView<&M::Target, G>> {
        self.reachable_face()
    }
}

/// Reachable API.
impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>>,
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
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent,
    G: Geometry,
{
    pub fn vertices(&self) -> impl Clone + Iterator<Item = VertexView<&M::Target, G>> {
        self.reachable_vertices()
    }
}

/// Reachable API.
impl<M, G> ArcView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
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
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>> + Consistent,
    G: Geometry,
{
    pub fn orphan_vertices(&mut self) -> impl Iterator<Item = OrphanVertexView<G>> {
        self.reachable_orphan_vertices()
    }
}

/// Reachable API.
impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>>,
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
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + Consistent,
    G: Geometry,
{
    pub fn faces(&self) -> impl Clone + Iterator<Item = FaceView<&M::Target, G>> {
        self.reachable_faces()
    }
}

/// Reachable API.
impl<M, G> ArcView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
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
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>> + Consistent,
    G: Geometry,
{
    pub fn orphan_faces(&mut self) -> impl Iterator<Item = OrphanFaceView<G>> {
        self.reachable_orphan_faces()
    }
}

impl<'a, M, G> ArcView<&'a mut M, G>
where
    M: AsStorage<Arc<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Default
        + From<OwnedCore<G>>
        + Into<OwnedCore<G>>,
    G: 'a + Geometry,
{
    pub fn remove(self) -> Result<VertexView<&'a mut M, G>, GraphError> {
        let a = self.source_vertex().key();
        let (ab, storage) = self.into_keyed_source();
        let cache = EdgeRemoveCache::snapshot(&storage, ab)?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| edge::remove_with_cache(mutation, cache))
            .map(|(storage, _)| (a, storage).into_view().expect_consistent())
    }

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
}

impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    G: EdgeMidpoint + Geometry,
{
    pub fn midpoint(&self) -> Result<G::Midpoint, GraphError> {
        G::midpoint(self.interior_reborrow())
    }
}

impl<'a, M, G> ArcView<&'a mut M, G>
where
    M: AsStorage<Arc<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Default
        + From<OwnedCore<G>>
        + Into<OwnedCore<G>>,
    G: 'a + EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
    G::Vertex: AsPosition,
{
    // TODO: Move this into a composite-edge type.
    pub fn split(self) -> Result<VertexView<&'a mut M, G>, GraphError> {
        let (ab, storage) = self.into_keyed_source();
        let cache = EdgeSplitCache::snapshot(&storage, ab)?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| edge::split_with_cache(mutation, cache))
            .map(|(storage, vertex)| (vertex, storage).into_view().expect_consistent())
    }
}

impl<M, G> ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    G: Geometry + EdgeLateral,
{
    pub fn lateral(&self) -> Result<G::Lateral, GraphError> {
        G::lateral(self.interior_reborrow())
    }
}

impl<'a, M, G> ArcView<&'a mut M, G>
where
    M: AsStorage<Arc<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Default
        + From<OwnedCore<G>>
        + Into<OwnedCore<G>>,
    G: 'a + Geometry + EdgeLateral,
    G::Vertex: AsPosition,
{
    pub fn extrude<T>(self, distance: T) -> Result<ArcView<&'a mut M, G>, GraphError>
    where
        G::Lateral: Mul<T>,
        ScaledEdgeLateral<G, T>: Clone,
        VertexPosition<G>: Add<ScaledEdgeLateral<G, T>, Output = VertexPosition<G>> + Clone,
    {
        let (ab, storage) = self.into_keyed_source();
        let cache = ArcExtrudeCache::snapshot(&storage, ab, distance)?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| edge::extrude_with_cache(mutation, cache))
            .map(|(storage, arc)| (arc, storage).into_view().expect_consistent())
    }
}

impl<M, G> Clone for ArcView<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<Arc<G>>,
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
    M::Target: AsStorage<Arc<G>>,
    G: Geometry,
{
}

impl<M, G> Deref for ArcView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>>,
    G: Geometry,
{
    type Target = Arc<G>;

    fn deref(&self) -> &Self::Target {
        self.storage.reborrow().as_storage().get(&self.key).unwrap()
    }
}

impl<M, G> DerefMut for ArcView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Arc<G>> + AsStorageMut<Arc<G>>,
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
    M::Target: AsStorage<Arc<G>>,
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
    M::Target: AsStorage<Arc<G>>,
    G: Geometry,
{
    fn into_keyed_source(self) -> (ArcKey, M) {
        let ArcView { key, storage, .. } = self;
        (key, storage)
    }
}

/// Orphan reference to a arc.
///
/// Consider using `OrphanEdge` instead. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
pub struct OrphanArcView<'a, G>
where
    G: 'a + Geometry,
{
    key: ArcKey,
    arc: &'a mut Arc<G>,
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
    type Target = Arc<G>;

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
    M: AsStorage<Arc<G>> + AsStorageMut<Arc<G>>,
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

impl<'a, G> FromKeyedSource<(ArcKey, &'a mut Arc<G>)> for OrphanArcView<'a, G>
where
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (ArcKey, &'a mut Arc<G>)) -> Option<Self> {
        let (key, arc) = source;
        Some(OrphanArcView { key, arc })
    }
}

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
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent,
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
    M::Target: AsStorage<Vertex<G>>,
    G: Geometry,
{
    storage: M,
    input: <ArrayVec<[VertexKey; 2]> as IntoIterator>::IntoIter,
    phantom: PhantomData<G>,
}

impl<M, G> VertexCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: Geometry,
{
    fn next(&mut self) -> Option<VertexKey> {
        self.input.next()
    }
}

impl<M, G> From<ArcView<M, G>> for VertexCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>>,
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
    M::Target: AsStorage<Vertex<G>>,
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
    M: 'a + AsStorage<Vertex<G>>,
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
    M: 'a + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: 'a + Geometry,
{
    type Item = OrphanVertexView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).and_then(|key| {
            (key, unsafe {
                mem::transmute::<&'_ mut Storage<Vertex<G>>, &'a mut Storage<Vertex<G>>>(
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
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    storage: M,
    input: <ArrayVec<[FaceKey; 2]> as IntoIterator>::IntoIter,
    phantom: PhantomData<G>,
}

impl<M, G> FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    fn next(&mut self) -> Option<FaceKey> {
        self.input.next()
    }
}

impl<M, G> Clone for FaceCirculator<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<Face<G>>,
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
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>>,
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
    M: 'a + AsStorage<Face<G>>,
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
    M: 'a + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: 'a + Geometry,
{
    type Item = OrphanFaceView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).and_then(|key| {
            (key, unsafe {
                mem::transmute::<&'_ mut Storage<Face<G>>, &'a mut Storage<Face<G>>>(
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

    fn find_arc_with_geometry<G, T>(graph: &MeshGraph<G>, geometry: (T, T)) -> Option<ArcKey>
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
        let ab = find_arc_with_geometry(&graph, ((0.0, 0.0), (0.0, 1.0))).unwrap();
        {
            let edge = graph.arc_mut(ab).unwrap();
            let vertex = edge.remove().unwrap().into_ref();

            // The path should be formed from 6 edges.
            assert_eq!(6, vertex.into_outgoing_arc().into_interior_path().arity());
        }

        // After the removal, the graph should have no faces.
        assert_eq!(0, graph.face_count());
    }

    #[test]
    fn extrude_edge() {
        let mut graph = MeshGraph::<Point2<f32>>::from_raw_buffers_with_arity(
            vec![0u32, 1, 2, 3],
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            4,
        )
        .unwrap();
        let source = find_arc_with_geometry(&graph, ((1.0, 1.0), (1.0, 0.0))).unwrap();
        graph.arc_mut(source).unwrap().extrude(1.0).unwrap();

        assert_eq!(14, graph.arc_count());
        assert_eq!(2, graph.face_count());
    }

    #[test]
    fn bridge_edges() {
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
        let source = find_arc_with_geometry(&graph, ((-1.0, 1.0, 0.0), (-1.0, 0.0, 0.0))).unwrap();
        let destination =
            find_arc_with_geometry(&graph, ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0))).unwrap();
        graph
            .arc_mut(source)
            .unwrap()
            .bridge(ByKey(destination))
            .unwrap();

        assert_eq!(20, graph.arc_count());
        assert_eq!(3, graph.face_count());
    }

    #[test]
    fn split_composite_edge() {
        let (indices, vertices) = Cube::new()
            .polygons_with_position() // 6 quads, 24 vertices.
            .index_vertices(HashIndexer::default());
        let mut graph = MeshGraph::<Point3<f32>>::from_raw_buffers(indices, vertices).unwrap();
        let key = graph.arcs().nth(0).unwrap().key();
        let vertex = graph.arc_mut(key).unwrap().split().unwrap().into_ref();

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
}
