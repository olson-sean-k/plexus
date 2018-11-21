use arrayvec::ArrayVec;
use fool::prelude::*;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Add, Deref, DerefMut, Mul};

use geometry::convert::AsPosition;
use geometry::Geometry;
use graph::container::alias::OwnedCore;
use graph::container::{Bind, Consistent, Reborrow, ReborrowMut};
use graph::geometry::alias::{ScaledEdgeLateral, VertexPosition};
use graph::geometry::{EdgeLateral, EdgeMidpoint};
use graph::mutation::edge::{self, EdgeExtrudeCache, EdgeJoinCache, EdgeSplitCache};
use graph::mutation::{Mutate, Mutation};
use graph::storage::convert::{AsStorage, AsStorageMut};
use graph::storage::{EdgeKey, FaceKey, Storage, VertexKey};
use graph::topology::{Edge, Face, Topological, Vertex};
use graph::view::convert::{FromKeyedSource, IntoView};
use graph::view::{FaceView, OrphanFaceView, OrphanVertexView, VertexView};
use graph::GraphError;

/// Reference to an edge.
///
/// Provides traversals, queries, and mutations related to edges in a mesh. See
/// the module documentation for more information about topological views.
pub struct EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>,
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
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
    // TODO: This may become useful as the `mutation` module is developed. It
    //       may also be necessary to expose this API to user code.
    #[allow(dead_code)]
    pub(in graph) fn bind<T, N>(self, storage: N) -> EdgeView<<M as Bind<T, N>>::Output, G>
    where
        T: Topological,
        M: Bind<T, N>,
        M::Output: Reborrow,
        <M::Output as Reborrow>::Target: AsStorage<Edge<G>>,
        N: AsStorage<T>,
    {
        let (key, origin) = self.into_keyed_storage();
        EdgeView::from_keyed_storage_unchecked(key, origin.bind(storage))
    }
}

impl<'a, M, G> EdgeView<&'a mut M, G>
where
    M: 'a + AsStorage<Edge<G>> + AsStorageMut<Edge<G>>,
    G: 'a + Geometry,
{
    /// Converts a mutable view into an orphan view.
    pub fn into_orphan(self) -> OrphanEdgeView<'a, G> {
        let (key, storage) = self.into_keyed_storage();
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
    /// let mut graph = MeshGraph::<Point2<f32>>::from_raw_buffers(
    ///     vec![0, 1, 2, 3],
    ///     vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    ///     4,
    /// )
    /// .unwrap();
    /// let key = graph
    ///     .edges()
    ///     .find(|edge| edge.is_boundary_edge())
    ///     .unwrap()
    ///     .key();
    /// let edge = graph
    ///     .edge_mut(key)
    ///     .unwrap()
    ///     .extrude(1.0)
    ///     .unwrap()
    ///     .into_ref();
    ///
    /// // This would not be possible without conversion into an immutable view.
    /// let _ = edge.into_next_edge().into_next_edge().into_face();
    /// let _ = edge.into_opposite_edge().into_face();
    /// # }
    /// ```
    pub fn into_ref(self) -> EdgeView<&'a M, G> {
        let (key, storage) = self.into_keyed_storage();
        EdgeView::from_keyed_storage_unchecked(key, &*storage)
    }
}

impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
    /// Gets the key for this edge.
    pub fn key(&self) -> EdgeKey {
        self.key
    }

    pub fn to_key_topology(&self) -> EdgeKeyTopology {
        EdgeKeyTopology::from(self.interior_reborrow())
    }

    pub fn is_boundary_edge(&self) -> bool {
        self.face.is_none()
    }

    fn from_keyed_storage(key: EdgeKey, storage: M) -> Option<Self> {
        storage
            .reborrow()
            .as_storage()
            .contains_key(&key)
            .some(EdgeView::from_keyed_storage_unchecked(key, storage))
    }

    fn from_keyed_storage_unchecked(key: EdgeKey, storage: M) -> Self {
        EdgeView {
            key,
            storage,
            phantom: PhantomData,
        }
    }

    fn into_keyed_storage(self) -> (EdgeKey, M) {
        let EdgeView { key, storage, .. } = self;
        (key, storage)
    }

    fn interior_reborrow(&self) -> EdgeView<&M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow();
        EdgeView::from_keyed_storage_unchecked(key, storage)
    }
}

impl<M, G> EdgeView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
    // TODO: This is basic infrastructure and it is reasonably likely this will
    //       be used in the future.
    #[allow(dead_code)]
    fn interior_reborrow_mut(&mut self) -> EdgeView<&mut M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow_mut();
        EdgeView::from_keyed_storage_unchecked(key, storage)
    }
}

/// Reachable API.
impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
    pub(in graph) fn into_reachable_boundary_edge(self) -> Option<Self> {
        if self.is_boundary_edge() {
            Some(self)
        }
        else {
            self.into_reachable_opposite_edge()
                .and_then(|opposite| opposite.is_boundary_edge().some(opposite))
        }
    }

    pub(in graph) fn into_reachable_opposite_edge(self) -> Option<Self> {
        let key = self.opposite;
        key.and_then(move |key| {
            let (_, storage) = self.into_keyed_storage();
            (key, storage).into_view()
        })
    }

    pub(in graph) fn into_reachable_next_edge(self) -> Option<Self> {
        let key = self.next;
        key.and_then(move |key| {
            let (_, storage) = self.into_keyed_storage();
            (key, storage).into_view()
        })
    }

    pub(in graph) fn into_reachable_previous_edge(self) -> Option<Self> {
        let key = self.previous;
        key.and_then(move |key| {
            let (_, storage) = self.into_keyed_storage();
            (key, storage).into_view()
        })
    }

    pub(in graph) fn reachable_boundary_edge(&self) -> Option<EdgeView<&M::Target, G>> {
        if self.is_boundary_edge() {
            Some(self.interior_reborrow())
        }
        else {
            self.reachable_opposite_edge()
                .and_then(|opposite| opposite.is_boundary_edge().some_with(|| opposite))
        }
    }

    pub(in graph) fn reachable_opposite_edge(&self) -> Option<EdgeView<&M::Target, G>> {
        self.opposite.and_then(|key| {
            let storage = self.storage.reborrow();
            (key, storage).into_view()
        })
    }

    pub(in graph) fn reachable_next_edge(&self) -> Option<EdgeView<&M::Target, G>> {
        self.next.and_then(|key| {
            let storage = self.storage.reborrow();
            (key, storage).into_view()
        })
    }

    pub(in graph) fn reachable_previous_edge(&self) -> Option<EdgeView<&M::Target, G>> {
        self.previous.and_then(|key| {
            let storage = self.storage.reborrow();
            (key, storage).into_view()
        })
    }
}

impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + Consistent,
    G: Geometry,
{
    pub fn into_boundary_edge(self) -> Option<Self> {
        self.into_reachable_boundary_edge()
    }

    pub fn into_opposite_edge(self) -> Self {
        self.into_reachable_opposite_edge().unwrap()
    }

    pub fn into_next_edge(self) -> Self {
        self.into_reachable_next_edge().unwrap()
    }

    pub fn into_previous_edge(self) -> Self {
        self.into_reachable_previous_edge().unwrap()
    }

    pub fn boundary_edge(&self) -> Option<EdgeView<&M::Target, G>> {
        self.reachable_boundary_edge()
    }

    pub fn opposite_edge(&self) -> EdgeView<&M::Target, G> {
        self.reachable_opposite_edge().unwrap()
    }

    pub fn next_edge(&self) -> EdgeView<&M::Target, G> {
        self.reachable_next_edge().unwrap()
    }

    pub fn previous_edge(&self) -> EdgeView<&M::Target, G> {
        self.reachable_previous_edge().unwrap()
    }
}

/// Reachable API.
impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub(in graph) fn into_reachable_source_vertex(self) -> Option<VertexView<M, G>> {
        let (key, _) = self.key.to_vertex_keys();
        let (_, storage) = self.into_keyed_storage();
        (key, storage).into_view()
    }

    pub(in graph) fn into_reachable_destination_vertex(self) -> Option<VertexView<M, G>> {
        let key = self.vertex;
        let (_, storage) = self.into_keyed_storage();
        (key, storage).into_view()
    }

    pub(in graph) fn reachable_source_vertex(&self) -> Option<VertexView<&M::Target, G>> {
        let (key, _) = self.key.to_vertex_keys();
        let storage = self.storage.reborrow();
        (key, storage).into_view()
    }

    pub(in graph) fn reachable_destination_vertex(&self) -> Option<VertexView<&M::Target, G>> {
        let key = self.vertex;
        let storage = self.storage.reborrow();
        (key, storage).into_view()
    }
}

impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Vertex<G>> + Consistent,
    G: Geometry,
{
    pub fn into_source_vertex(self) -> VertexView<M, G> {
        self.into_reachable_source_vertex().unwrap()
    }

    pub fn into_destination_vertex(self) -> VertexView<M, G> {
        self.into_reachable_destination_vertex().unwrap()
    }

    pub fn source_vertex(&self) -> VertexView<&M::Target, G> {
        self.reachable_source_vertex().unwrap()
    }

    pub fn destination_vertex(&self) -> VertexView<&M::Target, G> {
        self.reachable_destination_vertex().unwrap()
    }
}

/// Reachable API.
impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    pub(in graph) fn into_reachable_face(self) -> Option<FaceView<M, G>> {
        let key = self.face;
        key.and_then(move |key| {
            let (_, storage) = self.into_keyed_storage();
            (key, storage).into_view()
        })
    }

    pub(in graph) fn reachable_face(&self) -> Option<FaceView<&M::Target, G>> {
        self.face.and_then(|key| {
            let storage = self.storage.reborrow();
            (key, storage).into_view()
        })
    }
}

impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + Consistent,
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
impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub(in graph) fn reachable_vertices(&self) -> VertexCirculator<&M::Target, G> {
        let (a, b) = self.key.to_vertex_keys();
        let storage = self.storage.reborrow();
        (ArrayVec::from([b, a]), storage).into_view().unwrap()
    }
}

impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Vertex<G>> + Consistent,
    G: Geometry,
{
    pub fn vertices(&self) -> VertexCirculator<&M::Target, G> {
        self.reachable_vertices()
    }
}

/// Reachable API.
impl<M, G> EdgeView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>> + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: Geometry,
{
    pub(in graph) fn reachable_orphan_vertices(&mut self) -> VertexCirculator<&mut M::Target, G> {
        let (a, b) = self.key.to_vertex_keys();
        let storage = self.storage.reborrow_mut();
        (ArrayVec::from([b, a]), storage).into_view().unwrap()
    }
}

impl<M, G> EdgeView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>> + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>> + Consistent,
    G: Geometry,
{
    pub fn orphan_vertices(&mut self) -> VertexCirculator<&mut M::Target, G> {
        self.reachable_orphan_vertices()
    }
}

/// Reachable API.
impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    pub(in graph) fn reachable_faces(&self) -> FaceCirculator<&M::Target, G> {
        let keys = self
            .face
            .into_iter()
            .chain(
                self.reachable_opposite_edge()
                    .and_then(|opposite| opposite.face),
            )
            .collect();
        let storage = self.storage.reborrow();
        (keys, storage).into_view().unwrap()
    }
}

impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + Consistent,
    G: Geometry,
{
    pub fn faces(&self) -> FaceCirculator<&M::Target, G> {
        self.reachable_faces()
    }
}

impl<'a, M, G> EdgeView<&'a mut M, G>
where
    M: AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Default
        + From<OwnedCore<G>>
        + Into<OwnedCore<G>>,
    G: 'a + Geometry,
{
    // TODO: Rename this to something like "extend". It is very similar to
    //       `extrude`. Terms like "join" or "merge" are better suited for
    //       directly joining two adjacent faces over a shared edge.
    pub fn join(self, destination: EdgeKey) -> Result<EdgeView<&'a mut M, G>, GraphError> {
        let (source, storage) = self.into_keyed_storage();
        let cache = EdgeJoinCache::snapshot(&storage, source, destination)?;
        let (storage, edge) = Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| edge::join_with_cache(mutation, cache))
            .unwrap();
        Ok((edge, storage).into_view().unwrap())
    }
}

impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    G: EdgeMidpoint + Geometry,
{
    pub fn midpoint(&self) -> Result<G::Midpoint, GraphError> {
        G::midpoint(self.interior_reborrow())
    }
}

impl<'a, M, G> EdgeView<&'a mut M, G>
where
    M: AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Default
        + From<OwnedCore<G>>
        + Into<OwnedCore<G>>,
    G: 'a + EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
    G::Vertex: AsPosition,
{
    pub fn split(self) -> Result<VertexView<&'a mut M, G>, GraphError> {
        let (ab, storage) = self.into_keyed_storage();
        let cache = EdgeSplitCache::snapshot(&storage, ab)?;
        let (storage, vertex) = Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| edge::split_with_cache(mutation, cache))
            .unwrap();
        Ok((vertex, storage).into_view().unwrap())
    }
}

impl<M, G> EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    G: Geometry + EdgeLateral,
{
    pub fn lateral(&self) -> Result<G::Lateral, GraphError> {
        G::lateral(self.interior_reborrow())
    }
}

impl<'a, M, G> EdgeView<&'a mut M, G>
where
    M: AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Default
        + From<OwnedCore<G>>
        + Into<OwnedCore<G>>,
    G: 'a + Geometry + EdgeLateral,
    G::Vertex: AsPosition,
{
    pub fn extrude<T>(self, distance: T) -> Result<EdgeView<&'a mut M, G>, GraphError>
    where
        G::Lateral: Mul<T>,
        ScaledEdgeLateral<G, T>: Clone,
        VertexPosition<G>: Add<ScaledEdgeLateral<G, T>, Output = VertexPosition<G>> + Clone,
    {
        let (ab, storage) = self.into_keyed_storage();
        let cache = EdgeExtrudeCache::snapshot(&storage, ab, distance)?;
        let (storage, edge) = Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| edge::extrude_with_cache(mutation, cache))
            .unwrap();
        Ok((edge, storage).into_view().unwrap())
    }
}

impl<M, G> Clone for EdgeView<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<Edge<G>>,
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
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
}

impl<M, G> Deref for EdgeView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
    type Target = Edge<G>;

    fn deref(&self) -> &Self::Target {
        self.storage.reborrow().as_storage().get(&self.key).unwrap()
    }
}

impl<M, G> DerefMut for EdgeView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>> + AsStorageMut<Edge<G>>,
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
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (EdgeKey, M)) -> Option<Self> {
        let (key, storage) = source;
        EdgeView::from_keyed_storage(key, storage)
    }
}

/// Orphan reference to an edge.
///
/// Consider using `OrphanEdge` instead. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
pub struct OrphanEdgeView<'a, G>
where
    G: 'a + Geometry,
{
    key: EdgeKey,
    edge: &'a mut Edge<G>,
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
    type Target = Edge<G>;

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
    M: AsStorage<Edge<G>> + AsStorageMut<Edge<G>>,
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

impl<'a, G> FromKeyedSource<(EdgeKey, &'a mut Edge<G>)> for OrphanEdgeView<'a, G>
where
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (EdgeKey, &'a mut Edge<G>)) -> Option<Self> {
        let (key, edge) = source;
        Some(OrphanEdgeView { key, edge })
    }
}

#[derive(Clone, Debug)]
pub struct EdgeKeyTopology {
    key: EdgeKey,
    vertices: (VertexKey, VertexKey),
}

impl EdgeKeyTopology {
    pub fn key(&self) -> EdgeKey {
        self.key
    }

    pub fn vertices(&self) -> (VertexKey, VertexKey) {
        self.vertices
    }
}

impl<M, G> From<EdgeView<M, G>> for EdgeKeyTopology
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
    fn from(edge: EdgeView<M, G>) -> Self {
        EdgeKeyTopology {
            key: edge.key,
            vertices: edge.key.to_vertex_keys(),
        }
    }
}

pub struct VertexCirculator<M, G>
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

impl<M, G> FromKeyedSource<(ArrayVec<[VertexKey; 2]>, M)> for VertexCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (ArrayVec<[VertexKey; 2]>, M)) -> Option<Self> {
        let (input, storage) = source;
        Some(VertexCirculator {
            storage,
            input: input.into_iter(),
            phantom: PhantomData,
        })
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
                // Apply `'a` to the autoref from `reborrow_mut`,
                // `as_storage_mut`, and `get_mut`.
                mem::transmute::<&'_ mut Storage<Vertex<G>>, &'a mut Storage<Vertex<G>>>(
                    self.storage.as_storage_mut(),
                )
            })
                .into_view()
        })
    }
}

pub struct FaceCirculator<M, G>
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

impl<M, G> FromKeyedSource<(ArrayVec<[FaceKey; 2]>, M)> for FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (ArrayVec<[FaceKey; 2]>, M)) -> Option<Self> {
        let (input, storage) = source;
        Some(FaceCirculator {
            storage,
            input: input.into_iter(),
            phantom: PhantomData,
        })
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
                // Apply `'a` to the autoref from `reborrow_mut`,
                // `as_storage_mut`, and `get_mut`.
                mem::transmute::<&'_ mut Storage<Face<G>>, &'a mut Storage<Face<G>>>(
                    self.storage.as_storage_mut(),
                )
            })
                .into_view()
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point2, Point3};

    use geometry::convert::IntoGeometry;
    use geometry::*;
    use graph::*;
    use primitive::cube::Cube;
    use primitive::generate::*;
    use primitive::index::*;

    fn find_vertex_with_geometry<G, T>(graph: &MeshGraph<G>, geometry: T) -> Option<VertexKey>
    where
        G: Geometry,
        G::Vertex: PartialEq,
        T: IntoGeometry<G::Vertex>,
    {
        let geometry = geometry.into_geometry();
        graph
            .vertices()
            .find(|vertex| vertex.geometry == geometry)
            .map(|vertex| vertex.key())
    }

    fn find_edge_with_geometry<G, T>(graph: &MeshGraph<G>, geometry: (T, T)) -> Option<EdgeKey>
    where
        G: Geometry,
        G::Vertex: PartialEq,
        T: IntoGeometry<G::Vertex>,
    {
        let (source, destination) = geometry;
        match (
            find_vertex_with_geometry(graph, source),
            find_vertex_with_geometry(graph, destination),
        ) {
            (Some(source), Some(destination)) => Some((source, destination).into()),
            _ => None,
        }
    }

    #[test]
    fn extrude_edge() {
        let mut graph = MeshGraph::<Point2<f32>>::from_raw_buffers(
            vec![0, 1, 2, 3],
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            4,
        )
        .unwrap();
        let source = find_edge_with_geometry(&graph, ((1.0, 1.0), (1.0, 0.0))).unwrap();
        graph.edge_mut(source).unwrap().extrude(1.0).unwrap();

        assert_eq!(14, graph.edge_count());
        assert_eq!(2, graph.face_count());
    }

    #[test]
    fn join_edges() {
        // Construct a mesh with two independent quads.
        let mut graph = MeshGraph::<Point3<f32>>::from_raw_buffers(
            vec![0, 1, 2, 3, 4, 5, 6, 7],
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
        let source = find_edge_with_geometry(&graph, ((-1.0, 1.0, 0.0), (-1.0, 0.0, 0.0))).unwrap();
        let destination =
            find_edge_with_geometry(&graph, ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0))).unwrap();
        graph.edge_mut(source).unwrap().join(destination).unwrap();

        assert_eq!(20, graph.edge_count());
        assert_eq!(3, graph.face_count());
    }

    #[test]
    fn split_composite_edge() {
        let (indices, vertices) = Cube::new()
            .polygons_with_position() // 6 quads, 24 vertices.
            .flat_index_vertices(HashIndexer::default());
        let mut graph = MeshGraph::<Point3<f32>>::from_raw_buffers(indices, vertices, 4).unwrap();
        let key = graph.edges().nth(0).unwrap().key();
        let vertex = graph.edge_mut(key).unwrap().split().unwrap().into_ref();

        assert_eq!(5, vertex.into_outgoing_edge().into_face().unwrap().arity());
        assert_eq!(
            5,
            vertex
                .into_outgoing_edge()
                .into_opposite_edge()
                .into_face()
                .unwrap()
                .arity()
        );
    }
}
