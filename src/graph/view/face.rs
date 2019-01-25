use arrayvec::ArrayVec;
use fool::prelude::*;
use std::cmp;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Add, Deref, DerefMut, Mul};

use crate::geometry::alias::{ScaledFaceNormal, VertexPosition};
use crate::geometry::convert::AsPosition;
use crate::geometry::Geometry;
use crate::graph::container::alias::OwnedCore;
use crate::graph::container::{Bind, Consistent, Reborrow, ReborrowMut};
use crate::graph::geometry::{FaceCentroid, FaceNormal};
use crate::graph::mutation::face::{
    self, FaceBisectCache, FaceBridgeCache, FaceExtrudeCache, FaceInsertCache, FaceRemoveCache,
    FaceTriangulateCache,
};
use crate::graph::mutation::{Mutate, Mutation};
use crate::graph::storage::convert::{AsStorage, AsStorageMut};
use crate::graph::storage::{EdgeKey, FaceKey, Storage, VertexKey};
use crate::graph::topology::{Edge, Face, Topological, Vertex};
use crate::graph::view::convert::{FromKeyedSource, IntoView};
use crate::graph::view::{EdgeKeyTopology, EdgeView, OrphanEdgeView, OrphanVertexView, VertexView};
use crate::graph::{GraphError, OptionExt};

/// Reference to a face.
///
/// Provides traversals, queries, and mutations related to faces in a mesh. See
/// the module documentation for more information about topological views.
pub struct FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    key: FaceKey,
    storage: M,
    phantom: PhantomData<G>,
}

/// Storage.
impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    // TODO: This may become useful as the `mutation` module is developed. It
    //       may also be necessary to expose this API to user code.
    #[allow(dead_code)]
    pub(in crate::graph) fn bind<T, N>(self, storage: N) -> FaceView<<M as Bind<T, N>>::Output, G>
    where
        T: Topological,
        M: Bind<T, N>,
        M::Output: Reborrow,
        <M::Output as Reborrow>::Target: AsStorage<Face<G>>,
        N: AsStorage<T>,
    {
        let (key, origin) = self.into_keyed_storage();
        FaceView::from_keyed_storage_unchecked(key, origin.bind(storage))
    }
}

impl<'a, M, G> FaceView<&'a mut M, G>
where
    M: 'a + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: 'a + Geometry,
{
    /// Converts a mutable view into an orphan view.
    pub fn into_orphan(self) -> OrphanFaceView<'a, G> {
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
    /// use nalgebra::Point3;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    ///
    /// # fn main() {
    /// let mut graph = Cube::new()
    ///     .polygons_with_position()
    ///     .collect::<MeshGraph<Point3<f32>>>();
    /// let key = graph.faces().nth(0).unwrap().key();
    /// let face = graph
    ///     .face_mut(key)
    ///     .unwrap()
    ///     .extrude(1.0)
    ///     .unwrap()
    ///     .into_ref();
    ///
    /// // This would not be possible without conversion into an immutable view.
    /// let _ = face.into_edge();
    /// let _ = face.into_edge().into_next_edge();
    /// # }
    /// ```
    pub fn into_ref(self) -> FaceView<&'a M, G> {
        let (key, storage) = self.into_keyed_storage();
        (key, &*storage).into_view().unwrap()
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    /// Gets the key for this face.
    pub fn key(&self) -> FaceKey {
        self.key
    }

    fn from_keyed_storage(key: FaceKey, storage: M) -> Option<Self> {
        storage
            .reborrow()
            .as_storage()
            .contains_key(&key)
            .some(FaceView::from_keyed_storage_unchecked(key, storage))
    }

    fn from_keyed_storage_unchecked(key: FaceKey, storage: M) -> Self {
        FaceView {
            key,
            storage,
            phantom: PhantomData,
        }
    }

    fn into_keyed_storage(self) -> (FaceKey, M) {
        let FaceView { key, storage, .. } = self;
        (key, storage)
    }

    fn interior_reborrow(&self) -> FaceView<&M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow();
        FaceView::from_keyed_storage_unchecked(key, storage)
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    fn interior_reborrow_mut(&mut self) -> FaceView<&mut M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow_mut();
        FaceView::from_keyed_storage_unchecked(key, storage)
    }
}

/// Reachable API.
impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_edge(&self) -> Option<EdgeView<&M::Target, G>> {
        let key = self.edge;
        let storage = self.storage.reborrow();
        (key, storage).into_view()
    }

    pub(in crate::graph) fn into_reachable_edge(self) -> Option<EdgeView<M, G>> {
        let key = self.edge;
        let (_, storage) = self.into_keyed_storage();
        (key, storage).into_view()
    }

    pub(in crate::graph) fn reachable_interior_edges(
        &self,
    ) -> impl Iterator<Item = EdgeView<&M::Target, G>> {
        EdgeCirculator::from(self.interior_reborrow())
    }

    pub(in crate::graph) fn reachable_neighboring_faces(
        &self,
    ) -> impl Iterator<Item = FaceView<&M::Target, G>> {
        FaceCirculator::from(EdgeCirculator::from(self.interior_reborrow()))
    }

    pub(in crate::graph) fn reachable_arity(&self) -> usize {
        self.reachable_interior_edges().count()
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + Consistent,
    G: Geometry,
{
    pub fn into_closed_path(self) -> ClosedPath<M, G> {
        let key = self.edge().key();
        let (_, storage) = self.into_keyed_storage();
        (key, storage).into_view().expect_consistent()
    }

    pub fn into_edge(self) -> EdgeView<M, G> {
        self.into_reachable_edge().expect_consistent()
    }

    pub fn closed_path(&self) -> ClosedPath<&M::Target, G> {
        let key = self.edge().key();
        let storage = self.storage.reborrow();
        (key, storage).into_view().expect_consistent()
    }

    pub fn edge(&self) -> EdgeView<&M::Target, G> {
        self.reachable_edge().expect_consistent()
    }

    pub fn interior_edges(&self) -> impl Iterator<Item = EdgeView<&M::Target, G>> {
        self.reachable_interior_edges()
    }

    pub fn neighboring_faces(&self) -> impl Iterator<Item = FaceView<&M::Target, G>> {
        self.reachable_neighboring_faces()
    }

    pub fn arity(&self) -> usize {
        self.reachable_arity()
    }
}

/// Reachable API.
impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_vertices(
        &self,
    ) -> impl Iterator<Item = VertexView<&M::Target, G>> {
        VertexCirculator::from(EdgeCirculator::from(self.interior_reborrow()))
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    G: Geometry,
{
    pub fn to_key_topology(&self) -> FaceKeyTopology {
        FaceKeyTopology::from(self.interior_reborrow())
    }

    pub fn interior_path_distance(
        &self,
        source: VertexKey,
        destination: VertexKey,
    ) -> Result<usize, GraphError> {
        self.interior_reborrow()
            .into_closed_path()
            .interior_path_distance(source, destination)
    }

    pub fn vertices(&self) -> impl Iterator<Item = VertexView<&M::Target, G>> {
        self.reachable_vertices()
    }
}

/// Reachable API.
impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>> + AsStorageMut<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_interior_orphan_edges(
        &mut self,
    ) -> impl Iterator<Item = OrphanEdgeView<G>> {
        EdgeCirculator::from(self.interior_reborrow_mut())
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>> + AsStorageMut<Edge<G>> + AsStorage<Face<G>> + Consistent,
    G: Geometry,
{
    pub fn interior_orphan_edges(&mut self) -> impl Iterator<Item = OrphanEdgeView<G>> {
        self.reachable_interior_orphan_edges()
    }
}

/// Reachable API.
impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_neighboring_orphan_faces(
        &mut self,
    ) -> impl Iterator<Item = OrphanFaceView<G>> {
        FaceCirculator::from(EdgeCirculator::from(self.interior_reborrow_mut()))
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>> + Consistent,
    G: Geometry,
{
    pub fn neighboring_orphan_faces(&mut self) -> impl Iterator<Item = OrphanFaceView<G>> {
        self.reachable_neighboring_orphan_faces()
    }
}

/// Reachable API.
impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target:
        AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_orphan_vertices(
        &mut self,
    ) -> impl Iterator<Item = OrphanVertexView<G>> {
        VertexCirculator::from(EdgeCirculator::from(self.interior_reborrow_mut()))
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + AsStorageMut<Vertex<G>>
        + Consistent,
    G: Geometry,
{
    pub fn orphan_vertices(&mut self) -> impl Iterator<Item = OrphanVertexView<G>> {
        self.reachable_orphan_vertices()
    }
}

impl<'a, M, G> FaceView<&'a mut M, G>
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
    pub fn remove(self) -> Result<ClosedPath<&'a mut M, G>, GraphError> {
        let (abc, storage) = self.into_keyed_storage();
        let cache = FaceRemoveCache::snapshot(&storage, abc)?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| face::remove_with_cache(mutation, cache))
            .map(|(storage, face)| (face.edge, storage).into_view().expect_consistent())
    }

    pub fn bisect(
        self,
        source: VertexKey,
        destination: VertexKey,
    ) -> Result<EdgeView<&'a mut M, G>, GraphError> {
        let (abc, storage) = self.into_keyed_storage();
        let cache = FaceBisectCache::snapshot(&storage, abc, source, destination)?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| face::bisect_with_cache(mutation, cache))
            .map(|(storage, edge)| (edge, storage).into_view().expect_consistent())
    }

    pub fn bridge(self, destination: FaceKey) -> Result<(), GraphError> {
        let (source, storage) = self.into_keyed_storage();
        let cache = FaceBridgeCache::snapshot(&storage, source, destination)?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| face::bridge_with_cache(mutation, cache))
            .map(|_| ())
    }

    pub fn join(self, destination: FaceKey) -> Result<Self, GraphError> {
        let ab = {
            self.interior_edges()
                .find(|edge| match edge.opposite_edge().face() {
                    Some(face) => face.key() == destination,
                    _ => false,
                })
                .map(|edge| edge.key())
                .ok_or_else(|| GraphError::TopologyNotFound)?
        };
        let geometry = self.geometry.clone();
        let (_, storage) = self.into_keyed_storage();
        EdgeView::from_keyed_source((ab, storage))
            .expect_consistent()
            .remove()?
            .into_outgoing_edge()
            .into_closed_path()
            .get_or_insert_face_with(|| geometry)
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    G: FaceCentroid + Geometry,
{
    pub fn centroid(&self) -> Result<G::Centroid, GraphError> {
        G::centroid(self.interior_reborrow())
    }
}

impl<'a, M, G> FaceView<&'a mut M, G>
where
    M: AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Default
        + From<OwnedCore<G>>
        + Into<OwnedCore<G>>,
    G: 'a + FaceCentroid<Centroid = <G as Geometry>::Vertex> + Geometry,
{
    pub fn triangulate(self) -> Result<Option<VertexView<&'a mut M, G>>, GraphError> {
        let (abc, storage) = self.into_keyed_storage();
        let cache = FaceTriangulateCache::snapshot(&storage, abc)?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| face::triangulate_with_cache(mutation, cache))
            .map(|(storage, vertex)| {
                vertex.map(|vertex| (vertex, storage).into_view().expect_consistent())
            })
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    G: FaceNormal + Geometry,
{
    pub fn normal(&self) -> Result<G::Normal, GraphError> {
        G::normal(self.interior_reborrow())
    }
}

impl<'a, M, G> FaceView<&'a mut M, G>
where
    M: AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Default
        + From<OwnedCore<G>>
        + Into<OwnedCore<G>>,
    G: 'a + FaceNormal + Geometry,
    G::Vertex: AsPosition,
{
    pub fn extrude<T>(self, distance: T) -> Result<FaceView<&'a mut M, G>, GraphError>
    where
        G::Normal: Mul<T>,
        ScaledFaceNormal<G, T>: Clone,
        VertexPosition<G>: Add<ScaledFaceNormal<G, T>, Output = VertexPosition<G>> + Clone,
    {
        let (abc, storage) = self.into_keyed_storage();
        let cache = FaceExtrudeCache::snapshot(&storage, abc, distance)?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| face::extrude_with_cache(mutation, cache))
            .map(|(storage, face)| (face, storage).into_view().expect_consistent())
    }
}

impl<M, G> Clone for FaceView<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    fn clone(&self) -> Self {
        FaceView {
            storage: self.storage.clone(),
            key: self.key,
            phantom: PhantomData,
        }
    }
}

impl<M, G> Copy for FaceView<M, G>
where
    M: Copy + Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
}

impl<M, G> Deref for FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    type Target = Face<G>;

    fn deref(&self) -> &Self::Target {
        self.storage.reborrow().as_storage().get(&self.key).unwrap()
    }
}

impl<M, G> DerefMut for FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Face<G>> + AsStorageMut<Face<G>>,
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

impl<M, G> FromKeyedSource<(FaceKey, M)> for FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (FaceKey, M)) -> Option<Self> {
        let (key, storage) = source;
        FaceView::from_keyed_storage(key, storage)
    }
}

/// Orphan reference to a face.
///
/// Consider using `OrphanFace` instead. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
pub struct OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    key: FaceKey,
    face: &'a mut Face<G>,
}

impl<'a, G> OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    pub fn key(&self) -> FaceKey {
        self.key
    }
}

impl<'a, G> Deref for OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    type Target = Face<G>;

    fn deref(&self) -> &Self::Target {
        &*self.face
    }
}

impl<'a, G> DerefMut for OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.face
    }
}

impl<'a, M, G> FromKeyedSource<(FaceKey, &'a mut M)> for OrphanFaceView<'a, G>
where
    M: AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (FaceKey, &'a mut M)) -> Option<Self> {
        let (key, storage) = source;
        storage
            .as_storage_mut()
            .get_mut(&key)
            .map(|face| OrphanFaceView { key, face })
    }
}

impl<'a, G> FromKeyedSource<(FaceKey, &'a mut Face<G>)> for OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (FaceKey, &'a mut Face<G>)) -> Option<Self> {
        let (key, face) = source;
        Some(OrphanFaceView { key, face })
    }
}

#[derive(Clone, Debug)]
pub struct FaceKeyTopology {
    key: FaceKey,
    edges: Vec<EdgeKeyTopology>,
}

impl FaceKeyTopology {
    pub fn key(&self) -> FaceKey {
        self.key
    }

    pub fn interior_edges(&self) -> &[EdgeKeyTopology] {
        self.edges.as_slice()
    }
}

impl<M, G> From<FaceView<M, G>> for FaceKeyTopology
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    G: Geometry,
{
    fn from(face: FaceView<M, G>) -> Self {
        FaceKeyTopology {
            key: face.key,
            edges: face
                .reachable_interior_edges()
                .map(|edge| edge.to_key_topology())
                .collect(),
        }
    }
}

pub struct ClosedPath<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + Consistent,
    G: Geometry,
{
    storage: M,
    edge: EdgeKey,
    face: Option<FaceKey>,
    phantom: PhantomData<G>,
}

impl<'a, M, G> ClosedPath<&'a mut M, G>
where
    M: AsStorage<Edge<G>> + Consistent,
    G: 'a + Geometry,
{
    pub fn into_ref(self) -> ClosedPath<&'a M, G> {
        let (edge, face, storage) = self.into_keyed_storage();
        ClosedPath::from_keyed_storage_unchecked(edge, face, &*storage)
    }
}

impl<M, G> ClosedPath<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + Consistent,
    G: Geometry,
{
    fn from_keyed_storage(key: EdgeKey, storage: M) -> Option<Self> {
        // Because the storage is consistent, this code assumes that any and
        // all edges in the graph will form a loop. Note that this allows
        // exterior edges of non-enclosed meshes to form a region. For
        // conceptually flat meshes, this is odd, but is topologically
        // consistent.
        if let Some(edge) = storage.reborrow().as_storage().get(&key) {
            let face = edge.face.clone();
            Some(ClosedPath {
                storage,
                edge: key,
                face,
                phantom: PhantomData,
            })
        }
        else {
            None
        }
    }

    fn from_keyed_storage_unchecked(edge: EdgeKey, face: Option<FaceKey>, storage: M) -> Self {
        ClosedPath {
            storage,
            edge,
            face,
            phantom: PhantomData,
        }
    }

    fn into_keyed_storage(self) -> (EdgeKey, Option<FaceKey>, M) {
        let ClosedPath {
            storage,
            edge,
            face,
            ..
        } = self;
        (edge, face, storage)
    }

    pub fn arity(&self) -> usize {
        self.interior_edges().count()
    }

    pub fn interior_edges(&self) -> impl Iterator<Item = EdgeView<&M::Target, G>> {
        EdgeCirculator::from(self.interior_reborrow())
    }

    fn interior_reborrow(&self) -> ClosedPath<&M::Target, G> {
        let edge = self.edge;
        let face = self.face;
        let storage = self.storage.reborrow();
        ClosedPath::from_keyed_storage_unchecked(edge, face, storage)
    }
}

impl<M, G> ClosedPath<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Vertex<G>> + Consistent,
    G: Geometry,
{
    pub fn into_edge(self) -> EdgeView<M, G> {
        let (edge, _, storage) = self.into_keyed_storage();
        (edge, storage).into_view().expect_consistent()
    }

    pub fn edge(&self) -> EdgeView<&M::Target, G> {
        let edge = self.edge;
        let storage = self.storage.reborrow();
        (edge, storage).into_view().expect_consistent()
    }

    pub fn interior_path_distance(
        &self,
        source: VertexKey,
        destination: VertexKey,
    ) -> Result<usize, GraphError> {
        let indices = self
            .vertices()
            .map(|vertex| vertex.key())
            .enumerate()
            .filter(|(_, key)| *key == source || *key == destination)
            .map(|(index, _)| index as isize)
            .collect::<ArrayVec<[isize; 2]>>();
        match indices.len() {
            1 => Ok(0),
            2 => {
                let difference = (indices[0] - indices[1]).abs() as usize;
                Ok(cmp::min(difference, self.arity() - difference))
            }
            _ => Err(GraphError::TopologyNotFound),
        }
    }

    pub fn vertices(&self) -> impl Iterator<Item = VertexView<&M::Target, G>> {
        VertexCirculator::from(EdgeCirculator::from(self.interior_reborrow()))
    }
}

impl<M, G> ClosedPath<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + Consistent,
    G: Geometry,
{
    pub fn into_face(self) -> Option<FaceView<M, G>> {
        let (_, face, storage) = self.into_keyed_storage();
        if let Some(face) = face {
            Some((face, storage).into_view().expect_consistent())
        }
        else {
            None
        }
    }

    pub fn face(&self) -> Option<FaceView<&M::Target, G>> {
        if let Some(face) = self.face {
            let storage = self.storage.reborrow();
            Some((face, storage).into_view().expect_consistent())
        }
        else {
            None
        }
    }
}

impl<'a, M, G> ClosedPath<&'a mut M, G>
where
    M: AsStorage<Vertex<G>>
        + AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + Consistent
        + Default
        + From<OwnedCore<G>>
        + Into<OwnedCore<G>>,
    G: 'a + Geometry,
{
    pub fn get_or_insert_face(self) -> Result<FaceView<&'a mut M, G>, GraphError> {
        self.get_or_insert_face_with(|| Default::default())
    }

    pub fn get_or_insert_face_with<F>(self, f: F) -> Result<FaceView<&'a mut M, G>, GraphError>
    where
        F: FnOnce() -> G::Face,
    {
        if let Some(face) = self.face.clone().take() {
            let (_, _, storage) = self.into_keyed_storage();
            Ok((face, storage).into_view().expect_consistent())
        }
        else {
            let vertices = self
                .vertices()
                .map(|vertex| vertex.key())
                .collect::<Vec<_>>();
            let (_, _, storage) = self.into_keyed_storage();
            let cache = FaceInsertCache::snapshot(&storage, &vertices, (Default::default(), f()))?;
            Mutation::replace(storage, Default::default())
                .commit_with(move |mutation| mutation.insert_face_with_cache(cache))
                .map(|(storage, face)| (face, storage).into_view().expect_consistent())
        }
    }
}

impl<M, G> FromKeyedSource<(EdgeKey, M)> for ClosedPath<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + Consistent,
    G: Geometry,
{
    fn from_keyed_source(source: (EdgeKey, M)) -> Option<Self> {
        let (key, storage) = source;
        ClosedPath::from_keyed_storage(key, storage)
    }
}

struct VertexCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
    input: EdgeCirculator<M, G>,
}

impl<M, G> VertexCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
    fn next(&mut self) -> Option<VertexKey> {
        let ab = self.input.next();
        ab.map(|ab| {
            let (_, b) = ab.into();
            b
        })
    }
}

impl<M, G> From<EdgeCirculator<M, G>> for VertexCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
    fn from(input: EdgeCirculator<M, G>) -> Self {
        VertexCirculator { input }
    }
}

// TODO: This iterator could provide a size hint of `(3, None)`, but this is
//       only the case when the underlying mesh is consistent.
impl<'a, M, G> Iterator for VertexCirculator<&'a M, G>
where
    M: 'a + AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    G: 'a + Geometry,
{
    type Item = VertexView<&'a M, G>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).and_then(|key| (key, self.input.storage).into_view())
    }
}

// TODO: This iterator could provide a size hint of `(3, None)`, but this is
//       only the case when the underlying mesh is consistent.
impl<'a, M, G> Iterator for VertexCirculator<&'a mut M, G>
where
    M: 'a + AsStorage<Edge<G>> + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: 'a + Geometry,
{
    type Item = OrphanVertexView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).and_then(|key| {
            (key, unsafe {
                // Apply `'a` to the autoref from `reborrow_mut`,
                // `as_storage_mut`, and `get_mut`.
                mem::transmute::<&'_ mut Storage<Vertex<G>>, &'a mut Storage<Vertex<G>>>(
                    self.input.storage.as_storage_mut(),
                )
            })
                .into_view()
        })
    }
}

struct EdgeCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
    storage: M,
    edge: Option<EdgeKey>,
    breadcrumb: Option<EdgeKey>,
    phantom: PhantomData<G>,
}

impl<M, G> EdgeCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
    fn next(&mut self) -> Option<EdgeKey> {
        self.edge.and_then(|edge| {
            let next = self
                .storage
                .reborrow()
                .as_storage()
                .get(&edge)
                .and_then(|edge| edge.next);
            self.breadcrumb.map(|_| {
                if self.breadcrumb == next {
                    self.breadcrumb = None;
                }
                else {
                    self.edge = next;
                }
                edge
            })
        })
    }
}

impl<M, G> From<FaceView<M, G>> for EdgeCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    fn from(face: FaceView<M, G>) -> Self {
        let edge = face.edge;
        let (_, storage) = face.into_keyed_storage();
        EdgeCirculator {
            storage,
            edge: Some(edge),
            breadcrumb: Some(edge),
            phantom: PhantomData,
        }
    }
}

impl<M, G> From<ClosedPath<M, G>> for EdgeCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + Consistent,
    G: Geometry,
{
    fn from(region: ClosedPath<M, G>) -> Self {
        let (edge, _, storage) = region.into_keyed_storage();
        EdgeCirculator {
            storage,
            edge: Some(edge),
            breadcrumb: Some(edge),
            phantom: PhantomData,
        }
    }
}

// TODO: This iterator could provide a size hint of `(3, None)`, but this is
//       only the case when the underlying mesh is consistent.
impl<'a, M, G> Iterator for EdgeCirculator<&'a M, G>
where
    M: 'a + AsStorage<Edge<G>>,
    G: 'a + Geometry,
{
    type Item = EdgeView<&'a M, G>;

    fn next(&mut self) -> Option<Self::Item> {
        EdgeCirculator::next(self).and_then(|key| (key, self.storage).into_view())
    }
}

// TODO: This iterator could provide a size hint of `(3, None)`, but this is
//       only the case when the underlying mesh is consistent.
impl<'a, M, G> Iterator for EdgeCirculator<&'a mut M, G>
where
    M: 'a + AsStorage<Edge<G>> + AsStorageMut<Edge<G>>,
    G: 'a + Geometry,
{
    type Item = OrphanEdgeView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        EdgeCirculator::next(self).and_then(|key| {
            (key, unsafe {
                // Apply `'a` to the autoref from `reborrow_mut`,
                // `as_storage_mut`, and `get_mut`.
                mem::transmute::<&'_ mut Storage<Edge<G>>, &'a mut Storage<Edge<G>>>(
                    self.storage.as_storage_mut(),
                )
            })
                .into_view()
        })
    }
}

struct FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
    input: EdgeCirculator<M, G>,
}

impl<M, G> FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
    fn next(&mut self) -> Option<FaceKey> {
        while let Some(ba) = self.input.next().map(|ab| ab.opposite()) {
            if let Some(abc) = self
                .input
                .storage
                .reborrow()
                .as_storage()
                .get(&ba)
                .and_then(|opposite| opposite.face)
            {
                return Some(abc);
            }
            else {
                // Skip edges with no opposing face. This can occur within
                // non-enclosed meshes.
                continue;
            }
        }
        None
    }
}

impl<M, G> From<EdgeCirculator<M, G>> for FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>,
    G: Geometry,
{
    fn from(input: EdgeCirculator<M, G>) -> Self {
        FaceCirculator { input }
    }
}

impl<'a, M, G> Iterator for FaceCirculator<&'a M, G>
where
    M: 'a + AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: 'a + Geometry,
{
    type Item = FaceView<&'a M, G>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).and_then(|key| (key, self.input.storage).into_view())
    }
}

impl<'a, M, G> Iterator for FaceCirculator<&'a mut M, G>
where
    M: 'a + AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: 'a + Geometry,
{
    type Item = OrphanFaceView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).and_then(|key| {
            (key, unsafe {
                // Apply `'a` to the autoref from `reborrow_mut`,
                // `as_storage_mut`, and `get_mut`.
                mem::transmute::<&'_ mut Storage<Face<G>>, &'a mut Storage<Face<G>>>(
                    self.input.storage.as_storage_mut(),
                )
            })
                .into_view()
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point2, Point3};

    use crate::graph::*;
    use crate::primitive::cube::Cube;
    use crate::primitive::generate::*;
    use crate::primitive::index::*;
    use crate::primitive::sphere::UvSphere;
    use crate::*;

    #[test]
    fn circulate_over_edges() {
        let graph = UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f32>>>();
        let face = graph.faces().nth(0).unwrap();

        // All faces should be triangles and should have three edges.
        assert_eq!(3, face.interior_edges().count());
    }

    #[test]
    fn circulate_over_faces() {
        let graph = UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f32>>>();
        let face = graph.faces().nth(0).unwrap();

        // No matter which face is selected, it should have three neighbors.
        assert_eq!(3, face.neighboring_faces().count());
    }

    #[test]
    fn remove_face() {
        let mut graph = UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f32>>>();

        // The graph should begin with 6 faces.
        assert_eq!(6, graph.face_count());

        // Remove a face from the graph.
        let abc = graph.faces().nth(0).unwrap().key();
        {
            let face = graph.face_mut(abc).unwrap();
            assert_eq!(3, face.arity()); // The face should be triangular.

            let path = face.remove().unwrap().into_ref();
            assert_eq!(3, path.arity()); // The path should also be triangular.
        }

        // After the removal, the graph should have only 5 faces.
        assert_eq!(5, graph.face_count());
    }

    #[test]
    fn bisect_face() {
        let mut graph = MeshGraph::<Point2<f32>>::from_raw_buffers_with_arity(
            vec![0u32, 1, 2, 3],
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            4,
        )
        .unwrap();
        let abc = graph.faces().nth(0).unwrap().key();
        let (p, q) = {
            let face = graph.face(abc).unwrap();
            let mut vertices = face.vertices().map(|vertex| vertex.key()).step_by(2);
            (vertices.next().unwrap(), vertices.next().unwrap())
        };
        let edge = graph
            .face_mut(abc)
            .unwrap()
            .bisect(p, q)
            .unwrap()
            .into_ref();

        assert!(edge.face().is_some());
        assert!(edge.opposite_edge().face().is_some());
        assert_eq!(4, graph.vertex_count());
        assert_eq!(10, graph.edge_count());
        assert_eq!(2, graph.face_count());
    }

    #[test]
    fn extrude_face() {
        let mut graph = UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f32>>>();
        {
            let key = graph.faces().nth(0).unwrap().key();
            let face = graph
                .face_mut(key)
                .unwrap()
                .extrude(1.0)
                .unwrap()
                .into_ref();

            // The extruded face, being a triangle, should have three
            // neighboring faces.
            assert_eq!(3, face.neighboring_faces().count());
        }

        assert_eq!(8, graph.vertex_count());
        // The mesh begins with 18 edges. The extrusion adds three quads with
        // four interior edges each, so there are `18 + (3 * 4)` edges.
        assert_eq!(30, graph.edge_count());
        // All faces are triangles and the mesh begins with six such faces. The
        // extruded face remains, in addition to three connective faces, each
        // of which is constructed from quads.
        assert_eq!(9, graph.face_count());
    }

    #[test]
    fn join_faces() {
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

        // Get the keys for the two faces and join them.
        let abc = graph.faces().nth(0).unwrap().key();
        let def = graph.faces().nth(1).unwrap().key();
        graph.face_mut(abc).unwrap().join(def).unwrap();

        // After the removal, the graph should have 1 face.
        assert_eq!(1, graph.face_count());
        assert_eq!(6, graph.faces().nth(0).unwrap().arity());
    }

    #[test]
    fn triangulate_mesh() {
        let (indices, vertices) = Cube::new()
            .polygons_with_position() // 6 quads, 24 vertices.
            .index_vertices(HashIndexer::default());
        let mut graph = MeshGraph::<Point3<f32>>::from_raw_buffers(indices, vertices).unwrap();
        graph.triangulate().unwrap();

        // There are 8 unique vertices and a vertex is added for each quad,
        // yielding `8 + 6` vertices.
        assert_eq!(14, graph.vertex_count());
        assert_eq!(72, graph.edge_count());
        // Each quad becomes a tetrahedron, so 6 quads become 24 triangles.
        assert_eq!(24, graph.face_count());
    }

    #[test]
    fn interior_path_distance() {
        let graph = MeshGraph::<Point2<f32>>::from_raw_buffers_with_arity(
            vec![0u32, 1, 2, 3],
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            4,
        )
        .unwrap();
        let face = graph.faces().nth(0).unwrap();
        let keys = face
            .vertices()
            .map(|vertex| vertex.key())
            .collect::<Vec<_>>();
        assert_eq!(2, face.interior_path_distance(keys[0], keys[2]).unwrap());
        assert_eq!(1, face.interior_path_distance(keys[0], keys[3]).unwrap());
        assert_eq!(0, face.interior_path_distance(keys[0], keys[0]).unwrap());
    }
}
