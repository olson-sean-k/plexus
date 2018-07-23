use arrayvec::ArrayVec;
use failure::Error;
use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Mul};

use geometry::convert::AsPosition;
use geometry::Geometry;
use graph::geometry::alias::{ScaledEdgeLateral, VertexPosition};
use graph::geometry::{EdgeLateral, EdgeMidpoint};
use graph::mesh::Mesh;
use graph::mutation::edge::{self, EdgeExtrudeCache, EdgeJoinCache, EdgeSplitCache};
use graph::mutation::{Commit, Mutation};
use graph::storage::convert::{AsStorage, AsStorageMut};
use graph::storage::{Bind, EdgeKey, VertexKey};
use graph::topology::{Edge, Face, Topological, Vertex};
use graph::view::convert::{FromKeyedSource, IntoView};
use graph::view::{
    Consistency, Consistent, FaceView, Inconsistent, OrphanFaceView, OrphanVertexView, VertexView,
};
use BoolExt;

/// Do **not** use this type directly. Use `EdgeRef` and `EdgeMut` instead.
///
/// This type is only re-exported so that its members are shown in
/// documentation. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
pub struct EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    key: EdgeKey,
    storage: M,
    phantom: PhantomData<(G, C)>,
}

/// Storage.
impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn bind<T, N>(self, storage: N) -> EdgeView<<M as Bind<T, N>>::Output, G, C>
    where
        T: Topological,
        M: Bind<T, N>,
        M::Output: AsStorage<Edge<G>>,
        N: AsStorage<T>,
    {
        let (key, origin) = self.into_keyed_storage();
        EdgeView::from_keyed_storage_unchecked(key, origin.bind(storage))
    }
}

impl<'a, M, G, C> EdgeView<&'a mut M, G, C>
where
    M: 'a + AsStorage<Edge<G>> + AsStorageMut<Edge<G>>,
    G: 'a + Geometry,
    C: Consistency,
{
    pub fn into_orphan(self) -> OrphanEdgeView<'a, G> {
        let (key, storage) = self.into_keyed_storage();
        (key, storage.as_storage_mut().get_mut(&key).unwrap())
            .into_view()
            .unwrap()
    }

    pub fn into_ref(self) -> EdgeView<&'a M, G, Consistent> {
        let (key, storage) = self.into_keyed_storage();
        EdgeView::from_keyed_storage_unchecked(key, &*storage)
    }
}

impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn key(&self) -> EdgeKey {
        self.key
    }

    pub fn to_key_topology(&self) -> EdgeKeyTopology {
        EdgeKeyTopology::new(self.key, self.key.to_vertex_keys())
    }

    pub fn is_boundary_edge(&self) -> bool {
        self.face.is_none()
    }

    pub(in graph) fn from_keyed_storage(key: EdgeKey, storage: M) -> Option<Self> {
        storage
            .as_storage()
            .contains_key(&key)
            .into_some(EdgeView::from_keyed_storage_unchecked(key, storage))
    }

    fn from_keyed_storage_unchecked(key: EdgeKey, storage: M) -> Self {
        EdgeView {
            key,
            storage,
            phantom: PhantomData,
        }
    }

    pub(in graph) fn into_keyed_storage(self) -> (EdgeKey, M) {
        let EdgeView { key, storage, .. } = self;
        (key, storage)
    }
}

/// Reachable API.
impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn into_reachable_boundary_edge(self) -> Option<Self> {
        if self.is_boundary_edge() {
            Some(self)
        }
        else {
            self.into_reachable_opposite_edge()
                .and_then(|opposite| opposite.is_boundary_edge().into_some(opposite))
        }
    }

    pub(in graph) fn into_reachable_opposite_edge(self) -> Option<Self> {
        let key = self.opposite;
        key.and_then(move |key| {
            let (_, storage) = self.into_keyed_storage();
            storage
                .as_storage()
                .contains_key(&key)
                .into_some(EdgeView::<_, _, C>::from_keyed_storage(key, storage).unwrap())
        })
    }

    pub(in graph) fn into_reachable_next_edge(self) -> Option<Self> {
        let key = self.next;
        key.and_then(move |key| {
            let (_, storage) = self.into_keyed_storage();
            storage
                .as_storage()
                .contains_key(&key)
                .into_some(EdgeView::<_, _, C>::from_keyed_storage(key, storage).unwrap())
        })
    }

    pub(in graph) fn into_reachable_previous_edge(self) -> Option<Self> {
        let key = self.previous;
        key.and_then(move |key| {
            let (_, storage) = self.into_keyed_storage();
            storage
                .as_storage()
                .contains_key(&key)
                .into_some(EdgeView::<_, _, C>::from_keyed_storage(key, storage).unwrap())
        })
    }

    pub(in graph) fn reachable_boundary_edge(&self) -> Option<EdgeView<&M, G, C>> {
        if self.is_boundary_edge() {
            Some(EdgeView::from_keyed_storage_unchecked(
                self.key,
                &self.storage,
            ))
        }
        else {
            self.reachable_opposite_edge()
                .and_then(|opposite| opposite.is_boundary_edge().into_some(opposite))
        }
    }

    pub(in graph) fn reachable_opposite_edge(&self) -> Option<EdgeView<&M, G, C>> {
        self.opposite.and_then(|key| {
            let storage = &self.storage;
            storage
                .as_storage()
                .contains_key(&key)
                .into_some(EdgeView::<_, _, C>::from_keyed_storage(key, storage).unwrap())
        })
    }

    pub(in graph) fn reachable_next_edge(&self) -> Option<EdgeView<&M, G, C>> {
        self.next.and_then(|key| {
            let storage = &self.storage;
            storage
                .as_storage()
                .contains_key(&key)
                .into_some(EdgeView::<_, _, C>::from_keyed_storage(key, storage).unwrap())
        })
    }

    pub(in graph) fn reachable_previous_edge(&self) -> Option<EdgeView<&M, G, C>> {
        self.previous.and_then(|key| {
            let storage = &self.storage;
            storage
                .as_storage()
                .contains_key(&key)
                .into_some(EdgeView::<_, _, C>::from_keyed_storage(key, storage).unwrap())
        })
    }
}

impl<M, G> EdgeView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>>,
    G: Geometry,
{
    pub fn into_boundary_edge(self) -> Option<Self> {
        if self.is_boundary_edge() {
            Some(self)
        }
        else {
            let opposite = self.into_opposite_edge();
            opposite.is_boundary_edge().into_some(opposite)
        }
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

    pub fn boundary_edge(&self) -> Option<EdgeView<&Mesh<G>, G, Consistent>> {
        if self.is_boundary_edge() {
            Some(EdgeView::from_keyed_storage_unchecked(
                self.key,
                self.storage.as_ref(),
            ))
        }
        else {
            let opposite = self.opposite_edge();
            opposite.is_boundary_edge().into_some(opposite)
        }
    }

    pub fn opposite_edge(&self) -> EdgeView<&Mesh<G>, G, Consistent> {
        interior_deref!(edge => self.reachable_opposite_edge().unwrap())
    }

    pub fn next_edge(&self) -> EdgeView<&Mesh<G>, G, Consistent> {
        interior_deref!(edge => self.reachable_next_edge().unwrap())
    }

    pub fn previous_edge(&self) -> EdgeView<&Mesh<G>, G, Consistent> {
        interior_deref!(edge => self.reachable_previous_edge().unwrap())
    }
}

impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorageMut<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_opposite_orphan_edge(&mut self) -> Option<OrphanEdgeView<G>> {
        let key = self
            .opposite
            .and_then(|key| self.storage.as_storage().contains_key(&key).into_some(key));
        let edge = self.deref_mut();
        key.map(|key| (key, edge).into_view().unwrap())
    }

    pub(in graph) fn reachable_next_orphan_edge(&mut self) -> Option<OrphanEdgeView<G>> {
        let key = self
            .next
            .and_then(|key| self.storage.as_storage().contains_key(&key).into_some(key));
        let edge = self.deref_mut();
        key.map(|key| (key, edge).into_view().unwrap())
    }

    pub(in graph) fn reachable_previous_orphan_edge(&mut self) -> Option<OrphanEdgeView<G>> {
        let key = self
            .previous
            .and_then(|key| self.storage.as_storage().contains_key(&key).into_some(key));
        let edge = self.deref_mut();
        key.map(|key| (key, edge).into_view().unwrap())
    }

    pub(in graph) fn reachable_boundary_orphan_edge(&mut self) -> Option<OrphanEdgeView<G>> {
        if self.is_boundary_edge() {
            let key = self.key;
            let edge = self.deref_mut();
            Some((key, edge).into_view().unwrap())
        }
        else {
            let key = self
                .reachable_opposite_edge()
                .and_then(|opposite| opposite.is_boundary_edge().into_some(opposite.key()));
            if let Some(key) = key {
                Some(
                    (key, self.storage.as_storage_mut().get_mut(&key).unwrap())
                        .into_view()
                        .unwrap(),
                )
            }
            else {
                None
            }
        }
    }
}

impl<M, G> EdgeView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>> + AsStorage<Edge<G>> + AsStorageMut<Edge<G>>,
    G: Geometry,
{
    pub fn opposite_orphan_edge(&mut self) -> OrphanEdgeView<G> {
        self.reachable_opposite_orphan_edge().unwrap()
    }

    pub fn next_orphan_edge(&mut self) -> OrphanEdgeView<G> {
        self.reachable_next_orphan_edge().unwrap()
    }

    pub fn previous_orphan_edge(&mut self) -> OrphanEdgeView<G> {
        self.reachable_previous_orphan_edge().unwrap()
    }

    pub fn boundary_orphan_edge(&mut self) -> Option<OrphanEdgeView<G>> {
        if self.is_boundary_edge() {
            let key = self.key;
            let edge = self.deref_mut();
            Some((key, edge).into_view().unwrap())
        }
        else {
            self.opposite_edge()
                .is_boundary_edge()
                .into_some(self.opposite_orphan_edge())
        }
    }
}

// Note that there is no reachable API for source and destination vertices.
impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn into_source_vertex(self) -> VertexView<M, G, C> {
        let (key, _) = self.key.to_vertex_keys();
        let (_, storage) = self.into_keyed_storage();
        VertexView::<_, _, C>::from_keyed_storage(key, storage).unwrap()
    }

    pub fn into_destination_vertex(self) -> VertexView<M, G, C> {
        let key = self.vertex;
        let (_, storage) = self.into_keyed_storage();
        VertexView::<_, _, C>::from_keyed_storage(key, storage).unwrap()
    }

    pub fn source_vertex<'a>(&'a self) -> VertexView<&'a M, G, C> {
        let (key, _) = self.key.to_vertex_keys();
        let storage = &self.storage;
        VertexView::<_, _, C>::from_keyed_storage(key, storage).unwrap()
    }

    pub fn destination_vertex<'a>(&'a self) -> VertexView<&'a M, G, C> {
        let key = self.vertex;
        let storage = &self.storage;
        VertexView::<_, _, C>::from_keyed_storage(key, storage).unwrap()
    }
}

// Note that there is no reachable API for source and destination vertices.
impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn source_orphan_vertex(&mut self) -> OrphanVertexView<G> {
        let (key, _) = self.key.to_vertex_keys();
        let vertex = self.storage.as_storage_mut().get_mut(&key).unwrap();
        (key, vertex).into_view().unwrap()
    }

    pub fn destination_orphan_vertex(&mut self) -> OrphanVertexView<G> {
        let key = self.vertex;
        let vertex = self.storage.as_storage_mut().get_mut(&key).unwrap();
        (key, vertex).into_view().unwrap()
    }
}

/// Reachable API.
impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn into_reachable_face(self) -> Option<FaceView<M, G, C>> {
        let key = self.face;
        key.and_then(move |key| {
            let (_, storage) = self.into_keyed_storage();
            AsStorage::<Face<G>>::as_storage(&storage)
                .contains_key(&key)
                .into_some(FaceView::<_, _, C>::from_keyed_storage(key, storage).unwrap())
        })
    }

    pub fn reachable_face(&self) -> Option<FaceView<&M, G, C>> {
        self.face.and_then(|key| {
            let storage = &self.storage;
            AsStorage::<Face<G>>::as_storage(storage)
                .contains_key(&key)
                .into_some(FaceView::<_, _, C>::from_keyed_storage(key, storage).unwrap())
        })
    }
}

impl<M, G> EdgeView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    pub fn into_face(self) -> Option<FaceView<M, G, Consistent>> {
        let key = self.face;
        key.map(move |key| {
            let (_, storage) = self.into_keyed_storage();
            FaceView::<_, _, Consistent>::from_keyed_storage(key, storage).unwrap()
        })
    }

    pub fn face(&self) -> Option<FaceView<&Mesh<G>, G, Consistent>> {
        self.reachable_face()
            .map(|face| interior_deref!(face => face))
    }
}

impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn orphan_face(&mut self) -> Option<OrphanFaceView<G>> {
        if let Some(key) = self.face {
            if let Some(face) = self.storage.as_storage_mut().get_mut(&key) {
                return Some((key, face).into_view().unwrap());
            }
        }
        return None;
    }
}

impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_vertices(&self) -> impl Iterator<Item = VertexView<&M, G, C>> {
        let (a, b) = self.key.to_vertex_keys();
        let storage = &self.storage;
        ArrayVec::from([b, a])
            .into_iter()
            .map(move |key| VertexView::<_, _, C>::from_keyed_storage(key, storage).unwrap())
    }
}

impl<M, G> EdgeView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub fn vertices(&self) -> impl Iterator<Item = VertexView<&Mesh<G>, G, Consistent>> {
        self.reachable_vertices()
            .map(|vertex| interior_deref!(vertex => vertex))
    }
}

impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_faces(&self) -> impl Iterator<Item = FaceView<&M, G, C>> {
        let storage = &self.storage;
        self.face
            .into_iter()
            .chain(
                self.reachable_opposite_edge()
                    .and_then(|opposite| opposite.face),
            )
            .map(move |key| FaceView::<_, _, C>::from_keyed_storage(key, storage).unwrap())
    }
}

impl<M, G> EdgeView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    pub fn faces(&self) -> impl Iterator<Item = FaceView<&Mesh<G>, G, Consistent>> {
        self.reachable_faces()
            .map(|face| interior_deref!(face => face))
    }
}

impl<'a, G> EdgeView<&'a mut Mesh<G>, G, Consistent>
where
    G: Geometry,
{
    // TODO: Rename this to something like "extend". It is very similar to
    //       `extrude`. Terms like "join" or "merge" are better suited for
    //       directly joining two adjacent faces over a shared edge.
    pub fn join(
        self,
        destination: EdgeKey,
    ) -> Result<EdgeView<&'a mut Mesh<G>, G, Consistent>, Error> {
        let (source, storage) = self.into_keyed_storage();
        let cache = EdgeJoinCache::snapshot(&storage, source, destination)?;
        let (storage, edge) = Mutation::replace(storage, Mesh::empty())
            .commit_with(move |mutation| edge::join_with_cache(&mut *mutation, cache))
            .unwrap();
        Ok(EdgeView::<_, _, Consistent>::from_keyed_storage(edge, storage).unwrap())
    }
}

impl<M, G> EdgeView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: EdgeMidpoint + Geometry,
{
    pub fn midpoint(&self) -> Result<G::Midpoint, Error> {
        G::midpoint(self)
    }
}

impl<'a, G> EdgeView<&'a mut Mesh<G>, G, Consistent>
where
    G: EdgeMidpoint + Geometry,
    G::Vertex: AsPosition,
{
    pub fn split(self) -> Result<VertexView<&'a mut Mesh<G>, G, Consistent>, Error>
    where
        G: EdgeMidpoint<Midpoint = VertexPosition<G>>,
    {
        let (ab, storage) = self.into_keyed_storage();
        let cache = EdgeSplitCache::snapshot(&storage, ab)?;
        let (storage, vertex) = Mutation::replace(storage, Mesh::empty())
            .commit_with(move |mutation| edge::split_with_cache(&mut *mutation, cache))
            .unwrap();
        Ok(VertexView::<_, _, Consistent>::from_keyed_storage(vertex, storage).unwrap())
    }
}

impl<M, G> EdgeView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: Geometry + EdgeLateral,
{
    pub fn lateral(&self) -> Result<G::Lateral, Error> {
        G::lateral(self)
    }
}

impl<'a, G> EdgeView<&'a mut Mesh<G>, G, Consistent>
where
    G: Geometry + EdgeLateral,
    G::Vertex: AsPosition,
{
    pub fn extrude<T>(self, distance: T) -> Result<EdgeView<&'a mut Mesh<G>, G, Consistent>, Error>
    where
        G::Lateral: Mul<T>,
        ScaledEdgeLateral<G, T>: Clone,
        VertexPosition<G>: Add<ScaledEdgeLateral<G, T>, Output = VertexPosition<G>> + Clone,
    {
        let (ab, storage) = self.into_keyed_storage();
        let cache = EdgeExtrudeCache::snapshot(storage, ab, distance)?;
        let (storage, edge) = Mutation::replace(storage, Mesh::empty())
            .commit_with(move |mutation| edge::extrude_with_cache(&mut *mutation, cache))
            .unwrap();
        Ok(EdgeView::<_, _, Consistent>::from_keyed_storage(edge, storage).unwrap())
    }
}

impl<M, G, C> Clone for EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + Clone,
    G: Geometry,
    C: Consistency,
{
    fn clone(&self) -> Self {
        EdgeView {
            key: self.key,
            storage: self.storage.clone(),
            phantom: PhantomData,
        }
    }
}

impl<M, G, C> Copy for EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + Copy,
    G: Geometry,
    C: Consistency,
{}

impl<M, G, C> Deref for EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    type Target = Edge<G>;

    fn deref(&self) -> &Self::Target {
        self.storage.as_storage().get(&self.key).unwrap()
    }
}

impl<M, G, C> DerefMut for EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorageMut<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.storage.as_storage_mut().get_mut(&self.key).unwrap()
    }
}

impl<M, G> FromKeyedSource<(EdgeKey, M)> for EdgeView<M, G, Inconsistent>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (EdgeKey, M)) -> Option<Self> {
        let (key, storage) = source;
        EdgeView::<_, _, Inconsistent>::from_keyed_storage(key, storage)
    }
}

impl<M, G> FromKeyedSource<(EdgeKey, M)> for EdgeView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (EdgeKey, M)) -> Option<Self> {
        let (key, storage) = source;
        EdgeView::<_, _, Consistent>::from_keyed_storage(key, storage)
    }
}

/// Do **not** use this type directly. Use `OrphanEdge` instead.
///
/// This type is only re-exported so that its members are shown in
/// documentation. See this issue:
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
    fn from_keyed_storage(key: EdgeKey, edge: &'a mut Edge<G>) -> Self {
        OrphanEdgeView { key, edge }
    }

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

impl<'a, G> FromKeyedSource<(EdgeKey, &'a mut Edge<G>)> for OrphanEdgeView<'a, G>
where
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (EdgeKey, &'a mut Edge<G>)) -> Option<Self> {
        let (key, edge) = source;
        Some(OrphanEdgeView::from_keyed_storage(key, edge))
    }
}

#[derive(Clone, Debug)]
pub struct EdgeKeyTopology {
    key: EdgeKey,
    vertices: (VertexKey, VertexKey),
}

impl EdgeKeyTopology {
    fn new(edge: EdgeKey, vertices: (VertexKey, VertexKey)) -> Self {
        EdgeKeyTopology {
            key: edge,
            vertices: vertices,
        }
    }

    pub fn key(&self) -> EdgeKey {
        self.key
    }

    pub fn vertices(&self) -> (VertexKey, VertexKey) {
        self.vertices
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point2, Point3};

    use generate::*;
    use geometry::convert::IntoGeometry;
    use geometry::*;
    use graph::*;

    fn find_vertex_with_geometry<G, T>(mesh: &Mesh<G>, geometry: T) -> Option<VertexKey>
    where
        G: Geometry,
        G::Vertex: PartialEq,
        T: IntoGeometry<G::Vertex>,
    {
        let geometry = geometry.into_geometry();
        mesh.vertices()
            .find(|vertex| vertex.geometry == geometry)
            .map(|vertex| vertex.key())
    }

    fn find_edge_with_geometry<G, T>(mesh: &Mesh<G>, geometry: (T, T)) -> Option<EdgeKey>
    where
        G: Geometry,
        G::Vertex: PartialEq,
        T: IntoGeometry<G::Vertex>,
    {
        let (source, destination) = geometry;
        match (
            find_vertex_with_geometry(mesh, source),
            find_vertex_with_geometry(mesh, destination),
        ) {
            (Some(source), Some(destination)) => Some((source, destination).into()),
            _ => None,
        }
    }

    #[test]
    fn extrude_edge() {
        let mut mesh = Mesh::<Point2<f32>>::from_raw_buffers(
            vec![0, 1, 2, 3],
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            4,
        ).unwrap();
        let source = find_edge_with_geometry(&mesh, ((1.0, 1.0), (1.0, 0.0))).unwrap();
        mesh.edge_mut(source).unwrap().extrude(1.0).unwrap();

        assert_eq!(14, mesh.edge_count());
        assert_eq!(2, mesh.face_count());
    }

    #[test]
    fn join_edges() {
        // Construct a mesh with two independent quads.
        let mut mesh = Mesh::<Point3<f32>>::from_raw_buffers(
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
        ).unwrap();
        let source = find_edge_with_geometry(&mesh, ((-1.0, 1.0, 0.0), (-1.0, 0.0, 0.0))).unwrap();
        let destination =
            find_edge_with_geometry(&mesh, ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0))).unwrap();
        mesh.edge_mut(source).unwrap().join(destination).unwrap();

        assert_eq!(20, mesh.edge_count());
        assert_eq!(3, mesh.face_count());
    }

    #[test]
    fn split_composite_edge() {
        let (indeces, vertices) = cube::Cube::new()
            .polygons_with_position() // 6 quads, 24 vertices.
            .flat_index_vertices(HashIndexer::default());
        let mut mesh = Mesh::<Point3<f32>>::from_raw_buffers(indeces, vertices, 4).unwrap();
        let key = mesh.edges().nth(0).unwrap().key();
        let vertex = mesh.edge_mut(key).unwrap().split().unwrap().into_ref();

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
