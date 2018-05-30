use arrayvec::ArrayVec;
use failure::Error;
use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Mul};

use geometry::convert::AsPosition;
use geometry::Geometry;
use graph::geometry::alias::{ScaledEdgeLateral, VertexPosition};
use graph::geometry::{EdgeLateral, EdgeMidpoint};
use graph::mesh::{Edge, Face, Mesh, Vertex};
use graph::mutation::{ModalMutation, Mutation};
use graph::storage::convert::{AsStorage, AsStorageMut};
use graph::storage::{Bind, EdgeKey, FaceKey, Storage, Topological, VertexKey};
use graph::view::convert::{FromKeyedSource, IntoView};
use graph::view::{
    Consistency, Consistent, FaceView, Inconsistent, OrphanFaceView, OrphanVertexView, VertexView,
};
use graph::{GraphError, Perimeter};
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
        let EdgeView {
            key,
            storage: origin,
            ..
        } = self;
        EdgeView {
            key,
            storage: origin.bind(storage),
            phantom: PhantomData,
        }
    }

    pub(in graph) fn as_storage<T>(&self) -> &Storage<T>
    where
        T: Topological,
        M: AsStorage<T>,
    {
        AsStorage::<T>::as_storage(&self.storage)
    }

    pub(in graph) fn as_storage_mut<T>(&mut self) -> &mut Storage<T>
    where
        T: Topological,
        M: AsStorageMut<T>,
    {
        AsStorageMut::<T>::as_storage_mut(&mut self.storage)
    }
}

impl<'a, 'b, M, G, C> EdgeView<&'a &'b M, G, C>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn into_interior_deref(self) -> EdgeView<&'b M, G, C>
    where
        (EdgeKey, &'b M): IntoView<EdgeView<&'b M, G, C>>,
    {
        let key = self.key;
        let storage = *self.storage;
        (key, storage).into_view()
    }

    pub fn interior_deref(&self) -> EdgeView<&'b M, G, C>
    where
        (EdgeKey, &'b M): IntoView<EdgeView<&'b M, G, C>>,
    {
        let key = self.key;
        let storage = *self.storage;
        (key, storage).into_view()
    }
}

impl<'a, M, G, C> EdgeView<&'a mut M, G, C>
where
    M: 'a + AsStorage<Edge<G>> + AsStorageMut<Edge<G>>,
    G: 'a + Geometry,
    C: Consistency,
{
    pub fn into_orphan(self) -> OrphanEdgeView<'a, G> {
        let EdgeView { key, storage, .. } = self;
        (key, storage.as_storage_mut().get_mut(&key).unwrap()).into_view()
    }
}

impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorageMut<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn to_orphan<'a>(&'a mut self) -> OrphanEdgeView<'a, G> {
        let key = self.key;
        (key, self.storage.as_storage_mut().get_mut(&key).unwrap()).into_view()
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

    pub(in graph) fn from_keyed_storage(key: EdgeKey, storage: M) -> Self {
        EdgeView {
            key,
            storage,
            phantom: PhantomData,
        }
    }

    fn to_ref(&self) -> EdgeView<&M, G, C> {
        let key = self.key;
        let storage = &self.storage;
        EdgeView::from_keyed_storage(key, storage)
    }

    fn to_mut(&mut self) -> EdgeView<&mut M, G, C> {
        let key = self.key;
        let storage = &mut self.storage;
        EdgeView::from_keyed_storage(key, storage)
    }
}

impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn into_reachable_boundary_edge(self) -> Option<Self>
    where
        (EdgeKey, M): IntoView<Self>,
    {
        if self.is_boundary_edge() {
            Some(self)
        }
        else {
            self.into_reachable_opposite_edge()
                .and_then(|opposite| opposite.is_boundary_edge().into_some(opposite))
        }
    }

    pub(in graph) fn into_reachable_opposite_edge(self) -> Option<Self>
    where
        (EdgeKey, M): IntoView<Self>,
    {
        let key = self.opposite;
        key.map(move |key| {
            let EdgeView { storage, .. } = self;
            (key, storage).into_view()
        })
    }

    pub(in graph) fn into_reachable_next_edge(self) -> Option<Self>
    where
        (EdgeKey, M): IntoView<Self>,
    {
        let key = self.next;
        key.map(move |key| {
            let EdgeView { storage, .. } = self;
            (key, storage).into_view()
        })
    }

    pub(in graph) fn into_reachable_previous_edge(self) -> Option<Self>
    where
        (EdgeKey, M): IntoView<Self>,
    {
        let key = self.previous;
        key.map(move |key| {
            let EdgeView { storage, .. } = self;
            (key, storage).into_view()
        })
    }

    pub(in graph) fn reachable_boundary_edge<'a>(&'a self) -> Option<EdgeView<&'a M, G, C>>
    where
        (EdgeKey, &'a M): IntoView<EdgeView<&'a M, G, C>>,
    {
        if self.is_boundary_edge() {
            Some((self.key, &self.storage).into_view())
        }
        else {
            self.reachable_opposite_edge()
                .and_then(|opposite| opposite.is_boundary_edge().into_some(opposite))
        }
    }

    pub(in graph) fn reachable_opposite_edge<'a>(&'a self) -> Option<EdgeView<&'a M, G, C>>
    where
        (EdgeKey, &'a M): IntoView<EdgeView<&'a M, G, C>>,
    {
        self.opposite.map(|key| {
            let storage = &self.storage;
            (key, storage).into_view()
        })
    }

    pub(in graph) fn reachable_next_edge<'a>(&'a self) -> Option<EdgeView<&'a M, G, C>>
    where
        (EdgeKey, &'a M): IntoView<EdgeView<&'a M, G, C>>,
    {
        self.next.map(|key| {
            let storage = &self.storage;
            (key, storage).into_view()
        })
    }

    pub(in graph) fn reachable_previous_edge<'a>(&'a self) -> Option<EdgeView<&'a M, G, C>>
    where
        (EdgeKey, &'a M): IntoView<EdgeView<&'a M, G, C>>,
    {
        self.previous.map(|key| {
            let storage = &self.storage;
            (key, storage).into_view()
        })
    }
}

impl<M, G> EdgeView<M, G, Consistent>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
{
    pub fn into_boundary_edge(self) -> Option<Self>
    where
        (EdgeKey, M): IntoView<Self>,
    {
        if self.is_boundary_edge() {
            Some(self)
        }
        else {
            let opposite = self.into_opposite_edge();
            opposite.is_boundary_edge().into_some(opposite)
        }
    }

    pub fn into_opposite_edge(self) -> Self
    where
        (EdgeKey, M): IntoView<Self>,
    {
        self.into_reachable_opposite_edge().unwrap()
    }

    pub fn into_next_edge(self) -> Self
    where
        (EdgeKey, M): IntoView<Self>,
    {
        self.into_reachable_next_edge().unwrap()
    }

    pub fn into_previous_edge(self) -> Self
    where
        (EdgeKey, M): IntoView<Self>,
    {
        self.into_reachable_previous_edge().unwrap()
    }

    pub fn boundary_edge<'a>(&'a self) -> Option<EdgeView<&'a M, G, Consistent>>
    where
        (EdgeKey, &'a M): IntoView<EdgeView<&'a M, G, Consistent>>,
    {
        if self.is_boundary_edge() {
            Some((self.key, &self.storage).into_view())
        }
        else {
            let opposite = self.opposite_edge();
            opposite.is_boundary_edge().into_some(opposite)
        }
    }

    pub fn opposite_edge<'a>(&'a self) -> EdgeView<&'a M, G, Consistent>
    where
        (EdgeKey, &'a M): IntoView<EdgeView<&'a M, G, Consistent>>,
    {
        self.reachable_opposite_edge().unwrap()
    }

    pub fn next_edge<'a>(&'a self) -> EdgeView<&'a M, G, Consistent>
    where
        (EdgeKey, &'a M): IntoView<EdgeView<&'a M, G, Consistent>>,
    {
        self.reachable_next_edge().unwrap()
    }

    pub fn previous_edge<'a>(&'a self) -> EdgeView<&'a M, G, Consistent>
    where
        (EdgeKey, &'a M): IntoView<EdgeView<&'a M, G, Consistent>>,
    {
        self.reachable_previous_edge().unwrap()
    }
}

impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorageMut<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_opposite_orphan_edge<'a>(
        &'a mut self,
    ) -> Option<OrphanEdgeView<'a, G>> {
        let key = self.opposite;
        let edge = self.deref_mut();
        key.map(|key| (key, edge).into_view())
    }

    pub(in graph) fn reachable_next_orphan_edge<'a>(&'a mut self) -> Option<OrphanEdgeView<'a, G>> {
        let key = self.next;
        let edge = self.deref_mut();
        key.map(|key| (key, edge).into_view())
    }

    pub(in graph) fn reachable_previous_orphan_edge<'a>(
        &'a mut self,
    ) -> Option<OrphanEdgeView<'a, G>> {
        let key = self.previous;
        let edge = self.deref_mut();
        key.map(|key| (key, edge).into_view())
    }

    pub(in graph) fn reachable_boundary_orphan_edge<'a>(
        &'a mut self,
    ) -> Option<OrphanEdgeView<'a, G>>
    where
        (EdgeKey, &'a M): IntoView<EdgeView<&'a M, G, C>>,
    {
        if self.is_boundary_edge() {
            let key = self.key;
            let edge = self.deref_mut();
            Some((key, edge).into_view())
        }
        else {
            unimplemented!() // TODO:
                             //let opposite = self.reachable_opposite_edge();
                             //opposite.and_then(|opposite| {
                             //    opposite
                             //        .is_boundary_edge()
                             //        .into_some(opposite.into_orphan())
                             //})
        }
    }
}

impl<M, G> EdgeView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>> + AsStorageMut<Edge<G>>,
    G: Geometry,
{
    pub fn opposite_orphan_edge<'a>(&'a mut self) -> OrphanEdgeView<'a, G> {
        self.reachable_opposite_orphan_edge().unwrap()
    }

    pub fn next_orphan_edge<'a>(&'a mut self) -> OrphanEdgeView<'a, G> {
        self.reachable_next_orphan_edge().unwrap()
    }

    pub fn previous_orphan_edge<'a>(&'a mut self) -> OrphanEdgeView<'a, G> {
        self.reachable_previous_orphan_edge().unwrap()
    }

    pub fn boundary_orphan_edge<'a>(&'a mut self) -> Option<OrphanEdgeView<'a, G>> {
        if self.is_boundary_edge() {
            let key = self.key;
            let edge = self.deref_mut();
            Some((key, edge).into_view())
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
    pub fn into_source_vertex(self) -> VertexView<M, G, C>
    where
        (VertexKey, M): IntoView<VertexView<M, G, C>>,
    {
        let (key, _) = self.key.to_vertex_keys();
        let EdgeView { storage, .. } = self;
        (key, storage).into_view()
    }

    pub fn into_destination_vertex(self) -> VertexView<M, G, C>
    where
        (VertexKey, M): IntoView<VertexView<M, G, C>>,
    {
        let key = self.vertex;
        let EdgeView { storage, .. } = self;
        (key, storage).into_view()
    }

    pub fn source_vertex<'a>(&'a self) -> VertexView<&'a M, G, C>
    where
        (VertexKey, &'a M): IntoView<VertexView<&'a M, G, C>>,
    {
        let (key, _) = self.key.to_vertex_keys();
        let storage = &self.storage;
        (key, storage).into_view()
    }

    pub fn destination_vertex<'a>(&'a self) -> VertexView<&'a M, G, C>
    where
        (VertexKey, &'a M): IntoView<VertexView<&'a M, G, C>>,
    {
        let key = self.vertex;
        let storage = &self.storage;
        (key, storage).into_view()
    }
}

// Note that there is no reachable API for source and destination vertices.
impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn source_orphan_vertex<'a>(&'a mut self) -> OrphanVertexView<'a, G> {
        let (key, _) = self.key.to_vertex_keys();
        let vertex = self.storage.as_storage_mut().get_mut(&key).unwrap();
        (key, vertex).into_view()
    }

    pub fn destination_orphan_vertex<'a>(&'a mut self) -> OrphanVertexView<'a, G> {
        let key = self.vertex;
        let vertex = self.storage.as_storage_mut().get_mut(&key).unwrap();
        (key, vertex).into_view()
    }
}

impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn into_face(self) -> Option<FaceView<M, G, C>>
    where
        (FaceKey, M): IntoView<FaceView<M, G, C>>,
    {
        let key = self.face;
        key.map(move |key| {
            let EdgeView { storage, .. } = self;
            (key, storage).into_view()
        })
    }

    pub fn face<'a>(&'a self) -> Option<FaceView<&'a M, G, C>>
    where
        (FaceKey, &'a M): IntoView<FaceView<&'a M, G, C>>,
    {
        self.face.map(|key| {
            let storage = &self.storage;
            (key, storage).into_view()
        })
    }
}

impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn orphan_face<'a>(&'a mut self) -> Option<OrphanFaceView<'a, G>> {
        if let Some(key) = self.face {
            if let Some(face) = self.storage.as_storage_mut().get_mut(&key) {
                return Some((key, face).into_view());
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
    pub(in graph) fn reachable_vertices<'a>(
        &'a self,
    ) -> impl Iterator<Item = VertexView<&'a M, G, C>>
    where
        (VertexKey, &'a M): IntoView<VertexView<&'a M, G, C>>,
    {
        let (a, b) = self.key.to_vertex_keys();
        let storage = &self.storage;
        ArrayVec::from([b, a])
            .into_iter()
            .map(move |key| (key, storage).into_view())
    }
}

impl<M, G> EdgeView<M, G, Consistent>
where
    M: AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub fn vertices<'a>(&'a self) -> impl Iterator<Item = VertexView<&'a M, G, Consistent>>
    where
        (VertexKey, &'a M): IntoView<VertexView<&'a M, G, Consistent>>,
    {
        self.reachable_vertices()
    }
}

impl<M, G, C> EdgeView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_faces<'a>(&'a self) -> impl Iterator<Item = FaceView<&'a M, G, C>>
    where
        (EdgeKey, &'a M): IntoView<EdgeView<&'a M, G, C>>,
        (FaceKey, &'a M): IntoView<FaceView<&'a M, G, C>>,
    {
        let storage = &self.storage;
        self.face
            .into_iter()
            .chain(
                self.reachable_opposite_edge()
                    .and_then(|opposite| opposite.face),
            )
            .map(move |key| (key, storage).into_view())
    }
}

impl<M, G> EdgeView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    pub fn faces<'a>(&'a self) -> impl Iterator<Item = FaceView<&'a M, G, Consistent>>
    where
        (FaceKey, &'a M): IntoView<FaceView<&'a M, G, Consistent>>,
    {
        self.reachable_faces()
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
        let EdgeView {
            storage,
            key: source,
            ..
        } = self;
        let mut mutation = Mutation::immediate(storage);
        let edge = join(&mut mutation, source, destination)?;
        Ok((edge, mutation.commit()).into_view())
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
        let EdgeView { storage, key, .. } = self;
        let mut mutation = Mutation::immediate(storage);
        let vertex = split(&mut mutation, key)?;
        Ok((vertex, mutation.commit()).into_view())
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
        let EdgeView { storage, key, .. } = self;
        let mut mutation = Mutation::immediate(storage);
        let edge = extrude(&mut mutation, key, distance)?;
        Ok((edge, mutation.commit()).into_view())
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
{
}

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
    fn from_keyed_source(source: (EdgeKey, M)) -> Self {
        let (key, storage) = source;
        EdgeView {
            key,
            storage,
            phantom: PhantomData,
        }
    }
}

impl<M, G> FromKeyedSource<(EdgeKey, M)> for EdgeView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (EdgeKey, M)) -> Self {
        let (key, storage) = source;
        EdgeView {
            key,
            storage,
            phantom: PhantomData,
        }
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
    fn from_keyed_source(source: (EdgeKey, &'a mut Edge<G>)) -> Self {
        let (key, edge) = source;
        OrphanEdgeView { key, edge }
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

pub(in graph) fn split<'a, M, G>(mutation: &mut M, ab: EdgeKey) -> Result<VertexKey, Error>
where
    M: ModalMutation<'a, G>,
    G: 'a + EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
    G::Vertex: AsPosition,
{
    fn split_at_vertex<'a, M, G>(
        mutation: &mut M,
        ab: EdgeKey,
        m: VertexKey,
    ) -> Result<(EdgeKey, EdgeKey), Error>
    where
        M: ModalMutation<'a, G>,
        G: 'a + EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
        G::Vertex: AsPosition,
    {
        // Remove the edge and insert two truncated edges in its place.
        let (a, b) = ab.to_vertex_keys();
        let span = mutation.remove_edge(ab).unwrap();
        let am = mutation.insert_edge((a, m), span.geometry.clone())?;
        let mb = mutation.insert_edge((m, b), span.geometry.clone())?;
        // Connect the new edges to each other and their leading edges.
        {
            let mut edge = mutation.as_mesh_mut().edge_mut(am).unwrap();
            edge.next = Some(mb);
            edge.previous = span.previous;
            edge.face = span.face
        }
        {
            let mut edge = mutation.as_mesh_mut().edge_mut(mb).unwrap();
            edge.next = span.next;
            edge.previous = Some(am);
            edge.face = span.face;
        }
        if let Some(pa) = span.previous {
            mutation.as_mesh_mut().edge_mut(pa).unwrap().next = Some(am);
        }
        if let Some(bn) = span.next {
            mutation.as_mesh_mut().edge_mut(bn).unwrap().previous = Some(mb);
        }
        // Update the associated face, if any, because it may refer to the
        // removed edge.
        if let Some(face) = span.face {
            mutation.as_mesh_mut().face_mut(face).unwrap().edge = am;
        }
        Ok((am, mb))
    }

    let (ba, m) = {
        // Insert a new vertex at the midpoint.
        let (ba, midpoint) = {
            let edge = match mutation.as_mesh().edge(ab) {
                Some(edge) => edge,
                _ => return Err(GraphError::TopologyNotFound.into()),
            };
            let mut midpoint = edge.source_vertex().geometry.clone();
            *midpoint.as_position_mut() = edge.midpoint()?;
            (
                edge.reachable_opposite_edge()
                    .map(|opposite| opposite.key()),
                midpoint,
            )
        };
        (ba, mutation.insert_vertex(midpoint))
    };
    // Split the half-edges. This should not fail; unwrap the results.
    split_at_vertex(mutation, ab, m).unwrap();
    if let Some(ba) = ba {
        split_at_vertex(mutation, ba, m).unwrap();
    }
    Ok(m)
}

pub(in graph) fn join<'a, M, G>(
    mutation: &mut M,
    source: EdgeKey,
    destination: EdgeKey,
) -> Result<EdgeKey, Error>
where
    M: ModalMutation<'a, G>,
    G: 'a + Geometry,
{
    match (
        mutation.as_mesh().edge(source),
        mutation.as_mesh().edge(destination),
    ) {
        (Some(_), Some(_)) => {}
        _ => return Err(GraphError::TopologyNotFound.into()),
    }
    let (a, b) = source.to_vertex_keys();
    let (c, d) = destination.to_vertex_keys();
    // At this point, we can assume the points a, b, c, and d exist in the
    // mesh. Before mutating the mesh, ensure that existing interior edges
    // are boundaries.
    for edge in [a, b, c, d]
        .perimeter()
        .flat_map(|ab| mutation.as_mesh().edge(ab.into()))
    {
        if !edge.is_boundary_edge() {
            return Err(GraphError::TopologyConflict.into());
        }
    }
    // Insert a quad joining the edges. These operations should not fail;
    // unwrap their results.
    let (edge, face) = {
        let source = mutation.as_mesh().edge(source).unwrap();
        (
            source.geometry.clone(),
            source
                .opposite_edge()
                .face()
                .map(|face| face.geometry.clone())
                .unwrap_or_else(Default::default),
        )
    };
    // TODO: Split the face to form triangles.
    mutation.insert_face(&[a, b, c, d], (edge, face)).unwrap();
    Ok(source)
}

pub(in graph) fn extrude<'a, M, G, T>(
    mutation: &mut M,
    ab: EdgeKey,
    distance: T,
) -> Result<EdgeKey, Error>
where
    M: ModalMutation<'a, G>,
    G: 'a + Geometry + EdgeLateral,
    G::Lateral: Mul<T>,
    G::Vertex: AsPosition,
    ScaledEdgeLateral<G, T>: Clone,
    VertexPosition<G>: Add<ScaledEdgeLateral<G, T>, Output = VertexPosition<G>> + Clone,
{
    // Get the extruded geometry.
    let (vertices, edge) = {
        let edge = match mutation.as_mesh().edge(ab) {
            Some(edge) => edge,
            _ => return Err(GraphError::TopologyNotFound.into()),
        };
        if !edge.is_boundary_edge() {
            return Err(GraphError::TopologyConflict.into());
        }
        let mut vertices = (
            edge.destination_vertex().geometry.clone(),
            edge.source_vertex().geometry.clone(),
        );
        let translation = edge.lateral()? * distance;
        *vertices.0.as_position_mut() = vertices.0.as_position().clone() + translation.clone();
        *vertices.1.as_position_mut() = vertices.1.as_position().clone() + translation;
        (vertices, edge.geometry.clone())
    };
    let c = mutation.insert_vertex(vertices.0);
    let d = mutation.insert_vertex(vertices.1);
    let cd = mutation.insert_edge((c, d), edge).unwrap();
    Ok(join(mutation, ab, cd).unwrap())
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
        let vertex = mesh.edge_mut(key).unwrap().split().unwrap();

        assert_eq!(5, vertex.outgoing_edge().face().unwrap().edges().count());
        assert_eq!(
            5,
            vertex
                .outgoing_edge()
                .opposite_edge()
                .face()
                .unwrap()
                .edges()
                .count()
        );
    }
}
