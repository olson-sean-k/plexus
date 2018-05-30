use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};

use geometry::Geometry;
use graph::mesh::{Edge, Face, Mesh, Vertex};
use graph::storage::convert::{AsStorage, AsStorageMut};
use graph::storage::{Bind, EdgeKey, FaceKey, Storage, Topological, VertexKey};
use graph::view::convert::{FromKeyedSource, IntoView};
use graph::view::{
    Consistency, Consistent, EdgeView, FaceView, Inconsistent, IteratorExt, OrphanEdgeView,
    OrphanFaceView,
};

/// Do **not** use this type directly. Use `VertexRef` and `VertexMut` instead.
///
/// This type is only re-exported so that its members are shown in
/// documentation. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
pub struct VertexView<M, G, C = Inconsistent>
where
    M: AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    key: VertexKey,
    storage: M,
    phantom: PhantomData<(G, C)>,
}

/// Storage.
impl<M, G, C> VertexView<M, G, C>
where
    M: AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn bind<T, N>(self, storage: N) -> VertexView<<M as Bind<T, N>>::Output, G, C>
    where
        T: Topological,
        M: Bind<T, N>,
        M::Output: AsStorage<Vertex<G>>,
        N: AsStorage<T>,
    {
        let VertexView {
            key,
            storage: origin,
            ..
        } = self;
        VertexView {
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

impl<'a, 'b, M, G, C> VertexView<&'a &'b M, G, C>
where
    M: AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn into_interior_deref(self) -> VertexView<&'b M, G, C>
    where
        (VertexKey, &'b M): IntoView<VertexView<&'b M, G, C>>,
    {
        let key = self.key;
        let storage = *self.storage;
        (key, storage).into_view()
    }

    pub fn interior_deref(&self) -> VertexView<&'b M, G, C>
    where
        (VertexKey, &'b M): IntoView<VertexView<&'b M, G, C>>,
    {
        let key = self.key;
        let storage = *self.storage;
        (key, storage).into_view()
    }
}

impl<'a, M, G, C> VertexView<&'a mut M, G, C>
where
    M: 'a + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: 'a + Geometry,
    C: Consistency,
{
    pub fn into_orphan(self) -> OrphanVertexView<'a, G> {
        let VertexView { key, storage, .. } = self;
        (key, storage.as_storage_mut().get_mut(&key).unwrap()).into_view()
    }
}

impl<M, G, C> VertexView<M, G, C>
where
    M: AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn to_orphan<'a>(&'a mut self) -> OrphanVertexView<'a, G> {
        let key = self.key;
        (key, self.storage.as_storage_mut().get_mut(&key).unwrap()).into_view()
    }
}

impl<M, G, C> VertexView<M, G, C>
where
    M: AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn key(&self) -> VertexKey {
        self.key
    }

    pub(in graph) fn from_keyed_storage(key: VertexKey, storage: M) -> Self {
        VertexView {
            key,
            storage,
            phantom: PhantomData,
        }
    }

    fn to_ref(&self) -> VertexView<&M, G, C> {
        let key = self.key;
        let storage = &self.storage;
        VertexView::from_keyed_storage(key, storage)
    }

    fn to_mut(&mut self) -> VertexView<&mut M, G, C> {
        let key = self.key;
        let storage = &mut self.storage;
        VertexView::from_keyed_storage(key, storage)
    }
}

/// Reachable API.
impl<M, G, C> VertexView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn into_reachable_outgoing_edge(self) -> Option<EdgeView<M, G, C>>
    where
        (EdgeKey, M): IntoView<EdgeView<M, G, C>>,
    {
        let key = self.edge;
        key.map(move |key| {
            let VertexView { storage, .. } = self;
            (key, storage).into_view()
        })
    }
}

/// Reachable API.
impl<M, G, C> VertexView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_outgoing_edge<'a>(&'a self) -> Option<EdgeView<&'a M, G, C>>
    where
        (EdgeKey, &'a M): IntoView<EdgeView<&'a M, G, C>>,
    {
        self.edge.map(|key| {
            let storage = &self.storage;
            (key, storage).into_view()
        })
    }

    pub(in graph) fn reachable_incoming_edges<'a>(
        &'a self,
    ) -> impl Iterator<Item = EdgeView<&'a M, G, C>>
    where
        (EdgeKey, &'a M): IntoView<EdgeView<&'a M, G, C>>,
    {
        let key = self.edge;
        let storage = &self.storage;
        EdgeCirculator::from_keyed_source((key, storage))
            .map_with_ref(|circulator, key| (key, circulator.storage).into_view())
    }
}

impl<M, G> VertexView<M, G, Consistent>
where
    M: AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub fn into_outgoing_edge(self) -> EdgeView<M, G, Consistent>
    where
        (EdgeKey, M): IntoView<EdgeView<M, G, Consistent>>,
    {
        self.into_reachable_outgoing_edge().unwrap()
    }

    pub fn outgoing_edge<'a>(&'a self) -> EdgeView<&'a M, G, Consistent>
    where
        (EdgeKey, &'a M): IntoView<EdgeView<&'a M, G, Consistent>>,
    {
        self.reachable_outgoing_edge().unwrap()
    }
}

impl<M, G> VertexView<M, G, Consistent>
where
    M: AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub fn incoming_edges<'a>(&'a self) -> impl Iterator<Item = EdgeView<&'a M, G, Consistent>>
    where
        (EdgeKey, &'a M): IntoView<EdgeView<&'a M, G, Consistent>>,
    {
        self.reachable_incoming_edges()
    }
}

/// Reachable API.
impl<M, G, C> VertexView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_neighboring_faces(
        &self,
    ) -> impl Iterator<Item = FaceView<&M, G, C>> {
        FaceCirculator::from(EdgeCirculator::from(self.to_ref())).map_with_ref(|circulator, key| {
            FaceView::from_keyed_storage(key, circulator.input.storage)
        })
    }
}

impl<M, G> VertexView<M, G, Consistent>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub fn neighboring_faces(&self) -> impl Iterator<Item = FaceView<&M, G, Consistent>> {
        self.reachable_neighboring_faces()
    }
}

/// Reachable API.
impl<M, G, C> VertexView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorageMut<Edge<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_outgoing_orphan_edge<'a>(
        &'a mut self,
    ) -> Option<OrphanEdgeView<'a, G>> {
        if let Some(key) = self.edge {
            Some((key, self.storage.as_storage_mut().get_mut(&key).unwrap()).into_view())
        }
        else {
            None
        }
    }

    pub(in graph) fn reachable_incoming_orphan_edges<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = OrphanEdgeView<'a, G>> {
        let key = self.edge;
        let storage = &mut self.storage;
        EdgeCirculator::from_keyed_source((key, storage)).map_with_mut(|circulator, key| {
            (key, unsafe {
                // Apply `'a` to the autoref from `as_storage_mut` and
                // `get_mut`.
                mem::transmute::<&'_ mut Edge<G>, &'a mut Edge<G>>(
                    circulator.storage.as_storage_mut().get_mut(&key).unwrap(),
                )
            }).into_view()
        })
    }
}

impl<M, G> VertexView<M, G, Consistent>
where
    M: AsStorage<Edge<G>> + AsStorageMut<Edge<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub fn outgoing_orphan_edge<'a>(&'a mut self) -> OrphanEdgeView<'a, G> {
        self.reachable_outgoing_orphan_edge().unwrap()
    }

    pub fn incoming_orphan_edges<'a>(&'a mut self) -> impl Iterator<Item = OrphanEdgeView<'a, G>> {
        self.reachable_incoming_orphan_edges()
    }
}

/// Reachable API.
impl<M, G, C> VertexView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_neighboring_orphan_faces<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = OrphanFaceView<'a, G>> {
        let key = self.edge;
        let storage = &mut self.storage;
        FaceCirculator::from(EdgeCirculator::from_keyed_source((key, storage))).map_with_mut(
            |circulator, key| {
                (key, unsafe {
                    // Apply `'a` to the autoref from `as_storage_mut` and
                    // `get_mut`.
                    mem::transmute::<&'_ mut Face<G>, &'a mut Face<G>>(
                        circulator
                            .input
                            .storage
                            .as_storage_mut()
                            .get_mut(&key)
                            .unwrap(),
                    )
                }).into_view()
            },
        )
    }
}

impl<M, G> VertexView<M, G, Consistent>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    // TODO: This will require a constraint for the conversion.
    pub fn neighboring_orphan_faces<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = OrphanFaceView<'a, G>> {
        self.reachable_neighboring_orphan_faces()
    }
}

impl<M, G, C> Clone for VertexView<M, G, C>
where
    M: AsStorage<Vertex<G>> + Clone,
    G: Geometry,
    C: Consistency,
{
    fn clone(&self) -> Self {
        VertexView {
            key: self.key,
            storage: self.storage.clone(),
            phantom: PhantomData,
        }
    }
}

impl<M, G, C> Copy for VertexView<M, G, C>
where
    M: AsStorage<Vertex<G>> + Copy,
    G: Geometry,
    C: Consistency,
{
}

impl<M, G, C> Deref for VertexView<M, G, C>
where
    M: AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    type Target = Vertex<G>;

    fn deref(&self) -> &Self::Target {
        self.storage.as_storage().get(&self.key).unwrap()
    }
}

impl<M, G, C> DerefMut for VertexView<M, G, C>
where
    M: AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.storage.as_storage_mut().get_mut(&self.key).unwrap()
    }
}

impl<M, G> FromKeyedSource<(VertexKey, M)> for VertexView<M, G, Inconsistent>
where
    M: AsStorage<Vertex<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (VertexKey, M)) -> Self {
        let (key, storage) = source;
        VertexView {
            key,
            storage,
            phantom: PhantomData,
        }
    }
}

impl<M, G> FromKeyedSource<(VertexKey, M)> for VertexView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (VertexKey, M)) -> Self {
        let (key, storage) = source;
        VertexView {
            key,
            storage,
            phantom: PhantomData,
        }
    }
}

/// Do **not** use this type directly. Use `OrphanVertex` instead.
///
/// This type is only re-exported so that its members are shown in
/// documentation. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
pub struct OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    key: VertexKey,
    vertex: &'a mut Vertex<G>,
}

impl<'a, G> OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    pub fn key(&self) -> VertexKey {
        self.key
    }
}

impl<'a, G> Deref for OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    type Target = Vertex<G>;

    fn deref(&self) -> &Self::Target {
        &*self.vertex
    }
}

impl<'a, G> DerefMut for OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.vertex
    }
}

impl<'a, G> FromKeyedSource<(VertexKey, &'a mut Vertex<G>)> for OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (VertexKey, &'a mut Vertex<G>)) -> Self {
        let (key, vertex) = source;
        OrphanVertexView { key, vertex }
    }
}

pub struct EdgeCirculator<M, G>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
{
    storage: M,
    outgoing: Option<EdgeKey>,
    breadcrumb: Option<EdgeKey>,
    phantom: PhantomData<G>,
}

impl<M, G, C> From<VertexView<M, G, C>> for EdgeCirculator<M, G>
where
    M: AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    fn from(vertex: VertexView<M, G, C>) -> Self {
        let key = vertex.edge;
        let VertexView { storage, .. } = vertex;
        EdgeCirculator {
            storage,
            outgoing: key,
            breadcrumb: key,
            phantom: PhantomData,
        }
    }
}

impl<M, G> FromKeyedSource<(Option<EdgeKey>, M)> for EdgeCirculator<M, G>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (Option<EdgeKey>, M)) -> Self {
        let (key, storage) = source;
        EdgeCirculator {
            storage,
            outgoing: key,
            breadcrumb: key,
            phantom: PhantomData,
        }
    }
}

impl<M, G> Iterator for EdgeCirculator<M, G>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
{
    type Item = EdgeKey;

    fn next(&mut self) -> Option<Self::Item> {
        self.outgoing
            .and_then(|outgoing| self.storage.as_storage().get(&outgoing))
            .and_then(|outgoing| outgoing.opposite)
            .and_then(|incoming| {
                self.storage
                    .as_storage()
                    .get(&incoming)
                    .map(|incoming| incoming.next)
                    .map(|outgoing| (incoming, outgoing))
            })
            .and_then(|(incoming, outgoing)| {
                self.breadcrumb.map(|_| {
                    if self.breadcrumb == outgoing {
                        self.breadcrumb = None;
                    }
                    else {
                        self.outgoing = outgoing;
                    }
                    incoming
                })
            })
    }
}

pub struct FaceCirculator<M, G>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    input: EdgeCirculator<M, G>,
}

impl<M, G> From<EdgeCirculator<M, G>> for FaceCirculator<M, G>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    fn from(input: EdgeCirculator<M, G>) -> Self {
        FaceCirculator { input }
    }
}

impl<M, G> Iterator for FaceCirculator<M, G>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    type Item = FaceKey;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(edge) = self.input.next() {
            if let Some(face) = AsStorage::<Edge<G>>::as_storage(&self.input.storage)
                .get(&edge)
                .and_then(|edge| edge.face)
            {
                return Some(face);
            }
            else {
                // Skip edges with no face. This can occur within non-enclosed
                // meshes.
                continue;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use generate::*;
    use graph::*;

    #[test]
    fn circulate_over_edges() {
        let mesh = sphere::UvSphere::new(4, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();

        // All faces should be triangles and all vertices should have 4
        // (incoming) edges.
        for vertex in mesh.vertices() {
            assert_eq!(4, vertex.incoming_edges().count());
        }
    }
}
