use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};

use geometry::Geometry;
use graph::container::{Bind, Consistent, Container, Reborrow, ReborrowMut};
use graph::storage::convert::{AsStorage, AsStorageMut};
use graph::storage::{EdgeKey, FaceKey, Storage, VertexKey};
use graph::topology::{Edge, Face, Topological, Vertex};
use graph::view::convert::{FromKeyedSource, IntoView};
use graph::view::{EdgeView, FaceView, OrphanEdgeView, OrphanFaceView};
use BoolExt;

/// Reference to a vertex.
///
/// Provides traversals, queries, and mutations related to vertices in a mesh.
/// See the module documentation for more information about topological views.
pub struct VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>> + Container,
    G: Geometry,
{
    key: VertexKey,
    storage: M,
    phantom: PhantomData<G>,
}

/// Storage.
impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>> + Container,
    G: Geometry,
{
    pub(in graph) fn bind<T, N>(self, storage: N) -> VertexView<<M as Bind<T, N>>::Output, G>
    where
        T: Topological,
        M: Bind<T, N>,
        M::Output: Reborrow,
        <M::Output as Reborrow>::Target: AsStorage<Vertex<G>> + Container,
        N: AsStorage<T> + Container,
    {
        let (key, origin) = self.into_keyed_storage();
        VertexView::from_keyed_storage_unchecked(key, origin.bind(storage))
    }
}

impl<'a, M, G> VertexView<&'a mut M, G>
where
    M: 'a + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>> + Container,
    G: 'a + Geometry,
{
    /// Converts a mutable view into an orphan view.
    pub fn into_orphan(self) -> OrphanVertexView<'a, G> {
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
    /// use plexus::graph::Mesh;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    ///
    /// # fn main() {
    /// let mut mesh = Cube::new()
    ///     .polygons_with_position()
    ///     .collect::<Mesh<Point3<f32>>>();
    /// let key = mesh.edges().nth(0).unwrap().key();
    /// let vertex = mesh.edge_mut(key).unwrap().split().unwrap().into_ref();
    ///
    /// // This would not be possible without conversion into an immutable view.
    /// let _ = vertex.into_outgoing_edge().into_face().unwrap();
    /// let _ = vertex.into_outgoing_edge().into_opposite_edge().into_face().unwrap();
    /// # }
    /// ```
    pub fn into_ref(self) -> VertexView<&'a M, G> {
        let (key, storage) = self.into_keyed_storage();
        (key, &*storage).into_view().unwrap()
    }
}

impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>> + Container,
    G: Geometry,
{
    /// Gets the key for this vertex.
    pub fn key(&self) -> VertexKey {
        self.key
    }

    fn from_keyed_storage(key: VertexKey, storage: M) -> Option<Self> {
        storage
            .reborrow()
            .as_storage()
            .contains_key(&key)
            .some(VertexView::from_keyed_storage_unchecked(key, storage))
    }

    fn from_keyed_storage_unchecked(key: VertexKey, storage: M) -> Self {
        VertexView {
            key,
            storage,
            phantom: PhantomData,
        }
    }

    fn into_keyed_storage(self) -> (VertexKey, M) {
        let VertexView { key, storage, .. } = self;
        (key, storage)
    }

    fn interior_reborrow(&self) -> VertexView<&M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow();
        VertexView::from_keyed_storage_unchecked(key, storage)
    }
}

impl<M, G> VertexView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Vertex<G>> + Container,
    G: Geometry,
{
    fn interior_reborrow_mut(&mut self) -> VertexView<&mut M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow_mut();
        VertexView::from_keyed_storage_unchecked(key, storage)
    }
}

/// Reachable API.
impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Vertex<G>> + Container,
    G: Geometry,
{
    pub(in graph) fn into_reachable_outgoing_edge(self) -> Option<EdgeView<M, G>> {
        let key = self.edge;
        key.and_then(move |key| {
            let (_, storage) = self.into_keyed_storage();
            (key, storage).into_view()
        })
    }

    pub(in graph) fn reachable_outgoing_edge(&self) -> Option<EdgeView<&M::Target, G>> {
        self.edge.and_then(|key| {
            let storage = self.storage.reborrow();
            (key, storage).into_view()
        })
    }

    pub(in graph) fn reachable_incoming_edges(&self) -> EdgeCirculator<&M::Target, G> {
        EdgeCirculator::from(self.interior_reborrow())
    }
}

impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Vertex<G>> + Container<Contract = Consistent>,
    G: Geometry,
{
    pub fn into_outgoing_edge(self) -> EdgeView<M, G> {
        self.into_reachable_outgoing_edge().unwrap()
    }

    pub fn outgoing_edge(&self) -> EdgeView<&M::Target, G> {
        self.reachable_outgoing_edge().unwrap()
    }

    pub fn incoming_edges(&self) -> EdgeCirculator<&M::Target, G> {
        self.reachable_incoming_edges()
    }
}

/// Reachable API.
impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Container,
    G: Geometry,
{
    pub(in graph) fn reachable_neighboring_faces(&self) -> FaceCirculator<&M::Target, G> {
        FaceCirculator::from(EdgeCirculator::from(self.interior_reborrow()))
    }
}

impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Container<Contract = Consistent>,
    G: Geometry,
{
    pub fn neighboring_faces(&self) -> FaceCirculator<&M::Target, G> {
        self.reachable_neighboring_faces()
    }
}

/// Reachable API.
impl<M, G> VertexView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>> + AsStorageMut<Edge<G>> + AsStorage<Vertex<G>> + Container,
    G: Geometry,
{
    pub(in graph) fn reachable_outgoing_orphan_edge(&mut self) -> Option<OrphanEdgeView<G>> {
        if let Some(key) = self.edge {
            (key, self.storage.reborrow_mut()).into_view()
        }
        else {
            None
        }
    }

    pub(in graph) fn reachable_incoming_orphan_edges(
        &mut self,
    ) -> EdgeCirculator<&mut M::Target, G> {
        EdgeCirculator::from(self.interior_reborrow_mut())
    }
}

impl<M, G> VertexView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>>
        + AsStorageMut<Edge<G>>
        + AsStorage<Vertex<G>>
        + Container<Contract = Consistent>,
    G: Geometry,
{
    pub fn outgoing_orphan_edge(&mut self) -> OrphanEdgeView<G> {
        self.reachable_outgoing_orphan_edge().unwrap()
    }

    pub fn incoming_orphan_edges(&mut self) -> EdgeCirculator<&mut M::Target, G> {
        self.reachable_incoming_orphan_edges()
    }
}

/// Reachable API.
impl<M, G> VertexView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorageMut<Face<G>>
        + AsStorage<Vertex<G>>
        + Container,
    G: Geometry,
{
    pub(in graph) fn reachable_neighboring_orphan_faces(
        &mut self,
    ) -> FaceCirculator<&mut M::Target, G> {
        FaceCirculator::from(EdgeCirculator::from(self.interior_reborrow_mut()))
    }
}

impl<M, G> VertexView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorageMut<Face<G>>
        + AsStorage<Vertex<G>>
        + Container<Contract = Consistent>,
    G: Geometry,
{
    pub fn neighboring_orphan_faces(&mut self) -> FaceCirculator<&mut M::Target, G> {
        self.reachable_neighboring_orphan_faces()
    }
}

impl<M, G> Clone for VertexView<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<Vertex<G>> + Container,
    G: Geometry,
{
    fn clone(&self) -> Self {
        VertexView {
            key: self.key,
            storage: self.storage.clone(),
            phantom: PhantomData,
        }
    }
}

impl<M, G> Copy for VertexView<M, G>
where
    M: Copy + Reborrow,
    M::Target: AsStorage<Vertex<G>> + Container,
    G: Geometry,
{}

impl<M, G> Deref for VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>> + Container,
    G: Geometry,
{
    type Target = Vertex<G>;

    fn deref(&self) -> &Self::Target {
        self.storage.reborrow().as_storage().get(&self.key).unwrap()
    }
}

impl<M, G> DerefMut for VertexView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>> + Container,
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

impl<M, G> FromKeyedSource<(VertexKey, M)> for VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>> + Container,
    G: Geometry,
{
    fn from_keyed_source(source: (VertexKey, M)) -> Option<Self> {
        let (key, storage) = source;
        VertexView::from_keyed_storage(key, storage)
    }
}

/// Orphan reference to a vertex.
///
/// Consider using `OrphanVertex` instead. See this issue:
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

impl<'a, M, G> FromKeyedSource<(VertexKey, &'a mut M)> for OrphanVertexView<'a, G>
where
    M: AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (VertexKey, &'a mut M)) -> Option<Self> {
        let (key, storage) = source;
        storage
            .as_storage_mut()
            .get_mut(&key)
            .map(|vertex| OrphanVertexView { key, vertex })
    }
}

impl<'a, G> FromKeyedSource<(VertexKey, &'a mut Vertex<G>)> for OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (VertexKey, &'a mut Vertex<G>)) -> Option<Self> {
        let (key, vertex) = source;
        Some(OrphanVertexView { key, vertex })
    }
}

pub struct EdgeCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + Container,
    G: Geometry,
{
    storage: M,
    outgoing: Option<EdgeKey>,
    breadcrumb: Option<EdgeKey>,
    phantom: PhantomData<G>,
}

impl<M, G> EdgeCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + Container,
    G: Geometry,
{
    fn next(&mut self) -> Option<EdgeKey> {
        self.outgoing
            .and_then(|outgoing| self.storage.reborrow().as_storage().get(&outgoing))
            .and_then(|outgoing| outgoing.opposite)
            .and_then(|incoming| {
                self.storage
                    .reborrow()
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

impl<M, G> From<VertexView<M, G>> for EdgeCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Vertex<G>> + Container,
    G: Geometry,
{
    fn from(vertex: VertexView<M, G>) -> Self {
        let key = vertex.edge;
        let (_, storage) = vertex.into_keyed_storage();
        EdgeCirculator {
            storage,
            outgoing: key,
            breadcrumb: key,
            phantom: PhantomData,
        }
    }
}

impl<'a, M, G> Iterator for EdgeCirculator<&'a M, G>
where
    M: 'a + AsStorage<Edge<G>> + Container,
    G: 'a + Geometry,
{
    type Item = EdgeView<&'a M, G>;

    fn next(&mut self) -> Option<Self::Item> {
        EdgeCirculator::next(self).and_then(|key| (key, self.storage).into_view())
    }
}

impl<'a, M, G> Iterator for EdgeCirculator<&'a mut M, G>
where
    M: 'a + AsStorage<Edge<G>> + AsStorageMut<Edge<G>> + Container,
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
            }).into_view()
        })
    }
}

pub struct FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + Container,
    G: Geometry,
{
    input: EdgeCirculator<M, G>,
}

impl<M, G> FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + Container,
    G: Geometry,
{
    fn next(&mut self) -> Option<FaceKey> {
        while let Some(edge) = self.input.next() {
            if let Some(face) = AsStorage::<Edge<G>>::as_storage(&self.input.storage.reborrow())
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

impl<M, G> From<EdgeCirculator<M, G>> for FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + Container,
    G: Geometry,
{
    fn from(input: EdgeCirculator<M, G>) -> Self {
        FaceCirculator { input }
    }
}

impl<'a, M, G> Iterator for FaceCirculator<&'a M, G>
where
    M: 'a + AsStorage<Edge<G>> + AsStorage<Face<G>> + Container,
    G: 'a + Geometry,
{
    type Item = FaceView<&'a M, G>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).and_then(|key| (key, self.input.storage).into_view())
    }
}

impl<'a, M, G> Iterator for FaceCirculator<&'a mut M, G>
where
    M: 'a + AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>> + Container,
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
            }).into_view()
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use graph::*;
    use primitive::*;

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
