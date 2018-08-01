use failure::Error;
use std::collections::HashSet;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Add, Deref, DerefMut, Mul};

use geometry::convert::AsPosition;
use geometry::Geometry;
use graph::geometry::alias::{ScaledFaceNormal, VertexPosition};
use graph::geometry::{FaceCentroid, FaceNormal};
use graph::mesh::Mesh;
use graph::mutation::face::{self, FaceExtrudeCache, FaceJoinCache, FaceTriangulateCache};
use graph::mutation::{Commit, Mutation};
use graph::storage::convert::{AsStorage, AsStorageMut};
use graph::storage::{Bind, EdgeKey, FaceKey, Storage, VertexKey};
use graph::topology::{Edge, Face, Topological, Vertex};
use graph::view::convert::{FromKeyedSource, IntoView};
use graph::view::{
    Consistent, Container, EdgeKeyTopology, EdgeView, OrphanEdgeView, OrphanVertexView, Reborrow,
    ReborrowMut, VertexView,
};
use BoolExt;

/// Do **not** use this type directly. Use `FaceRef` and `FaceMut` instead.
///
/// This type is only re-exported so that its members are shown in
/// documentation. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
pub struct FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>> + Container,
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
    M::Target: AsStorage<Face<G>> + Container,
    G: Geometry,
{
    pub(in graph) fn bind<T, N>(self, storage: N) -> FaceView<<M as Bind<T, N>>::Output, G>
    where
        T: Topological,
        M: Bind<T, N>,
        M::Output: Reborrow,
        <M::Output as Reborrow>::Target: AsStorage<Face<G>> + Container,
        N: AsStorage<T>,
    {
        let (key, origin) = self.into_keyed_storage();
        FaceView::from_keyed_storage_unchecked(key, origin.bind(storage))
    }
}

impl<'a, M, G> FaceView<&'a mut M, G>
where
    M: 'a + AsStorage<Face<G>> + AsStorageMut<Face<G>> + Container,
    G: 'a + Geometry,
{
    pub fn into_orphan(self) -> OrphanFaceView<'a, G> {
        let (key, storage) = self.into_keyed_storage();
        (key, storage).into_view().unwrap()
    }

    pub fn into_ref(self) -> FaceView<&'a M, G> {
        let (key, storage) = self.into_keyed_storage();
        (key, &*storage).into_view().unwrap()
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>> + Container,
    G: Geometry,
{
    pub fn key(&self) -> FaceKey {
        self.key
    }

    fn from_keyed_storage(key: FaceKey, storage: M) -> Option<Self> {
        storage
            .reborrow()
            .as_storage()
            .contains_key(&key)
            .into_some(FaceView::from_keyed_storage_unchecked(key, storage))
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
    M::Target: AsStorage<Face<G>> + Container,
    G: Geometry,
{
    fn interior_reborrow_mut(&mut self) -> FaceView<&mut M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow_mut();
        FaceView::from_keyed_storage_unchecked(key, storage)
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + Container,
    G: Geometry,
{
    pub fn to_key_topology(&self) -> FaceKeyTopology {
        FaceKeyTopology::new(
            self.key,
            self.reachable_interior_edges()
                .map(|edge| edge.to_key_topology()),
        )
    }

    pub fn arity(&self) -> usize {
        self.reachable_interior_edges().count()
    }

    pub(in graph) fn reachable_interior_edges(&self) -> EdgeCirculator<&M::Target, G> {
        EdgeCirculator::from(self.interior_reborrow())
    }

    pub(in graph) fn reachable_neighboring_faces(&self) -> FaceCirculator<&M::Target, G> {
        FaceCirculator::from(EdgeCirculator::from(self.interior_reborrow()))
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Container,
    G: Geometry,
{
    pub(in graph) fn mutuals(&self) -> HashSet<VertexKey> {
        self.reachable_neighboring_faces()
            .map(|face| {
                face.reachable_vertices()
                    .map(|vertex| vertex.key())
                    .collect::<HashSet<_>>()
            })
            .fold(
                self.reachable_vertices()
                    .map(|vertex| vertex.key())
                    .collect::<HashSet<_>>(),
                |intersection, vertices| intersection.intersection(&vertices).cloned().collect(),
            )
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + Container<Consistency = Consistent>,
    G: Geometry,
{
    pub fn interior_edges(&self) -> EdgeCirculator<&M::Target, G> {
        self.reachable_interior_edges()
    }

    pub fn neighboring_faces(&self) -> FaceCirculator<&M::Target, G> {
        self.reachable_neighboring_faces()
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Container,
    G: Geometry,
{
    pub(in graph) fn reachable_vertices(&self) -> VertexCirculator<&M::Target, G> {
        VertexCirculator::from(EdgeCirculator::from(self.interior_reborrow()))
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Container<Consistency = Consistent>,
    G: Geometry,
{
    pub fn vertices(&self) -> VertexCirculator<&M::Target, G> {
        self.reachable_vertices()
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>> + AsStorageMut<Edge<G>> + AsStorage<Face<G>> + Container,
    G: Geometry,
{
    pub(in graph) fn reachable_interior_orphan_edges(
        &mut self,
    ) -> EdgeCirculator<&mut M::Target, G> {
        EdgeCirculator::from(self.interior_reborrow_mut())
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>> + Container,
    G: Geometry,
{
    pub(in graph) fn reachable_neighboring_orphan_faces(
        &mut self,
    ) -> FaceCirculator<&mut M::Target, G> {
        FaceCirculator::from(EdgeCirculator::from(self.interior_reborrow_mut()))
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>>
        + AsStorageMut<Edge<G>>
        + AsStorage<Face<G>>
        + Container<Consistency = Consistent>,
    G: Geometry,
{
    pub fn interior_orphan_edges(&mut self) -> EdgeCirculator<&mut M::Target, G> {
        self.reachable_interior_orphan_edges()
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorageMut<Face<G>>
        + Container<Consistency = Consistent>,
    G: Geometry,
{
    pub fn neighboring_orphan_faces(&mut self) -> FaceCirculator<&mut M::Target, G> {
        self.reachable_neighboring_orphan_faces()
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + AsStorageMut<Vertex<G>>
        + Container,
    G: Geometry,
{
    pub(in graph) fn reachable_orphan_vertices(&mut self) -> VertexCirculator<&mut M::Target, G> {
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
        + Container<Consistency = Consistent>,
    G: Geometry,
{
    pub fn orphan_vertices(&mut self) -> VertexCirculator<&mut M::Target, G> {
        self.reachable_orphan_vertices()
    }
}

impl<'a, G> FaceView<&'a mut Mesh<G>, G>
where
    G: Geometry,
{
    pub fn join(self, destination: FaceKey) -> Result<(), Error> {
        let (source, storage) = self.into_keyed_storage();
        let cache = FaceJoinCache::snapshot(storage, source, destination)?;
        Mutation::replace(storage, Mesh::empty())
            .commit_with(move |mutation| face::join_with_cache(&mut *mutation, cache))
            .unwrap();
        Ok(())
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Container<Consistency = Consistent>,
    G: FaceCentroid + Geometry,
{
    pub fn centroid(&self) -> Result<G::Centroid, Error> {
        G::centroid(self)
    }
}

impl<'a, G> FaceView<&'a mut Mesh<G>, G>
where
    G: FaceCentroid<Centroid = <G as Geometry>::Vertex> + Geometry,
{
    pub fn triangulate(self) -> Result<Option<VertexView<&'a mut Mesh<G>, G>>, Error> {
        let (abc, storage) = self.into_keyed_storage();
        let cache = FaceTriangulateCache::snapshot(storage, abc)?;
        let (storage, vertex) = Mutation::replace(storage, Mesh::empty())
            .commit_with(move |mutation| face::triangulate_with_cache(&mut *mutation, cache))
            .unwrap();
        Ok(vertex.map(|vertex| (vertex, storage).into_view().unwrap()))
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Container<Consistency = Consistent>,
    G: FaceNormal + Geometry,
{
    pub fn normal(&self) -> Result<G::Normal, Error> {
        G::normal(self)
    }
}

impl<'a, G> FaceView<&'a mut Mesh<G>, G>
where
    G: FaceNormal + Geometry,
    G::Vertex: AsPosition,
{
    pub fn extrude<T>(self, distance: T) -> Result<FaceView<&'a mut Mesh<G>, G>, Error>
    where
        G::Normal: Mul<T>,
        ScaledFaceNormal<G, T>: Clone,
        VertexPosition<G>: Add<ScaledFaceNormal<G, T>, Output = VertexPosition<G>> + Clone,
    {
        let (abc, storage) = self.into_keyed_storage();
        let cache = FaceExtrudeCache::snapshot(storage, abc, distance)?;
        let (storage, face) = Mutation::replace(storage, Mesh::empty())
            .commit_with(move |mutation| face::extrude_with_cache(&mut *mutation, cache))
            .unwrap();
        Ok((face, storage).into_view().unwrap())
    }
}

impl<M, G> Clone for FaceView<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<Face<G>> + Container,
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
    M::Target: AsStorage<Face<G>> + Container,
    G: Geometry,
{}

impl<M, G> Deref for FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>> + Container,
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
    M::Target: AsStorage<Face<G>> + AsStorageMut<Face<G>> + Container,
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
    M::Target: AsStorage<Face<G>> + Container,
    G: Geometry,
{
    fn from_keyed_source(source: (FaceKey, M)) -> Option<Self> {
        let (key, storage) = source;
        FaceView::from_keyed_storage(key, storage)
    }
}

/// Do **not** use this type directly. Use `OrphanFace` instead.
///
/// This type is only re-exported so that its members are shown in
/// documentation. See this issue:
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
    fn new<I>(face: FaceKey, edges: I) -> Self
    where
        I: IntoIterator<Item = EdgeKeyTopology>,
    {
        FaceKeyTopology {
            key: face,
            edges: edges.into_iter().collect(),
        }
    }

    pub fn key(&self) -> FaceKey {
        self.key
    }

    pub fn interior_edges(&self) -> &[EdgeKeyTopology] {
        self.edges.as_slice()
    }
}

pub struct VertexCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + Container,
    G: Geometry,
{
    input: EdgeCirculator<M, G>,
}

impl<M, G> VertexCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + Container,
    G: Geometry,
{
    fn next(&mut self) -> Option<VertexKey> {
        let edge = self.input.next();
        edge.and_then(|edge| self.input.storage.reborrow().as_storage().get(&edge))
            .map(|edge| edge.vertex)
    }
}

impl<M, G> From<EdgeCirculator<M, G>> for VertexCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + Container,
    G: Geometry,
{
    fn from(input: EdgeCirculator<M, G>) -> Self {
        VertexCirculator { input }
    }
}

impl<'a, M, G> Iterator for VertexCirculator<&'a M, G>
where
    M: 'a + AsStorage<Edge<G>> + AsStorage<Vertex<G>> + Container,
    G: 'a + Geometry,
{
    type Item = VertexView<&'a M, G>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).and_then(|key| (key, self.input.storage).into_view())
    }
}

impl<'a, M, G> Iterator for VertexCirculator<&'a mut M, G>
where
    M: 'a + AsStorage<Edge<G>> + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>> + Container,
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
            }).into_view()
        })
    }
}

pub struct EdgeCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + Container,
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
    M::Target: AsStorage<Edge<G>> + Container,
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
    M::Target: AsStorage<Edge<G>> + AsStorage<Face<G>> + Container,
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
    M::Target: AsStorage<Edge<G>> + Container,
    G: Geometry,
{
    input: EdgeCirculator<M, G>,
}

impl<M, G> FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Edge<G>> + Container,
    G: Geometry,
{
    fn next(&mut self) -> Option<FaceKey> {
        while let Some(edge) = self
            .input
            .next()
            .and_then(|edge| self.input.storage.reborrow().as_storage().get(&edge))
        {
            if let Some(face) = edge
                .opposite
                .and_then(|opposite| self.input.storage.reborrow().as_storage().get(&opposite))
                .and_then(|opposite| opposite.face)
            {
                return Some(face);
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
    M::Target: AsStorage<Edge<G>> + Container,
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

    use generate::*;
    use graph::*;

    #[test]
    fn circulate_over_edges() {
        let mesh = sphere::UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();
        let face = mesh.faces().nth(0).unwrap();

        // All faces should be triangles and should have three edges.
        assert_eq!(3, face.interior_edges().count());
    }

    #[test]
    fn circulate_over_faces() {
        let mesh = sphere::UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();
        let face = mesh.faces().nth(0).unwrap();

        // No matter which face is selected, it should have three neighbors.
        assert_eq!(3, face.neighboring_faces().count());
    }

    #[test]
    fn extrude_face() {
        let mut mesh = sphere::UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();
        {
            let key = mesh.faces().nth(0).unwrap().key();
            let face = mesh.face_mut(key).unwrap().extrude(1.0).unwrap();

            // The extruded face, being a triangle, should have three
            // neighboring faces.
            assert_eq!(3, face.neighboring_faces().count());
        }

        assert_eq!(8, mesh.vertex_count());
        // The mesh begins with 18 edges. The extrusion adds three quads with
        // four interior edges each, so there are `18 + (3 * 4)` edges.
        assert_eq!(30, mesh.edge_count());
        // All faces are triangles and the mesh begins with six such faces. The
        // extruded face remains, in addition to three connective faces, each
        // of which is constructed from quads.
        assert_eq!(9, mesh.face_count());
    }

    #[test]
    fn triangulate_mesh() {
        let (indeces, vertices) = cube::Cube::new()
            .polygons_with_position() // 6 quads, 24 vertices.
            .flat_index_vertices(HashIndexer::default());
        let mut mesh = Mesh::<Point3<f32>>::from_raw_buffers(indeces, vertices, 4).unwrap();
        mesh.triangulate().unwrap();

        // There are 8 unique vertices and a vertex is added for each quad,
        // yielding `8 + 6` vertices.
        assert_eq!(14, mesh.vertex_count());
        assert_eq!(72, mesh.edge_count());
        // Each quad becomes a tetrahedron, so 6 quads become 24 triangles.
        assert_eq!(24, mesh.face_count());
    }
}
