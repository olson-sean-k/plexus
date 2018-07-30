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
use graph::storage::{Bind, EdgeKey, FaceKey, VertexKey};
use graph::topology::{Edge, Face, Topological, Vertex};
use graph::view::convert::{FromKeyedSource, IntoView};
use graph::view::{
    Consistency, Consistent, EdgeKeyTopology, EdgeView, Inconsistent, OrphanEdgeView,
    OrphanVertexView, VertexView,
};
use BoolExt;

/// Do **not** use this type directly. Use `FaceRef` and `FaceMut` instead.
///
/// This type is only re-exported so that its members are shown in
/// documentation. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
pub struct FaceView<M, G, C>
where
    M: AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    key: FaceKey,
    storage: M,
    phantom: PhantomData<(G, C)>,
}

/// Storage.
impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn bind<T, N>(self, storage: N) -> FaceView<<M as Bind<T, N>>::Output, G, C>
    where
        T: Topological,
        M: Bind<T, N>,
        M::Output: AsStorage<Face<G>>,
        N: AsStorage<T>,
    {
        let (key, origin) = self.into_keyed_storage();
        FaceView::from_keyed_storage_unchecked(key, origin.bind(storage))
    }
}

impl<'a, M, G, C> FaceView<&'a mut M, G, C>
where
    M: 'a + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: 'a + Geometry,
    C: Consistency,
{
    pub fn into_orphan(self) -> OrphanFaceView<'a, G> {
        let (key, storage) = self.into_keyed_storage();
        (key, storage.as_storage_mut().get_mut(&key).unwrap())
            .into_view()
            .unwrap()
    }

    pub fn into_ref(self) -> FaceView<&'a M, G, Consistent> {
        let (key, storage) = self.into_keyed_storage();
        FaceView::from_keyed_storage_unchecked(key, &*storage)
    }
}

impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn key(&self) -> FaceKey {
        self.key
    }

    pub(in graph) fn from_keyed_storage(key: FaceKey, storage: M) -> Option<Self> {
        storage
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

    pub(in graph) fn into_keyed_storage(self) -> (FaceKey, M) {
        let FaceView { key, storage, .. } = self;
        (key, storage)
    }

    fn interior_ref(&self) -> FaceView<&M, G, C> {
        let key = self.key;
        let storage = &self.storage;
        FaceView::from_keyed_storage_unchecked(key, storage)
    }

    fn interior_mut(&mut self) -> FaceView<&mut M, G, C> {
        let key = self.key;
        let storage = &mut self.storage;
        FaceView::from_keyed_storage_unchecked(key, storage)
    }
}

impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
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

    pub(in graph) fn reachable_interior_edges(&self) -> EdgeCirculator<&M, G, C> {
        EdgeCirculator::from(self.interior_ref())
    }

    pub(in graph) fn reachable_neighboring_faces(&self) -> FaceCirculator<&M, G, C> {
        FaceCirculator::from(EdgeCirculator::from(self.interior_ref()))
    }
}

impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
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

impl<M, G> FaceView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    pub fn interior_edges(&self) -> EdgeCirculator<&Mesh<G>, G, Consistent> {
        let key = self.edge;
        let storage = self.storage.as_ref();
        EdgeCirculator::from_keyed_storage(key, storage)
    }

    pub fn neighboring_faces(&self) -> FaceCirculator<&Mesh<G>, G, Consistent> {
        let key = self.edge;
        let storage = self.storage.as_ref();
        FaceCirculator::from(EdgeCirculator::from_keyed_storage(key, storage))
    }
}

impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_vertices(&self) -> VertexCirculator<&M, G, C> {
        VertexCirculator::from(EdgeCirculator::from(self.interior_ref()))
    }
}

impl<M, G> FaceView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub fn vertices(&self) -> VertexCirculator<&Mesh<G>, G, Consistent> {
        let key = self.edge;
        let storage = self.storage.as_ref();
        VertexCirculator::from(EdgeCirculator::from_keyed_storage(key, storage))
    }
}

impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorageMut<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_interior_orphan_edges(&mut self) -> EdgeCirculator<&mut M, G, C> {
        EdgeCirculator::from(self.interior_mut())
    }
}

impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_neighboring_orphan_faces(&mut self) -> FaceCirculator<&mut M, G, C> {
        FaceCirculator::from(EdgeCirculator::from(self.interior_mut()))
    }
}

impl<M, G> FaceView<M, G, Consistent>
where
    M: AsRef<Mesh<G>>
        + AsMut<Mesh<G>>
        + AsStorage<Edge<G>>
        + AsStorageMut<Edge<G>>
        + AsStorage<Face<G>>,
    G: Geometry,
{
    pub fn interior_orphan_edges(&mut self) -> EdgeCirculator<&mut Mesh<G>, G, Consistent> {
        let key = self.edge;
        let storage = self.storage.as_mut();
        EdgeCirculator::from_keyed_storage(key, storage)
    }
}

impl<M, G> FaceView<M, G, Consistent>
where
    M: AsRef<Mesh<G>>
        + AsMut<Mesh<G>>
        + AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorageMut<Face<G>>,
    G: Geometry,
{
    pub fn neighboring_orphan_faces(&mut self) -> FaceCirculator<&mut Mesh<G>, G, Consistent> {
        let key = self.edge;
        let storage = self.storage.as_mut();
        FaceCirculator::from(EdgeCirculator::from_keyed_storage(key, storage))
    }
}

impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_orphan_vertices(&mut self) -> VertexCirculator<&mut M, G, C> {
        VertexCirculator::from(EdgeCirculator::from(self.interior_mut()))
    }
}

impl<M, G> FaceView<M, G, Consistent>
where
    M: AsRef<Mesh<G>>
        + AsMut<Mesh<G>>
        + AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + AsStorageMut<Vertex<G>>,
    G: Geometry,
{
    pub fn orphan_vertices(&mut self) -> VertexCirculator<&mut Mesh<G>, G, Consistent> {
        let key = self.edge;
        let storage = self.storage.as_mut();
        VertexCirculator::from(EdgeCirculator::from_keyed_storage(key, storage))
    }
}

impl<'a, G> FaceView<&'a mut Mesh<G>, G, Consistent>
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

impl<M, G> FaceView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: FaceCentroid + Geometry,
{
    pub fn centroid(&self) -> Result<G::Centroid, Error> {
        G::centroid(self)
    }
}

impl<'a, G> FaceView<&'a mut Mesh<G>, G, Consistent>
where
    G: FaceCentroid<Centroid = <G as Geometry>::Vertex> + Geometry,
{
    pub fn triangulate(self) -> Result<Option<VertexView<&'a mut Mesh<G>, G, Consistent>>, Error> {
        let (abc, storage) = self.into_keyed_storage();
        let cache = FaceTriangulateCache::snapshot(storage, abc)?;
        let (storage, vertex) = Mutation::replace(storage, Mesh::empty())
            .commit_with(move |mutation| face::triangulate_with_cache(&mut *mutation, cache))
            .unwrap();
        Ok(vertex.map(|vertex| {
            VertexView::<_, _, Consistent>::from_keyed_storage(vertex, storage).unwrap()
        }))
    }
}

impl<M, G> FaceView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: FaceNormal + Geometry,
{
    pub fn normal(&self) -> Result<G::Normal, Error> {
        G::normal(self)
    }
}

impl<'a, G> FaceView<&'a mut Mesh<G>, G, Consistent>
where
    G: FaceNormal + Geometry,
    G::Vertex: AsPosition,
{
    pub fn extrude<T>(self, distance: T) -> Result<FaceView<&'a mut Mesh<G>, G, Consistent>, Error>
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
        Ok(FaceView::<_, _, Consistent>::from_keyed_storage(face, storage).unwrap())
    }
}

impl<M, G, C> Clone for FaceView<M, G, C>
where
    M: AsStorage<Face<G>> + Clone,
    G: Geometry,
    C: Consistency,
{
    fn clone(&self) -> Self {
        FaceView {
            storage: self.storage.clone(),
            key: self.key,
            phantom: PhantomData,
        }
    }
}

impl<M, G, C> Copy for FaceView<M, G, C>
where
    M: AsStorage<Face<G>> + Copy,
    G: Geometry,
    C: Consistency,
{}

impl<M, G, C> Deref for FaceView<M, G, C>
where
    M: AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    type Target = Face<G>;

    fn deref(&self) -> &Self::Target {
        self.storage.as_storage().get(&self.key).unwrap()
    }
}

impl<M, G, C> DerefMut for FaceView<M, G, C>
where
    M: AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.storage.as_storage_mut().get_mut(&self.key).unwrap()
    }
}

impl<M, G> FromKeyedSource<(FaceKey, M)> for FaceView<M, G, Inconsistent>
where
    M: AsStorage<Face<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (FaceKey, M)) -> Option<Self> {
        let (key, storage) = source;
        FaceView::<_, _, Inconsistent>::from_keyed_storage(key, storage)
    }
}

impl<M, G> FromKeyedSource<(FaceKey, M)> for FaceView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (FaceKey, M)) -> Option<Self> {
        let (key, storage) = source;
        FaceView::<_, _, Consistent>::from_keyed_storage(key, storage)
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
    fn from_keyed_storage(key: FaceKey, face: &'a mut Face<G>) -> Self {
        OrphanFaceView { key, face }
    }

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

impl<'a, G> FromKeyedSource<(FaceKey, &'a mut Face<G>)> for OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (FaceKey, &'a mut Face<G>)) -> Option<Self> {
        let (key, face) = source;
        Some(OrphanFaceView::from_keyed_storage(key, face))
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

pub struct VertexCirculator<M, G, C>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    input: EdgeCirculator<M, G, C>,
}

impl<M, G, C> VertexCirculator<M, G, C>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    fn next(&mut self) -> Option<VertexKey> {
        let edge = self.input.next();
        edge.and_then(|edge| self.input.storage.as_storage().get(&edge))
            .map(|edge| edge.vertex)
    }
}

impl<M, G, C> From<EdgeCirculator<M, G, C>> for VertexCirculator<M, G, C>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    fn from(input: EdgeCirculator<M, G, C>) -> Self {
        VertexCirculator { input }
    }
}

impl<'a, M, G, C> Iterator for VertexCirculator<&'a M, G, C>
where
    M: 'a + AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    G: 'a + Geometry,
    C: Consistency,
{
    type Item = VertexView<&'a M, G, C>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self)
            .and_then(|key| VertexView::from_keyed_storage(key, self.input.storage))
    }
}

impl<'a, M, G, C> Iterator for VertexCirculator<&'a mut M, G, C>
where
    M: 'a + AsStorage<Edge<G>> + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: 'a + Geometry,
    C: Consistency,
{
    type Item = OrphanVertexView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).and_then(|key| {
            (key, unsafe {
                // Apply `'a` to the autoref from `as_storage_mut` and
                // `get_mut`.
                mem::transmute::<&'_ mut Vertex<G>, &'a mut Vertex<G>>(
                    self.input.storage.as_storage_mut().get_mut(&key).unwrap(),
                )
            }).into_view()
        })
    }
}

pub struct EdgeCirculator<M, G, C>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    storage: M,
    edge: Option<EdgeKey>,
    breadcrumb: Option<EdgeKey>,
    phantom: PhantomData<(G, C)>,
}

impl<M, G, C> EdgeCirculator<M, G, C>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    fn from_keyed_storage(key: EdgeKey, storage: M) -> Self {
        EdgeCirculator {
            storage,
            edge: Some(key),
            breadcrumb: Some(key),
            phantom: PhantomData,
        }
    }

    fn next(&mut self) -> Option<EdgeKey> {
        self.edge.and_then(|edge| {
            let next = self
                .storage
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

impl<M, G, C> From<FaceView<M, G, C>> for EdgeCirculator<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    fn from(face: FaceView<M, G, C>) -> Self {
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

impl<'a, M, G, C> Iterator for EdgeCirculator<&'a M, G, C>
where
    M: 'a + AsStorage<Edge<G>>,
    G: 'a + Geometry,
    C: Consistency,
{
    type Item = EdgeView<&'a M, G, C>;

    fn next(&mut self) -> Option<Self::Item> {
        EdgeCirculator::next(self).and_then(|key| EdgeView::from_keyed_storage(key, self.storage))
    }
}

impl<'a, M, G, C> Iterator for EdgeCirculator<&'a mut M, G, C>
where
    M: 'a + AsStorage<Edge<G>> + AsStorageMut<Edge<G>>,
    G: 'a + Geometry,
    C: Consistency,
{
    type Item = OrphanEdgeView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        EdgeCirculator::next(self).and_then(|key| {
            (key, unsafe {
                // Apply `'a` to the autoref from `as_storage_mut` and
                // `get_mut`.
                mem::transmute::<&'_ mut Edge<G>, &'a mut Edge<G>>(
                    self.storage.as_storage_mut().get_mut(&key).unwrap(),
                )
            }).into_view()
        })
    }
}

pub struct FaceCirculator<M, G, C>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    input: EdgeCirculator<M, G, C>,
}

impl<M, G, C> FaceCirculator<M, G, C>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    fn next(&mut self) -> Option<FaceKey> {
        while let Some(edge) = self
            .input
            .next()
            .and_then(|edge| self.input.storage.as_storage().get(&edge))
        {
            if let Some(face) = edge
                .opposite
                .and_then(|opposite| self.input.storage.as_storage().get(&opposite))
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

impl<M, G, C> From<EdgeCirculator<M, G, C>> for FaceCirculator<M, G, C>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
    C: Consistency,
{
    fn from(input: EdgeCirculator<M, G, C>) -> Self {
        FaceCirculator { input }
    }
}

impl<'a, M, G, C> Iterator for FaceCirculator<&'a M, G, C>
where
    M: 'a + AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: 'a + Geometry,
    C: Consistency,
{
    type Item = FaceView<&'a M, G, C>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self)
            .and_then(|key| FaceView::from_keyed_storage(key, self.input.storage))
    }
}

impl<'a, M, G, C> Iterator for FaceCirculator<&'a mut M, G, C>
where
    M: 'a + AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: 'a + Geometry,
    C: Consistency,
{
    type Item = OrphanFaceView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).and_then(|key| {
            (key, unsafe {
                // Apply `'a` to the autoref from `as_storage_mut` and
                // `get_mut`.
                mem::transmute::<&'_ mut Face<G>, &'a mut Face<G>>(
                    self.input.storage.as_storage_mut().get_mut(&key).unwrap(),
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
