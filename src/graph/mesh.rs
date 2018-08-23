use arrayvec::ArrayVec;
use failure::{Error, Fail};
use itertools::Itertools;
use num::{Integer, NumCast, Unsigned};
use std::collections::HashMap;
use std::hash::Hash;
use std::iter::FromIterator;
use std::marker::PhantomData;

use buffer::MeshBuffer;
use generate::{
    self, Arity, FromIndexer, HashIndexer, IndexVertices, Indexer, IntoVertices, MapVerticesInto,
    Quad,
};
use geometry::convert::{FromGeometry, FromInteriorGeometry, IntoGeometry};
use geometry::Geometry;
use graph::container::alias::OwnedCore;
use graph::container::{Bind, Consistent, Container, Core};
use graph::geometry::FaceCentroid;
use graph::mutation::{Mutate, Mutation};
use graph::storage::alias::InnerKey;
use graph::storage::convert::{AsStorage, AsStorageMut};
use graph::storage::{EdgeKey, FaceKey, Storage, VertexKey};
use graph::topology::{Edge, Face, Topological, Vertex};
use graph::view::convert::{FromKeyedSource, IntoView};
use graph::view::{
    EdgeView, FaceView, OrphanEdgeView, OrphanFaceView, OrphanVertexView, VertexView,
};
use graph::GraphError;

/// Half-edge graph representation of a mesh.
///
/// Provides topological data in the form of vertices, half-edges, and faces. A
/// half-edge is directed from one vertex to another, with an opposing
/// half-edge joining the vertices in the other direction.
///
/// `Mesh`es expose topological views, which can be used to traverse and
/// manipulate topology and geometry.
///
/// See the module documentation for more details.
pub struct Mesh<G = ()>
where
    G: Geometry,
{
    core: OwnedCore<G>,
}

/// Storage.
impl<G> Mesh<G>
where
    G: Geometry,
{
    fn as_storage<T>(&self) -> &Storage<T>
    where
        Self: AsStorage<T>,
        T: Topological,
    {
        AsStorage::<T>::as_storage(self)
    }

    fn as_storage_mut<T>(&mut self) -> &mut Storage<T>
    where
        Self: AsStorageMut<T>,
        T: Topological,
    {
        AsStorageMut::<T>::as_storage_mut(self)
    }
}

impl<G> Mesh<G>
where
    G: Geometry,
{
    /// Creates an empty `Mesh`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::graph::Mesh;
    ///
    /// let mut mesh = Mesh::<()>::new();
    /// ```
    pub fn new() -> Self {
        Mesh::from(
            Core::empty()
                .bind(Storage::<Vertex<G>>::new())
                .bind(Storage::<Edge<G>>::new())
                .bind(Storage::<Face<G>>::new()),
        )
    }

    /// Creates an empty `Mesh`.
    ///
    /// Underlying storage has zero capacity and does not allocate until the
    /// first insertion.
    pub fn empty() -> Self {
        Mesh::from(
            Core::empty()
                .bind(Storage::<Vertex<G>>::empty())
                .bind(Storage::<Edge<G>>::empty())
                .bind(Storage::<Face<G>>::empty()),
        )
    }

    /// Creates a `Mesh` from raw index and vertex buffers. The arity of the
    /// polygons in the index buffer must be known and constant.
    ///
    /// # Errors
    ///
    /// Returns an error if the arity of the index buffer is not constant, any
    /// index is out of bounds, or there is an error inserting topology into
    /// the mesh.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// use nalgebra::Point3;
    /// use plexus::generate::sphere::UvSphere;
    /// use plexus::generate::LruIndexer;
    /// use plexus::graph::Mesh;
    /// use plexus::prelude::*;
    ///
    /// # fn main() {
    /// let (indeces, positions) = UvSphere::new(16, 16)
    ///     .polygons_with_position()
    ///     .triangulate()
    ///     .flat_index_vertices(LruIndexer::with_capacity(256));
    /// let mut mesh = Mesh::<Point3<f64>>::from_raw_buffers(indeces, positions, 3);
    /// # }
    /// ```
    pub fn from_raw_buffers<I, J>(indeces: I, vertices: J, arity: usize) -> Result<Self, Error>
    where
        I: IntoIterator<Item = usize>,
        J: IntoIterator,
        J::Item: IntoGeometry<G::Vertex>,
    {
        let mut mutation = Mutation::mutate(Mesh::new());
        let vertices = vertices
            .into_iter()
            .map(|vertex| mutation.insert_vertex(vertex.into_geometry()))
            .collect::<Vec<_>>();
        for face in &indeces.into_iter().chunks(arity) {
            let face = face.collect::<Vec<_>>();
            if face.len() != arity {
                return Err(GraphError::ArityConflict {
                    expected: arity,
                    actual: face.len(),
                }.context("index buffer lenght is not a multiple of arity")
                    .into());
            }
            let mut perimeter = Vec::with_capacity(arity);
            for index in face {
                perimeter.push(
                    *vertices
                        .get(index)
                        .ok_or_else(|| Error::from(GraphError::TopologyNotFound))?,
                );
            }
            mutation.insert_face(&perimeter, Default::default())?;
        }
        mutation.commit()
    }

    /// Gets the number of vertices in the mesh.
    pub fn vertex_count(&self) -> usize {
        self.as_storage::<Vertex<G>>().len()
    }

    /// Gets an immutable view of the vertex with the given key.
    pub fn vertex(&self, key: VertexKey) -> Option<VertexView<&Self, G>> {
        (key, self).into_view()
    }

    /// Gets a mutable view of the vertex with the given key.
    pub fn vertex_mut(&mut self, key: VertexKey) -> Option<VertexView<&mut Self, G>> {
        (key, self).into_view()
    }

    /// Gets an iterator of immutable views over the vertices in the mesh.
    pub fn vertices(&self) -> impl Iterator<Item = VertexView<&Self, G>> {
        Iter::<_, Vertex<G>, _, _>::from((self.as_storage::<Vertex<G>>().keys(), self))
    }

    /// Gets an iterator of orphan views over the vertices in the mesh.
    ///
    /// Because this only yields orphan views, only geometry can be mutated.
    /// For topological mutations, collect the necessary keys and use
    /// `vertex_mut` instead.
    pub fn orphan_vertices(&mut self) -> impl Iterator<Item = OrphanVertexView<G>> {
        IterMut::from(self.as_storage_mut::<Vertex<G>>().iter_mut())
    }

    /// Gets the number of edges in the mesh.
    pub fn edge_count(&self) -> usize {
        self.as_storage::<Edge<G>>().len()
    }

    /// Gets an immutable view of the edge with the given key.
    pub fn edge(&self, key: EdgeKey) -> Option<EdgeView<&Self, G>> {
        (key, self).into_view()
    }

    /// Gets a mutable view of the edge with the given key.
    pub fn edge_mut(&mut self, key: EdgeKey) -> Option<EdgeView<&mut Self, G>> {
        (key, self).into_view()
    }

    /// Gets an iterator of immutable views over the edges in the mesh.
    pub fn edges(&self) -> impl Iterator<Item = EdgeView<&Self, G>> {
        Iter::<_, Edge<G>, _, _>::from((self.as_storage::<Edge<G>>().keys(), self))
    }

    /// Gets an iterator of orphan views over the edges in the mesh.
    ///
    /// Because this only yields orphan views, only geometry can be mutated.
    /// For topological mutations, collect the necessary keys and use
    /// `edge_mut` instead.
    pub fn orphan_edges(&mut self) -> impl Iterator<Item = OrphanEdgeView<G>> {
        IterMut::from(self.as_storage_mut::<Edge<G>>().iter_mut())
    }

    /// Gets the number of faces in the mesh.
    pub fn face_count(&self) -> usize {
        self.as_storage::<Face<G>>().len()
    }

    /// Gets an immutable view of the face with the given key.
    pub fn face(&self, key: FaceKey) -> Option<FaceView<&Self, G>> {
        (key, self).into_view()
    }

    /// Gets a mutable view of the face with the given key.
    pub fn face_mut(&mut self, key: FaceKey) -> Option<FaceView<&mut Self, G>> {
        (key, self).into_view()
    }

    /// Gets an iterator of immutable views over the faces in the mesh.
    pub fn faces(&self) -> impl Iterator<Item = FaceView<&Self, G>> {
        Iter::<_, Face<G>, _, _>::from((self.as_storage::<Face<G>>().keys(), self))
    }

    /// Gets an iterator of orphan views over the faces in the mesh.
    ///
    /// Because this only yields orphan views, only geometry can be mutated.
    /// For topological mutations, collect the necessary keys and use
    /// `face_mut` instead.
    pub fn orphan_faces(&mut self) -> impl Iterator<Item = OrphanFaceView<G>> {
        IterMut::from(self.as_storage_mut::<Face<G>>().iter_mut())
    }

    /// Triangulates the mesh, tesselating all faces into triangles.
    pub fn triangulate(&mut self) -> Result<(), Error>
    where
        G: FaceCentroid<Centroid = <G as Geometry>::Vertex> + Geometry,
    {
        let faces = <Self as AsStorage<Face<G>>>::as_storage(self)
            .keys()
            .map(|key| FaceKey::from(*key))
            .collect::<Vec<_>>();
        for face in faces {
            let face = FaceView::<&mut Self, G>::from_keyed_source((face, self)).unwrap();
            face.triangulate()?;
        }
        Ok(())
    }

    /// Creates a mesh buffer from the mesh.
    ///
    /// The buffer is created using the vertex geometry of each unique vertex.
    ///
    /// # Errors
    ///
    /// Returns an error if the mesh does not have constant arity. Typically, a
    /// mesh is triangulated before being converted to a mesh buffer.
    pub fn to_mesh_buffer_by_vertex<N, V>(&self) -> Result<MeshBuffer<N, V>, Error>
    where
        G::Vertex: IntoGeometry<V>,
        N: Copy + Integer + NumCast + Unsigned,
    {
        self.to_mesh_buffer_by_vertex_with(|vertex| vertex.geometry.clone().into_geometry())
    }

    /// Creates a mesh buffer from the mesh.
    ///
    /// The buffer is created using each unique vertex, which is converted into
    /// the buffer geometry by the given function.
    ///
    /// # Errors
    ///
    /// Returns an error if the mesh does not have constant arity. Typically, a
    /// mesh is triangulated before being converted to a mesh buffer.
    pub fn to_mesh_buffer_by_vertex_with<N, V, F>(
        &self,
        mut f: F,
    ) -> Result<MeshBuffer<N, V>, Error>
    where
        N: Copy + Integer + NumCast + Unsigned,
        F: FnMut(VertexView<&Self, G>) -> V,
    {
        let (keys, vertices) = {
            let mut keys = HashMap::with_capacity(self.vertex_count());
            let mut vertices = Vec::with_capacity(self.vertex_count());
            for (n, vertex) in self.vertices().enumerate() {
                keys.insert(vertex.key(), n);
                vertices.push(f(vertex));
            }
            (keys, vertices)
        };
        let indeces = {
            let arity = match self.faces().nth(0) {
                Some(face) => face.arity(),
                _ => 0,
            };
            let mut indeces = Vec::with_capacity(arity * self.face_count());
            for face in self.faces() {
                if face.arity() != arity {
                    return Err(GraphError::ArityNonConstant.into());
                }
                for vertex in face.vertices() {
                    indeces.push(N::from(keys[&vertex.key()]).unwrap());
                }
            }
            indeces
        };
        MeshBuffer::from_raw_buffers(indeces, vertices)
    }

    /// Creates a mesh buffer from the mesh.
    ///
    /// The buffer is created using the vertex geometry of each face. Shared
    /// vertices are included for each face to which they belong.
    ///
    /// # Errors
    ///
    /// Returns an error if the mesh does not have constant arity. Typically, a
    /// mesh is triangulated before being converted to a mesh buffer.
    pub fn to_mesh_buffer_by_face<N, V>(&self) -> Result<MeshBuffer<N, V>, Error>
    where
        G::Vertex: IntoGeometry<V>,
        N: Copy + Integer + NumCast + Unsigned,
    {
        self.to_mesh_buffer_by_face_with(|_, vertex| vertex.geometry.clone().into_geometry())
    }

    /// Creates a mesh buffer from the mesh.
    ///
    /// The buffer is created from each face, which is converted into the
    /// buffer geometry by the given function.
    ///
    /// # Errors
    ///
    /// Returns an error if the mesh does not have constant arity. Typically, a
    /// mesh is triangulated before being converted to a mesh buffer.
    pub fn to_mesh_buffer_by_face_with<N, V, F>(&self, mut f: F) -> Result<MeshBuffer<N, V>, Error>
    where
        N: Copy + Integer + NumCast + Unsigned,
        F: FnMut(FaceView<&Self, G>, VertexView<&Self, G>) -> V,
    {
        let vertices = {
            let arity = match self.faces().nth(0) {
                Some(face) => face.arity(),
                _ => 0,
            };
            let mut vertices = Vec::with_capacity(arity * self.face_count());
            for face in self.faces() {
                if face.arity() != arity {
                    return Err(GraphError::ArityNonConstant.into());
                }
                for vertex in face.vertices() {
                    // TODO: Can some sort of dereference be used here?
                    vertices.push(f(face, vertex));
                }
            }
            vertices
        };
        MeshBuffer::from_raw_buffers(
            // TODO: Cannot use the bound `N: Step`, which is unstable.
            (0..vertices.len()).map(|index| N::from(index).unwrap()),
            vertices,
        )
    }
}

impl<G> AsStorage<Vertex<G>> for Mesh<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<Vertex<G>> {
        self.core.as_storage::<Vertex<G>>()
    }
}

impl<G> AsStorage<Edge<G>> for Mesh<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<Edge<G>> {
        self.core.as_storage::<Edge<G>>()
    }
}

impl<G> AsStorage<Face<G>> for Mesh<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<Face<G>> {
        self.core.as_storage::<Face<G>>()
    }
}

impl<G> AsStorageMut<Vertex<G>> for Mesh<G>
where
    G: Geometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<Vertex<G>> {
        self.core.as_storage_mut::<Vertex<G>>()
    }
}

impl<G> AsStorageMut<Edge<G>> for Mesh<G>
where
    G: Geometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<Edge<G>> {
        self.core.as_storage_mut::<Edge<G>>()
    }
}

impl<G> AsStorageMut<Face<G>> for Mesh<G>
where
    G: Geometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<Face<G>> {
        self.core.as_storage_mut::<Face<G>>()
    }
}

impl<G> Default for Mesh<G>
where
    G: Geometry,
{
    fn default() -> Self {
        // Because `default` is likely to be used in more generic contexts,
        // `empty` is used to avoid any unnecessary allocations.
        Mesh::empty()
    }
}

impl<G> From<OwnedCore<G>> for Mesh<G>
where
    G: Geometry,
{
    fn from(core: OwnedCore<G>) -> Self {
        Mesh { core }
    }
}

impl<G> Into<OwnedCore<G>> for Mesh<G>
where
    G: Geometry,
{
    fn into(self) -> OwnedCore<G> {
        let Mesh { core, .. } = self;
        core
    }
}

impl<G, H> FromInteriorGeometry<Mesh<H>> for Mesh<G>
where
    G: Geometry,
    G::Vertex: FromGeometry<H::Vertex>,
    G::Edge: FromGeometry<H::Edge>,
    G::Face: FromGeometry<H::Face>,
    H: Geometry,
{
    fn from_interior_geometry(mesh: Mesh<H>) -> Self {
        let Mesh { core, .. } = mesh;
        let (vertices, edges, faces) = core.into_storage();
        let core = Core::empty()
            .bind(vertices.map_values_into(|vertex| Vertex::<G>::from_interior_geometry(vertex)))
            .bind(edges.map_values_into(|edge| Edge::<G>::from_interior_geometry(edge)))
            .bind(faces.map_values_into(|face| Face::<G>::from_interior_geometry(face)));
        Mesh::from(core)
    }
}

impl<G, P> FromIndexer<P, P> for Mesh<G>
where
    G: Geometry,
    P: MapVerticesInto<usize> + generate::Topological,
    P::Output: IntoVertices,
    <P::Output as IntoVertices>::Output: AsRef<[usize]>,
    P::Vertex: IntoGeometry<G::Vertex>,
{
    fn from_indexer<I, N>(input: I, indexer: N) -> Self
    where
        I: IntoIterator<Item = P>,
        N: Indexer<P, P::Vertex>,
    {
        let mut mutation = Mutation::mutate(Mesh::new());
        let (indeces, vertices) = input.into_iter().index_vertices(indexer);
        let vertices = vertices
            .into_iter()
            .map(|vertex| mutation.insert_vertex(vertex.into_geometry()))
            .collect::<Vec<_>>();
        for face in indeces {
            let face = face.into_vertices();
            // The topology with the greatest arity emitted by indexing is a
            // quad. Avoid allocations by using an `ArrayVec`.
            let mut perimeter = ArrayVec::<[_; Quad::<usize>::ARITY]>::new();
            for index in face {
                perimeter.push(vertices[index]);
            }
            mutation
                .insert_face(&perimeter, Default::default())
                .unwrap();
        }
        mutation.commit().unwrap()
    }
}

impl<G, P> FromIterator<P> for Mesh<G>
where
    G: Geometry,
    P: MapVerticesInto<usize> + generate::Topological,
    P::Output: IntoVertices,
    <P::Output as IntoVertices>::Output: AsRef<[usize]>,
    P::Vertex: Eq + Hash + IntoGeometry<G::Vertex>,
{
    fn from_iter<I>(input: I) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        Self::from_indexer(input, HashIndexer::default())
    }
}

impl<G> Container for Mesh<G>
where
    G: Geometry,
{
    type Contract = Consistent;
}

pub struct Iter<'a, I, T, G, Output>
where
    I: 'a + Iterator<Item = &'a InnerKey<T>>,
    T: 'a + Topological,
    G: 'a + Geometry,
    (T::Key, &'a Mesh<G>): IntoView<Output>,
{
    input: I,
    storage: &'a Mesh<G>,
    phantom: PhantomData<(T, Output)>,
}

impl<'a, I, T, G, Output> From<(I, &'a Mesh<G>)> for Iter<'a, I, T, G, Output>
where
    I: 'a + Iterator<Item = &'a InnerKey<T>>,
    T: 'a + Topological,
    G: 'a + Geometry,
    (T::Key, &'a Mesh<G>): IntoView<Output>,
{
    fn from(source: (I, &'a Mesh<G>)) -> Self {
        let (input, storage) = source;
        Iter {
            input,
            storage,
            phantom: PhantomData,
        }
    }
}

impl<'a, I, T, G, Output> Iterator for Iter<'a, I, T, G, Output>
where
    I: 'a + Iterator<Item = &'a InnerKey<T>>,
    T: 'a + Topological,
    G: 'a + Geometry,
    (T::Key, &'a Mesh<G>): IntoView<Output>,
{
    type Item = Output;

    fn next(&mut self) -> Option<Self::Item> {
        self.input
            .next()
            .map(|key| ((*key).into(), self.storage).into_view().unwrap())
    }
}

pub struct IterMut<'a, I, T, Output>
where
    I: 'a + Iterator<Item = (&'a InnerKey<T>, &'a mut T)>,
    T: 'a + Topological,
    (T::Key, &'a mut T): IntoView<Output>,
{
    input: I,
    phantom: PhantomData<(T, Output)>,
}

impl<'a, I, T, Output> From<I> for IterMut<'a, I, T, Output>
where
    I: 'a + Iterator<Item = (&'a InnerKey<T>, &'a mut T)>,
    T: 'a + Topological,
    (T::Key, &'a mut T): IntoView<Output>,
{
    fn from(input: I) -> Self {
        IterMut {
            input,
            phantom: PhantomData,
        }
    }
}

impl<'a, I, T, Output> Iterator for IterMut<'a, I, T, Output>
where
    I: 'a + Iterator<Item = (&'a InnerKey<T>, &'a mut T)>,
    T: 'a + Topological,
    (T::Key, &'a mut T): IntoView<Output>,
{
    type Item = Output;

    fn next(&mut self) -> Option<Self::Item> {
        self.input.next().map(|entry| {
            ((*entry.0).into(), unsafe {
                use std::mem;

                mem::transmute::<&'_ mut T, &'a mut T>(entry.1)
            }).into_view()
                .unwrap()
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point2, Point3, Vector3};
    use num::Zero;
    use std::collections::HashSet;

    use generate::*;
    use geometry::*;
    use graph::mutation::face::FaceRemoveCache;
    use graph::mutation::{Mutate, Mutation};
    use graph::*;

    #[test]
    fn collect_topology_into_mesh() {
        let mesh = sphere::UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();

        assert_eq!(5, mesh.vertex_count());
        assert_eq!(18, mesh.edge_count());
        assert_eq!(6, mesh.face_count());
    }

    #[test]
    fn iterate_mesh_topology() {
        let mut mesh = sphere::UvSphere::new(4, 2)
            .polygons_with_position() // 8 triangles, 24 vertices.
            .collect::<Mesh<Point3<f32>>>();

        assert_eq!(6, mesh.vertices().count());
        assert_eq!(24, mesh.edges().count());
        assert_eq!(8, mesh.faces().count());
        for vertex in mesh.vertices() {
            // Every vertex is connected to 4 triangles with 4 (incoming)
            // edges. Traversal of topology should be possible.
            assert_eq!(4, vertex.incoming_edges().count());
        }
        for mut vertex in mesh.orphan_vertices() {
            // Geometry should be mutable.
            vertex.geometry += Vector3::zero();
        }
    }

    #[test]
    fn non_manifold_error_deferred() {
        let mesh = sphere::UvSphere::new(32, 32)
            .polygons_with_position()
            .triangulate()
            .collect::<Mesh<Point3<f32>>>();
        // This conversion will join faces by a single vertex, but ultimately
        // creates a manifold.
        mesh.to_mesh_buffer_by_face_with::<usize, Point3<f32>, _>(|_, vertex| vertex.geometry)
            .unwrap();
    }

    #[test]
    fn error_on_non_manifold_mesh() {
        // Construct a mesh with a "fan" of three triangles sharing the same
        // edge along the Z-axis. The edge would have three associated faces,
        // which should not be possible.
        let mesh = Mesh::<Point3<i32>>::from_raw_buffers(
            vec![0, 1, 2, 0, 1, 3, 0, 1, 4],
            vec![(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
            3,
        );

        assert!(match *mesh
            .err()
            .unwrap()
            .root_cause()
            .downcast_ref::<GraphError>()
            .unwrap()
        {
            GraphError::TopologyConflict => true,
            _ => false,
        });
    }

    #[test]
    fn error_on_singularity_mesh() {
        // Construct a mesh with three non-neighboring triangles sharing a
        // single vertex.
        let mesh = Mesh::<Point3<i32>>::from_raw_buffers(
            vec![0, 1, 2, 0, 3, 4, 0, 5, 6],
            vec![
                (0, 0, 0),
                (1, -1, 0),
                (-1, -1, 0),
                (-3, 1, 0),
                (-2, 1, 0),
                (2, 1, 0),
                (3, 1, 0),
            ],
            3,
        );

        assert!(match *mesh
            .err()
            .unwrap()
            .root_cause()
            .downcast_ref::<GraphError>()
            .unwrap()
        {
            GraphError::TopologyMalformed => true,
            _ => false,
        });

        // Construct a mesh with three triangles forming a rectangle, where one
        // vertex (at the origin) is shared by all three triangles.
        let mesh = Mesh::<Point2<i32>>::from_raw_buffers(
            vec![0, 1, 3, 1, 4, 3, 1, 2, 4],
            vec![(-1, 0), (0, 0), (1, 0), (-1, 1), (1, 1)],
            3,
        ).unwrap();
        // TODO: Create a shared testing geometry that allows topology to be
        //       marked and more easily located. Finding very specific geometry
        //       like this is cumbersome.
        // Find the "center" triangle and use a mutation to remove it. This
        // creates a singularity, with the two remaining triangles sharing no
        // edges but having a single common vertex.
        let geometry = &[(0, 0), (1, 1), (-1, 1)]
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        let key = mesh
            .faces()
            .find(|face| {
                face.vertices()
                    .map(|vertex| vertex.geometry.clone())
                    .map(|position| (position.x, position.y))
                    .collect::<HashSet<_>>() == *geometry
            })
            .unwrap()
            .key();
        let cache = FaceRemoveCache::snapshot(&mesh, key).unwrap();
        let mut mutation = Mutation::mutate(mesh);
        assert!(match *mutation
            .remove_face_with_cache(cache)
            .err()
            .unwrap()
            .root_cause()
            .downcast_ref::<GraphError>()
            .unwrap()
        {
            GraphError::TopologyConflict => true,
            _ => false,
        });
    }

    // This test is a sanity check for mesh iterators, topological views, and
    // the unsafe transmutations used to coerce lifetimes.
    #[test]
    fn read_write_geometry_ref() {
        impl Attribute for f32 {}

        struct ValueGeometry;

        impl Geometry for ValueGeometry {
            type Vertex = Point3<f32>;
            type Edge = ();
            type Face = f32;
        }

        // Create a mesh with a floating point value associated with each face.
        // Use a mutable iterator to write to the geometry of each face.
        let mut mesh = sphere::UvSphere::new(4, 4)
            .polygons_with_position()
            .collect::<Mesh<ValueGeometry>>();
        let value = 3.14;
        for mut face in mesh.orphan_faces() {
            face.geometry = value;
        }

        // Read the geometry of each face using an immutable iterator to ensure
        // it is what we expect.
        for face in mesh.faces() {
            assert_eq!(value, face.geometry);
        }
    }
}
