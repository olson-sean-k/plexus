use arrayvec::ArrayVec;
use itertools::Itertools;
use num::{Integer, NumCast, Unsigned};
use std::collections::HashMap;
use std::hash::Hash;
use std::iter::FromIterator;
use std::marker::PhantomData;
use typenum::{self, NonZero};

use buffer::{Flat, IndexBuffer, MeshBuffer};
use geometry::convert::{FromGeometry, FromInteriorGeometry, IntoGeometry};
use geometry::Geometry;
use graph::container::alias::OwnedCore;
use graph::container::{Bind, Consistent, Core};
use graph::geometry::FaceCentroid;
use graph::mutation::{Mutate, Mutation};
use graph::storage::convert::{AsStorage, AsStorageMut};
use graph::storage::{EdgeKey, FaceKey, Storage, VertexKey};
use graph::topology::{Edge, Face, Topological, Vertex};
use graph::view::convert::IntoView;
use graph::view::{
    EdgeView, FaceView, OrphanEdgeView, OrphanFaceView, OrphanVertexView, VertexView,
};
use graph::GraphError;
use primitive::decompose::IntoVertices;
use primitive::index::{FromIndexer, HashIndexer, IndexVertices, Indexer};
use primitive::{self, Arity, Map, Quad};

/// Half-edge graph representation of a mesh.
///
/// Provides topological data in the form of vertices, half-edges, and faces. A
/// half-edge is directed from one vertex to another, with an opposing
/// half-edge joining the vertices in the other direction.
///
/// `MeshGraph`s expose topological views, which can be used to traverse and
/// manipulate topology and geometry.
///
/// See the module documentation for more details.
pub struct MeshGraph<G = ()>
where
    G: Geometry,
{
    core: OwnedCore<G>,
}

/// Storage.
impl<G> MeshGraph<G>
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

impl<G> MeshGraph<G>
where
    G: Geometry,
{
    /// Creates an empty `MeshGraph`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::graph::MeshGraph;
    ///
    /// let mut graph = MeshGraph::<()>::new();
    /// ```
    pub fn new() -> Self {
        MeshGraph::from(
            Core::empty()
                .bind(Storage::<Vertex<G>>::new())
                .bind(Storage::<Edge<G>>::new())
                .bind(Storage::<Face<G>>::new()),
        )
    }

    /// Creates an empty `MeshGraph`.
    ///
    /// Underlying storage has zero capacity and does not allocate until the
    /// first insertion.
    pub fn empty() -> Self {
        MeshGraph::from(
            Core::empty()
                .bind(Storage::<Vertex<G>>::empty())
                .bind(Storage::<Edge<G>>::empty())
                .bind(Storage::<Face<G>>::empty()),
        )
    }

    /// Creates a `MeshGraph` from raw index and vertex buffers. The arity of
    /// the polygons in the index buffer must be known and constant.
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
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::index::LruIndexer;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// # fn main() {
    /// let (indices, positions) = UvSphere::new(16, 16)
    ///     .polygons_with_position()
    ///     .triangulate()
    ///     .flat_index_vertices(LruIndexer::with_capacity(256));
    /// let mut graph = MeshGraph::<Point3<f64>>::from_raw_buffers(indices, positions, 3).unwrap();
    /// # }
    /// ```
    pub fn from_raw_buffers<I, J>(indices: I, vertices: J, arity: usize) -> Result<Self, GraphError>
    where
        I: IntoIterator<Item = usize>,
        J: IntoIterator,
        J::Item: IntoGeometry<G::Vertex>,
    {
        let mut mutation = Mutation::mutate(MeshGraph::new());
        let vertices = vertices
            .into_iter()
            .map(|vertex| mutation.insert_vertex(vertex.into_geometry()))
            .collect::<Vec<_>>();
        for face in &indices.into_iter().chunks(arity) {
            let face = face.collect::<Vec<_>>();
            if face.len() != arity {
                // Index buffer length is not a multiple of arity.
                return Err(GraphError::ArityConflict {
                    expected: arity,
                    actual: face.len(),
                });
            }
            let mut perimeter = Vec::with_capacity(arity);
            for index in face {
                perimeter.push(
                    *vertices
                        .get(index)
                        .ok_or_else(|| GraphError::TopologyNotFound)?,
                );
            }
            mutation.insert_face(&perimeter, Default::default())?;
        }
        mutation.commit()
    }

    /// Creates a `MeshGraph` from a `MeshBuffer`. The arity of the polygons in
    /// the index buffer must be known and constant.
    ///
    /// `MeshGraph` also implements `From` for `MeshBuffer`, but will panic if
    /// the conversion fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// use nalgebra::Point2;
    /// use plexus::buffer::{Flat4, MeshBuffer};
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    ///
    /// # fn main() {
    /// let buffer = MeshBuffer::<Flat4, _>::from_raw_buffers(
    ///     vec![0, 1, 2, 3],
    ///     vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    /// )
    /// .unwrap();
    /// let mut graph = MeshGraph::<Point2<f32>>::from_mesh_buffer(buffer).unwrap();
    /// # }
    /// ```
    pub fn from_mesh_buffer<A, N, H>(buffer: MeshBuffer<Flat<A, N>, H>) -> Result<Self, GraphError>
    where
        A: NonZero + typenum::Unsigned,
        N: Copy + Integer + NumCast + Unsigned,
        H: Clone + IntoGeometry<G::Vertex>,
    {
        let arity = buffer.arity().unwrap();
        let (indices, vertices) = buffer.into_raw_buffers();
        MeshGraph::from_raw_buffers(
            indices
                .into_iter()
                .map(|index| NumCast::from(index).unwrap()),
            vertices.into_iter(),
            arity,
        )
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
    pub fn triangulate(&mut self) -> Result<(), GraphError>
    where
        G: FaceCentroid<Centroid = <G as Geometry>::Vertex> + Geometry,
    {
        let faces = <Self as AsStorage<Face<G>>>::as_storage(self)
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        for face in faces {
            self.face_mut(face).unwrap().triangulate()?;
        }
        Ok(())
    }

    /// Creates a mesh buffer from the mesh.
    ///
    /// The buffer is created using the vertex geometry of each unique vertex.
    ///
    /// # Errors
    ///
    /// Returns an error if the mesh does not have constant arity that is
    /// compatible with the index buffer. Typically, a mesh is triangulated
    /// before being converted to a mesh buffer.
    pub fn to_mesh_buffer_by_vertex<A, N, H>(&self) -> Result<MeshBuffer<Flat<A, N>, H>, GraphError>
    where
        G::Vertex: IntoGeometry<H>,
        A: NonZero + typenum::Unsigned,
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
    /// Returns an error if the mesh does not have constant arity that is
    /// compatible with the index buffer. Typically, a mesh is triangulated
    /// before being converted to a mesh buffer.
    pub fn to_mesh_buffer_by_vertex_with<A, N, H, F>(
        &self,
        mut f: F,
    ) -> Result<MeshBuffer<Flat<A, N>, H>, GraphError>
    where
        A: NonZero + typenum::Unsigned,
        N: Copy + Integer + NumCast + Unsigned,
        F: FnMut(VertexView<&Self, G>) -> H,
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
        let indices = {
            let arity = Flat::<A, N>::ARITY.unwrap();
            let mut indices = Vec::with_capacity(arity * self.face_count());
            for face in self.faces() {
                if face.arity() != arity {
                    return Err(GraphError::ArityConflict {
                        expected: arity,
                        actual: face.arity(),
                    });
                }
                for vertex in face.vertices() {
                    indices.push(N::from(keys[&vertex.key()]).unwrap());
                }
            }
            indices
        };
        MeshBuffer::<Flat<_, _>, _>::from_raw_buffers(indices, vertices)
            .map_err(|error| error.into())
    }

    /// Creates a mesh buffer from the mesh.
    ///
    /// The buffer is created using the vertex geometry of each face. Shared
    /// vertices are included for each face to which they belong.
    ///
    /// # Errors
    ///
    /// Returns an error if the mesh does not have constant arity that is
    /// compatible with the index buffer. Typically, a mesh is triangulated
    /// before being converted to a mesh buffer.
    pub fn to_mesh_buffer_by_face<A, N, H>(&self) -> Result<MeshBuffer<Flat<A, N>, H>, GraphError>
    where
        G::Vertex: IntoGeometry<H>,
        A: NonZero + typenum::Unsigned,
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
    /// Returns an error if the mesh does not have constant arity that is
    /// compatible with the index buffer. Typically, a mesh is triangulated
    /// before being converted to a mesh buffer.
    pub fn to_mesh_buffer_by_face_with<A, N, H, F>(
        &self,
        mut f: F,
    ) -> Result<MeshBuffer<Flat<A, N>, H>, GraphError>
    where
        A: NonZero + typenum::Unsigned,
        N: Copy + Integer + NumCast + Unsigned,
        F: FnMut(FaceView<&Self, G>, VertexView<&Self, G>) -> H,
    {
        let vertices = {
            let arity = Flat::<A, N>::ARITY.unwrap();
            let mut vertices = Vec::with_capacity(arity * self.face_count());
            for face in self.faces() {
                if face.arity() != arity {
                    return Err(GraphError::ArityConflict {
                        expected: arity,
                        actual: face.arity(),
                    });
                }
                for vertex in face.vertices() {
                    // TODO: Can some sort of dereference be used here?
                    vertices.push(f(face, vertex));
                }
            }
            vertices
        };
        MeshBuffer::<Flat<_, _>, _>::from_raw_buffers(
            // TODO: Cannot use the bound `N: Step`, which is unstable.
            (0..vertices.len()).map(|index| N::from(index).unwrap()),
            vertices,
        )
        .map_err(|error| error.into())
    }
}

impl<G> AsStorage<Vertex<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<Vertex<G>> {
        self.core.as_storage::<Vertex<G>>()
    }
}

impl<G> AsStorage<Edge<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<Edge<G>> {
        self.core.as_storage::<Edge<G>>()
    }
}

impl<G> AsStorage<Face<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<Face<G>> {
        self.core.as_storage::<Face<G>>()
    }
}

impl<G> AsStorageMut<Vertex<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<Vertex<G>> {
        self.core.as_storage_mut::<Vertex<G>>()
    }
}

impl<G> AsStorageMut<Edge<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<Edge<G>> {
        self.core.as_storage_mut::<Edge<G>>()
    }
}

impl<G> AsStorageMut<Face<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<Face<G>> {
        self.core.as_storage_mut::<Face<G>>()
    }
}

impl<G> Default for MeshGraph<G>
where
    G: Geometry,
{
    fn default() -> Self {
        // Because `default` is likely to be used in more generic contexts,
        // `empty` is used to avoid any unnecessary allocations.
        MeshGraph::empty()
    }
}

impl<A, N, H, G> From<MeshBuffer<Flat<A, N>, H>> for MeshGraph<G>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
    H: Clone + IntoGeometry<G::Vertex>,
    G: Geometry,
{
    fn from(buffer: MeshBuffer<Flat<A, N>, H>) -> Self {
        MeshGraph::from_mesh_buffer(buffer).unwrap()
    }
}

impl<G> From<OwnedCore<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn from(core: OwnedCore<G>) -> Self {
        MeshGraph { core }
    }
}

impl<G> Into<OwnedCore<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn into(self) -> OwnedCore<G> {
        let MeshGraph { core, .. } = self;
        core
    }
}

impl<G, H> FromInteriorGeometry<MeshGraph<H>> for MeshGraph<G>
where
    G: Geometry,
    G::Vertex: FromGeometry<H::Vertex>,
    G::Edge: FromGeometry<H::Edge>,
    G::Face: FromGeometry<H::Face>,
    H: Geometry,
{
    fn from_interior_geometry(graph: MeshGraph<H>) -> Self {
        let MeshGraph { core, .. } = graph;
        let (vertices, edges, faces) = core.into_storage();
        let core = Core::empty()
            .bind(vertices.map_values_into(|vertex| Vertex::<G>::from_interior_geometry(vertex)))
            .bind(edges.map_values_into(|edge| Edge::<G>::from_interior_geometry(edge)))
            .bind(faces.map_values_into(|face| Face::<G>::from_interior_geometry(face)));
        MeshGraph::from(core)
    }
}

impl<G, P> FromIndexer<P, P> for MeshGraph<G>
where
    G: Geometry,
    P: Map<usize> + primitive::Topological,
    P::Output: IntoVertices,
    P::Vertex: IntoGeometry<G::Vertex>,
{
    type Error = GraphError;

    fn from_indexer<I, N>(input: I, indexer: N) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = P>,
        N: Indexer<P, P::Vertex>,
    {
        let mut mutation = Mutation::mutate(MeshGraph::new());
        let (indices, vertices) = input.into_iter().index_vertices(indexer);
        let vertices = vertices
            .into_iter()
            .map(|vertex| mutation.insert_vertex(vertex.into_geometry()))
            .collect::<Vec<_>>();
        for face in indices {
            // The topology with the greatest arity emitted by indexing is a
            // quad. Avoid allocations by using an `ArrayVec`.
            let perimeter = face
                .into_vertices()
                .into_iter()
                .map(|index| vertices[index])
                .collect::<ArrayVec<[_; Quad::<usize>::ARITY]>>();
            mutation.insert_face(&perimeter, Default::default())?;
        }
        mutation.commit()
    }
}

impl<G, P> FromIterator<P> for MeshGraph<G>
where
    G: Geometry,
    P: Map<usize> + primitive::Topological,
    P::Output: IntoVertices,
    P::Vertex: Clone + Eq + Hash + IntoGeometry<G::Vertex>,
{
    fn from_iter<I>(input: I) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        Self::from_indexer(input, HashIndexer::default()).unwrap_or_else(|_| Self::default())
    }
}

impl<G> Consistent for MeshGraph<G> where G: Geometry {}

pub struct Iter<'a, I, T, G, Output>
where
    I: 'a + Iterator<Item = &'a T::Key>,
    T: 'a + Topological,
    G: 'a + Geometry,
    (T::Key, &'a MeshGraph<G>): IntoView<Output>,
{
    input: I,
    storage: &'a MeshGraph<G>,
    phantom: PhantomData<(T, Output)>,
}

impl<'a, I, T, G, Output> From<(I, &'a MeshGraph<G>)> for Iter<'a, I, T, G, Output>
where
    I: 'a + Iterator<Item = &'a T::Key>,
    T: 'a + Topological,
    G: 'a + Geometry,
    (T::Key, &'a MeshGraph<G>): IntoView<Output>,
{
    fn from(source: (I, &'a MeshGraph<G>)) -> Self {
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
    I: 'a + Iterator<Item = &'a T::Key>,
    T: 'a + Topological,
    G: 'a + Geometry,
    (T::Key, &'a MeshGraph<G>): IntoView<Output>,
{
    type Item = Output;

    fn next(&mut self) -> Option<Self::Item> {
        self.input
            .next()
            .map(|key| (*key, self.storage).into_view().unwrap())
    }
}

pub struct IterMut<'a, I, T, Output>
where
    I: 'a + Iterator<Item = (&'a T::Key, &'a mut T)>,
    T: 'a + Topological,
    (T::Key, &'a mut T): IntoView<Output>,
{
    input: I,
    phantom: PhantomData<(T, Output)>,
}

impl<'a, I, T, Output> From<I> for IterMut<'a, I, T, Output>
where
    I: 'a + Iterator<Item = (&'a T::Key, &'a mut T)>,
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
    I: 'a + Iterator<Item = (&'a T::Key, &'a mut T)>,
    T: 'a + Topological,
    (T::Key, &'a mut T): IntoView<Output>,
{
    type Item = Output;

    fn next(&mut self) -> Option<Self::Item> {
        self.input.next().map(|entry| {
            (*entry.0, unsafe {
                use std::mem;

                mem::transmute::<&'_ mut T, &'a mut T>(entry.1)
            })
                .into_view()
                .unwrap()
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point2, Point3, Vector3};
    use num::Zero;
    use std::collections::HashSet;

    use buffer::U3;
    use geometry::*;
    use graph::mutation::face::FaceRemoveCache;
    use graph::mutation::{Mutate, Mutation};
    use graph::*;
    use primitive::decompose::*;
    use primitive::generate::*;
    use primitive::sphere::UvSphere;

    #[test]
    fn collect_topology_into_mesh() {
        let graph = UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f32>>>();

        assert_eq!(5, graph.vertex_count());
        assert_eq!(18, graph.edge_count());
        assert_eq!(6, graph.face_count());
    }

    #[test]
    fn iterate_mesh_topology() {
        let mut graph = UvSphere::new(4, 2)
            .polygons_with_position() // 8 triangles, 24 vertices.
            .collect::<MeshGraph<Point3<f32>>>();

        assert_eq!(6, graph.vertices().count());
        assert_eq!(24, graph.edges().count());
        assert_eq!(8, graph.faces().count());
        for vertex in graph.vertices() {
            // Every vertex is connected to 4 triangles with 4 (incoming)
            // edges. Traversal of topology should be possible.
            assert_eq!(4, vertex.incoming_edges().count());
        }
        for mut vertex in graph.orphan_vertices() {
            // Geometry should be mutable.
            vertex.geometry += Vector3::zero();
        }
    }

    #[test]
    fn non_manifold_error_deferred() {
        let graph = UvSphere::new(32, 32)
            .polygons_with_position()
            .triangulate()
            .collect::<MeshGraph<Point3<f32>>>();
        // This conversion will join faces by a single vertex, but ultimately
        // creates a manifold.
        graph
            .to_mesh_buffer_by_face_with::<U3, usize, _, _>(|_, vertex| vertex.geometry)
            .unwrap();
    }

    #[test]
    fn error_on_non_manifold_mesh() {
        // Construct a mesh with a "fan" of three triangles sharing the same
        // edge along the Z-axis. The edge would have three associated faces,
        // which should not be possible.
        let graph = MeshGraph::<Point3<i32>>::from_raw_buffers(
            vec![0, 1, 2, 0, 1, 3, 0, 1, 4],
            vec![(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
            3,
        );

        assert!(match graph.err().unwrap() {
            GraphError::TopologyConflict => true,
            _ => false,
        });
    }

    #[test]
    fn error_on_singularity_mesh() {
        // Construct a mesh with three non-neighboring triangles sharing a
        // single vertex.
        let graph = MeshGraph::<Point3<i32>>::from_raw_buffers(
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

        assert!(match graph.err().unwrap() {
            GraphError::TopologyMalformed => true,
            _ => false,
        });

        // Construct a mesh with three triangles forming a rectangle, where one
        // vertex (at the origin) is shared by all three triangles.
        let graph = MeshGraph::<Point2<i32>>::from_raw_buffers(
            vec![0, 1, 3, 1, 4, 3, 1, 2, 4],
            vec![(-1, 0), (0, 0), (1, 0), (-1, 1), (1, 1)],
            3,
        )
        .unwrap();
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
        let key = graph
            .faces()
            .find(|face| {
                face.vertices()
                    .map(|vertex| vertex.geometry.clone())
                    .map(|position| (position.x, position.y))
                    .collect::<HashSet<_>>()
                    == *geometry
            })
            .unwrap()
            .key();
        let cache = FaceRemoveCache::snapshot(&graph, key).unwrap();
        let mut mutation = Mutation::mutate(graph);
        assert!(
            match mutation.remove_face_with_cache(cache).err().unwrap() {
                GraphError::TopologyConflict => true,
                _ => false,
            }
        );
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
        let mut graph = UvSphere::new(4, 4)
            .polygons_with_position()
            .collect::<MeshGraph<ValueGeometry>>();
        let value = 3.14;
        for mut face in graph.orphan_faces() {
            face.geometry = value;
        }

        // Read the geometry of each face using an immutable iterator to ensure
        // it is what we expect.
        for face in graph.faces() {
            assert_eq!(value, face.geometry);
        }
    }
}
