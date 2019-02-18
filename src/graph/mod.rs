//! Half-edge graph representation of meshes.
//!
//! This module provides a flexible representation of meshes as a [half-edge
//! graph](https://en.wikipedia.org/wiki/doubly_connected_edge_list).
//! _Half-edges_ and _edges_ are referred to as _arcs_ and _edges_,
//! respectively.  Meshes can store arbitrary geometric data associated with
//! any topological structure (vertices, arcs, edges, and faces).
//!
//! Geometry is vertex-based, meaning that geometric operations depend on
//! vertices exposing some notion of positional data. See the `geometry` module
//! and `AsPosition` trait. If geometry does not have this property, then most
//! spatial operations will not be available.
//!
//! # Representation
//!
//! A `MeshGraph` is conceptually composed of _vertices_, _arcs_, _edges_, and
//! _faces_. The figure below summarizes the connectivity in a `MeshGraph`.
//!
//! ![Half-Edge Graph Figure](https://raw.githubusercontent.com/olson-sean-k/plexus/master/doc/heg.svg?sanitize=true)
//!
//! Arcs are directed and connect vertices. An arc that is directed toward a
//! vertex **A** is an _incoming arc_ with respect to **A**.  Similarly, an arc
//! directed away from such a vertex is an _outgoing arc_. Every vertex is
//! associated with exactly one _leading arc_, which is always an outgoing arc.
//! The vertex toward which an arc is directed is the arc's _destination
//! vertex_ and the other is its _source vertex_.
//!
//! Every arc is paired with an _opposite arc_ with an opposing direction.
//! Given an arc from a vertex **A** to a vertex **B**, that arc will have an
//! opposite arc from **B** to **A**. Such arcs are typically labeled **AB**
//! and **BA**. Together, these arcs form an _edge_, which is not directed.
//! Occassionally, the term "edge" may refer to either an arc or an edge.
//!
//! Arcs are connected to their neighbors, known as _next_ and _previous arcs_.
//! When a face is present in the contiguous region formed by a perimeter of
//! vertices and their arcs, the arcs will refer to that face and the face will
//! refer to exactly one of the arcs in the interior. An arc with no associated
//! face is known as a _boundary arc_. If both of an edge's arcs are boundary
//! arcs, then that edge is a _disjoint edge_.
//!
//! Together with vertices and faces, the connectivity of arcs allows for
//! effecient traversals of topology. For example, it becomes trivial to find
//! neighboring topologies, such as the faces that share a given vertex or the
//! neighboring faces of a given face.
//!
//! `MeshGraph`s store topological data using associative collections and mesh
//! data is accessed using keys into this storage. Keys are exposed as strongly
//! typed and opaque values, which can be used to refer to a topological
//! structure, such as `VertexKey`. Topology is typically manipulated using a
//! _view_, such as `VertexView` (see below).
//!
//! # Topological Views
//!
//! `MeshGraph`s expose _views_ over their topological structures (vertices,
//! arcs, edges, and faces). Views are accessed via keys or iteration and
//! behave similarly to references. They provide the primary API for
//! interacting with a `MeshGraph`'s topology and geometry. There are three
//! types summarized below:
//!
//! | Type      | Traversal | Exclusive | Geometry  | Topology  |
//! |-----------|-----------|-----------|-----------|-----------|
//! | Immutable | Yes       | No        | Immutable | Immutable |
//! | Mutable   | Yes       | Yes       | Mutable   | Mutable   |
//! | Orphan    | No        | No        | Mutable   | N/A       |
//!
//! _Immutable_ and _mutable views_ behave similarly to references. Immutable
//! views cannot mutate a mesh in any way and it is possible to obtain multiple
//! such views at the same time. Mutable views are exclusive, but allow for
//! mutations.
//!
//! _Orphan views_ are similar to mutable views, but they only have access to
//! the geometry of a single topological structure in a mesh. Because they do
//! not know about other vertices, arcs, etc., an orphan view cannot traverse
//! the topology of a mesh in any way. These views are most useful for
//! modifying the geometry of a mesh and, unlike mutable views, multiple orphan
//! views can be obtained at the same time. Orphan views are mostly used by
//! mutable _circulators_ (iterators).
//!
//! Immutable and mutable views are both represented by view types, such as
//! `FaceView`. Orphan views are represented by an oprhan view type, such as
//! `OrphanFaceView`.
//!
//! # Circulators
//!
//! Topological views allow for traversals of a mesh's topology. One useful
//! type of traversal uses a _circulator_, which is a type of iterator that
//! examines the neighbors of a topological structure. For example, the face
//! circulator of a vertex yields all faces that share that vertex in order.
//!
//! Mutable circulators emit orphan views, not mutable views. This is because
//! it is not possible to instantiate more than one mutable view at a time. If
//! multiple mutable views are needed, it is possible to use an immutable
//! circulator to collect the keys of the target topology and then lookup each
//! mutable view using those keys.
//!
//! # Examples
//!
//! Generating a mesh from a UV-sphere:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point3;
//! use plexus::graph::MeshGraph;
//! use plexus::prelude::*;
//! use plexus::primitive::sphere::UvSphere;
//!
//! # fn main() {
//! let mut graph = UvSphere::new(16, 16)
//!     .polygons_with_position()
//!     .collect::<MeshGraph<Point3<f32>>>();
//! # }
//! ```
//!
//! Extruding a face in a mesh:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point3;
//! use plexus::graph::MeshGraph;
//! use plexus::prelude::*;
//! use plexus::primitive::sphere::UvSphere;
//!
//! # fn main() {
//! let mut graph = UvSphere::new(16, 16)
//!     .polygons_with_position()
//!     .collect::<MeshGraph<Point3<f32>>>();
//! let key = graph.faces().nth(0).unwrap().key(); // Get the key of the first face.
//! graph.face_mut(key).unwrap().extrude(1.0).unwrap(); // Extrude the face.
//! # }
//! ```
//!
//! Traversing and circulating over a mesh:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point2;
//! use plexus::graph::MeshGraph;
//! use plexus::prelude::*;
//!
//! # fn main() {
//! let mut graph = MeshGraph::<Point2<f32>>::from_raw_buffers_with_arity(
//!     vec![0u32, 1, 2, 3],
//!     vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
//!     4,
//! )
//! .unwrap();
//! graph.triangulate().unwrap();
//!
//! // Traverse an arc and use a circulator to get the faces of a nearby vertex.
//! let key = graph.arcs().nth(0).unwrap().key();
//! let mut vertex = graph
//!     .arc_mut(key)
//!     .unwrap()
//!     .into_opposite_arc()
//!     .into_next_arc()
//!     .into_destination_vertex();
//! for mut face in vertex.neighboring_orphan_faces() {
//!     // `face.geometry` is mutable here.
//! }
//! # }
//! ```

mod container;
// The `graph::geometry` module uses private members of its parent module. It
// is implemented here and re-exported in the `geometry::compose` module.
pub(in crate) mod geometry;
mod mutation;
mod payload;
mod storage;
mod view;

pub use self::payload::{ArcPayload, EdgePayload, FacePayload, VertexPayload};
pub use self::storage::{ArcKey, EdgeKey, FaceKey, VertexKey};
// TODO: It's unclear how view types should be exposed to users. Type aliases
//       for mutable, immutable, and orphan views over a `MeshGraph` would be
//       simpler and help insulate users from the complexity of views, but it
//       is currently not possible to document such aliases. See:
//       https://github.com/rust-lang/rust/issues/39437
//
//       Moreover, in the future it may be tenable to expose the internal
//       mutation APIs, and exposing the underlying view types would then be
//       necessary. For now, use them directly.
pub use self::view::{
    ArcNeighborhood, ArcView, EdgeView, FaceNeighborhood, FaceView, InteriorPathView,
    OrphanArcView, OrphanEdgeView, OrphanFaceView, OrphanVertexView, VertexView,
};

use arrayvec::ArrayVec;
use decorum::R64;
use itertools::Itertools;
use num::{Integer, NumCast, ToPrimitive, Unsigned};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::FromIterator;
use typenum::{self, NonZero};

use crate::buffer::{BufferError, Flat, IndexBuffer, MeshBuffer};
use crate::geometry::convert::{FromGeometry, FromInteriorGeometry, IntoGeometry};
use crate::geometry::{Geometry, Triplet};
use crate::graph::container::alias::OwnedCore;
use crate::graph::container::{Bind, Consistent, Core};
use crate::graph::mutation::{Mutate, Mutation};
use crate::graph::storage::alias::*;
use crate::graph::storage::convert::alias::*;
use crate::graph::storage::convert::{AsStorage, AsStorageMut};
use crate::graph::storage::{OpaqueKey, Storage};
use crate::graph::view::convert::IntoView;
use crate::primitive::decompose::IntoVertices;
use crate::primitive::index::{FromIndexer, HashIndexer, IndexVertices, Indexer};
use crate::primitive::{self, Arity, Map, Polygonal, Quad};
use crate::{FromRawBuffers, FromRawBuffersWithArity};

pub use Selector::ByIndex;
pub use Selector::ByKey;

#[derive(Debug, Fail, PartialEq)]
pub enum GraphError {
    #[fail(display = "required topology not found")]
    TopologyNotFound,
    #[fail(display = "conflicting topology found")]
    TopologyConflict,
    #[fail(display = "topology malformed")]
    TopologyMalformed,
    #[fail(
        display = "conflicting arity; expected {}, but got {}",
        expected, actual
    )]
    ArityConflict { expected: usize, actual: usize },
    #[fail(display = "face arity is non-constant")]
    ArityNonConstant,
}

impl From<BufferError> for GraphError {
    fn from(_: BufferError) -> Self {
        // TODO: How should buffer errors be handled? Is this sufficient?
        GraphError::TopologyMalformed
    }
}

trait OptionExt<T> {
    fn expect_consistent(self) -> T;
}

impl<T> OptionExt<T> for Option<T> {
    fn expect_consistent(self) -> T {
        self.expect("graph consistency violated")
    }
}

trait ResultExt<T, E> {
    fn expect_consistent(self) -> T
    where
        E: Debug;
}

impl<T, E> ResultExt<T, E> for Result<T, E> {
    fn expect_consistent(self) -> T
    where
        E: Debug,
    {
        self.expect("graph consistency violated")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Selector<K> {
    ByKey(K),
    ByIndex(usize),
}

impl<K> Selector<K> {
    pub fn key_or_else<E, F>(self, f: F) -> Result<K, GraphError>
    where
        E: Into<GraphError>,
        F: Fn(usize) -> Result<K, E>,
    {
        match self {
            Selector::ByKey(key) => Ok(key),
            Selector::ByIndex(index) => f(index).map_err(|error| error.into()),
        }
    }

    pub fn index_or_else<E, F>(self, f: F) -> Result<usize, GraphError>
    where
        E: Into<GraphError>,
        F: Fn(K) -> Result<usize, E>,
    {
        match self {
            Selector::ByKey(key) => f(key).map_err(|error| error.into()),
            Selector::ByIndex(index) => Ok(index),
        }
    }
}

impl<K> From<K> for Selector<K>
where
    K: OpaqueKey,
{
    fn from(key: K) -> Self {
        Selector::ByKey(key)
    }
}

impl<K> From<usize> for Selector<K> {
    fn from(index: usize) -> Self {
        Selector::ByIndex(index)
    }
}

/// Half-edge graph representation of a mesh.
///
/// Provides topological data in the form of vertices, arcs, edges, and faces.
/// An arc is directed from one vertex to another, with an opposing arc joining
/// the vertices in the other direction.
///
/// `MeshGraph`s expose topological views, which can be used to traverse and
/// manipulate topology and geometry in the graph.
///
/// See the module documentation for more details.
pub struct MeshGraph<G = Triplet<R64>>
where
    G: Geometry,
{
    core: OwnedCore<G>,
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
                .bind(VertexStorage::<G>::new())
                .bind(ArcStorage::<G>::new())
                .bind(EdgeStorage::<G>::new())
                .bind(FaceStorage::<G>::new()),
        )
    }

    /// Creates an empty `MeshGraph`.
    ///
    /// Underlying storage has zero capacity and does not allocate until the
    /// first insertion.
    pub fn empty() -> Self {
        MeshGraph::from(
            Core::empty()
                .bind(VertexStorage::<G>::empty())
                .bind(ArcStorage::<G>::empty())
                .bind(EdgeStorage::<G>::empty())
                .bind(FaceStorage::<G>::empty()),
        )
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
    ///     vec![0u64, 1, 2, 3],
    ///     vec![(0.0f64, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    /// )
    /// .unwrap();
    /// let mut graph = MeshGraph::<Point2<f64>>::from_mesh_buffer(buffer).unwrap();
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
        MeshGraph::from_raw_buffers_with_arity(indices, vertices, arity)
    }

    /// Gets the number of vertices in the mesh.
    pub fn vertex_count(&self) -> usize {
        self.as_vertex_storage().len()
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
    pub fn vertices(&self) -> impl Clone + Iterator<Item = VertexView<&Self, G>> {
        self.as_vertex_storage()
            .keys()
            .map(move |key| (*key, self).into_view().unwrap())
    }

    /// Gets an iterator of orphan views over the vertices in the mesh.
    ///
    /// Because this only yields orphan views, only geometry can be mutated.
    /// For topological mutations, collect the necessary keys and use
    /// `vertex_mut` instead.
    pub fn orphan_vertices(&mut self) -> impl Iterator<Item = OrphanVertexView<G>> {
        self.as_vertex_storage_mut()
            .iter_mut()
            .map(|(key, source)| (*key, source).into_view().unwrap())
    }

    /// Gets the number of arcs in the mesh.
    pub fn arc_count(&self) -> usize {
        self.as_arc_storage().len()
    }

    /// Gets an immutable view of the arc with the given key.
    pub fn arc(&self, key: ArcKey) -> Option<ArcView<&Self, G>> {
        (key, self).into_view()
    }

    /// Gets a mutable view of the arc with the given key.
    pub fn arc_mut(&mut self, key: ArcKey) -> Option<ArcView<&mut Self, G>> {
        (key, self).into_view()
    }

    /// Gets an iterator of immutable views over the arcs in the mesh.
    pub fn arcs(&self) -> impl Clone + Iterator<Item = ArcView<&Self, G>> {
        self.as_arc_storage()
            .keys()
            .map(move |key| (*key, self).into_view().unwrap())
    }

    /// Gets an iterator of orphan views over the arcs in the mesh.
    ///
    /// Because this only yields orphan views, only geometry can be mutated.
    /// For topological mutations, collect the necessary keys and use
    /// `arc_mut` instead.
    pub fn orphan_arcs(&mut self) -> impl Iterator<Item = OrphanArcView<G>> {
        self.as_arc_storage_mut()
            .iter_mut()
            .map(|(key, source)| (*key, source).into_view().unwrap())
    }

    /// Gets the number of edges in the mesh.
    pub fn edge_count(&self) -> usize {
        self.as_edge_storage().len()
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
    pub fn edges(&self) -> impl Clone + Iterator<Item = EdgeView<&Self, G>> {
        self.as_edge_storage()
            .keys()
            .map(move |key| (*key, self).into_view().unwrap())
    }

    /// Gets an iterator of orphan views over the edges in the mesh.
    ///
    /// Because this only yields orphan views, only geometry can be mutated.
    /// For topological mutations, collect the necessary keys and use
    /// `edge_mut` instead.
    pub fn orphan_edges(&mut self) -> impl Iterator<Item = OrphanEdgeView<G>> {
        self.as_edge_storage_mut()
            .iter_mut()
            .map(|(key, source)| (*key, source).into_view().unwrap())
    }

    /// Gets the number of faces in the mesh.
    pub fn face_count(&self) -> usize {
        self.as_face_storage().len()
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
    pub fn faces(&self) -> impl Clone + Iterator<Item = FaceView<&Self, G>> {
        self.as_face_storage()
            .keys()
            .map(move |key| (*key, self).into_view().unwrap())
    }

    /// Gets an iterator of orphan views over the faces in the mesh.
    ///
    /// Because this only yields orphan views, only geometry can be mutated.
    /// For topological mutations, collect the necessary keys and use
    /// `face_mut` instead.
    pub fn orphan_faces(&mut self) -> impl Iterator<Item = OrphanFaceView<G>> {
        self.as_face_storage_mut()
            .iter_mut()
            .map(|(key, source)| (*key, source).into_view().unwrap())
    }

    /// Triangulates the mesh, tesselating all faces into triangles.
    pub fn triangulate(&mut self) -> Result<(), GraphError> {
        let faces = self.as_face_storage().keys().cloned().collect::<Vec<_>>();
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

impl<G> AsStorage<VertexPayload<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<VertexPayload<G>> {
        self.core.as_vertex_storage()
    }
}

impl<G> AsStorage<ArcPayload<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<ArcPayload<G>> {
        self.core.as_arc_storage()
    }
}

impl<G> AsStorage<EdgePayload<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<EdgePayload<G>> {
        self.core.as_edge_storage()
    }
}

impl<G> AsStorage<FacePayload<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<FacePayload<G>> {
        self.core.as_face_storage()
    }
}

impl<G> AsStorageMut<VertexPayload<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<VertexPayload<G>> {
        self.core.as_vertex_storage_mut()
    }
}

impl<G> AsStorageMut<ArcPayload<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<ArcPayload<G>> {
        self.core.as_arc_storage_mut()
    }
}

impl<G> AsStorageMut<EdgePayload<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<EdgePayload<G>> {
        self.core.as_edge_storage_mut()
    }
}

impl<G> AsStorageMut<FacePayload<G>> for MeshGraph<G>
where
    G: Geometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<FacePayload<G>> {
        self.core.as_face_storage_mut()
    }
}

impl<G> Consistent for MeshGraph<G> where G: Geometry {}

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

impl<G, H> FromInteriorGeometry<MeshGraph<H>> for MeshGraph<G>
where
    G: Geometry,
    G::Vertex: FromGeometry<H::Vertex>,
    G::Arc: FromGeometry<H::Arc>,
    G::Edge: FromGeometry<H::Edge>,
    G::Face: FromGeometry<H::Face>,
    H: Geometry,
{
    fn from_interior_geometry(graph: MeshGraph<H>) -> Self {
        let MeshGraph { core, .. } = graph;
        let (vertices, arcs, edges, faces) = core.into_storage();
        let core = Core::empty()
            .bind(
                vertices
                    .map_values_into(|vertex| VertexPayload::<G>::from_interior_geometry(vertex)),
            )
            .bind(arcs.map_values_into(|arc| ArcPayload::<G>::from_interior_geometry(arc)))
            .bind(edges.map_values_into(|edge| EdgePayload::<G>::from_interior_geometry(edge)))
            .bind(faces.map_values_into(|face| FacePayload::<G>::from_interior_geometry(face)));
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

impl<P, G, H> FromRawBuffers<P, H> for MeshGraph<G>
where
    P: IntoVertices + Polygonal,
    P::Vertex: Integer + ToPrimitive + Unsigned,
    G: Geometry,
    H: IntoGeometry<G::Vertex>,
{
    type Error = GraphError;

    fn from_raw_buffers<I, J>(indices: I, vertices: J) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = P>,
        J: IntoIterator<Item = H>,
    {
        let mut mutation = Mutation::mutate(MeshGraph::new());
        let vertices = vertices
            .into_iter()
            .map(|vertex| mutation.insert_vertex(vertex.into_geometry()))
            .collect::<Vec<_>>();
        for face in indices {
            let mut perimeter = SmallVec::<[_; 4]>::with_capacity(face.arity());
            for index in face.into_vertices() {
                let index = <usize as NumCast>::from(index).unwrap();
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
}

impl<N, G, H> FromRawBuffersWithArity<N, H> for MeshGraph<G>
where
    N: Integer + ToPrimitive + Unsigned,
    G: Geometry,
    H: IntoGeometry<G::Vertex>,
{
    type Error = GraphError;

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
    /// let mut graph =
    ///     MeshGraph::<Point3<f64>>::from_raw_buffers_with_arity(indices, positions, 3).unwrap();
    /// # }
    /// ```
    fn from_raw_buffers_with_arity<I, J>(
        indices: I,
        vertices: J,
        arity: usize,
    ) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = N>,
        J: IntoIterator<Item = H>,
    {
        let mut mutation = Mutation::mutate(MeshGraph::new());
        let vertices = vertices
            .into_iter()
            .map(|vertex| mutation.insert_vertex(vertex.into_geometry()))
            .collect::<Vec<_>>();
        for face in &indices
            .into_iter()
            .map(|index| <usize as NumCast>::from(index).unwrap())
            .chunks(arity)
        {
            let face = face.collect::<Vec<_>>();
            if face.len() != arity {
                // Index buffer length is not a multiple of arity.
                return Err(GraphError::ArityConflict {
                    expected: arity,
                    actual: face.len(),
                });
            }
            let mut perimeter = SmallVec::<[_; 4]>::with_capacity(arity);
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

#[cfg(test)]
mod tests {
    use nalgebra::{Point3, Vector3};
    use num::Zero;

    use crate::buffer::U3;
    use crate::geometry::*;
    use crate::graph::*;
    use crate::primitive::decompose::*;
    use crate::primitive::generate::*;
    use crate::primitive::sphere::UvSphere;
    use crate::*;

    #[test]
    fn collect_topology_into_mesh() {
        let graph = UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f32>>>();

        assert_eq!(5, graph.vertex_count());
        assert_eq!(18, graph.arc_count());
        assert_eq!(6, graph.face_count());
    }

    #[test]
    fn iterate_mesh_topology() {
        let mut graph = UvSphere::new(4, 2)
            .polygons_with_position() // 8 triangles, 24 vertices.
            .collect::<MeshGraph<Point3<f32>>>();

        assert_eq!(6, graph.vertices().count());
        assert_eq!(24, graph.arcs().count());
        assert_eq!(8, graph.faces().count());
        for vertex in graph.vertices() {
            // Every vertex is connected to 4 triangles with 4 (incoming) arcs.
            // Traversal of topology should be possible.
            assert_eq!(4, vertex.incoming_arcs().count());
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
        // arc along the Z-axis. The edge would have three associated faces,
        // which should not be possible.
        let graph = MeshGraph::<Point3<i32>>::from_raw_buffers_with_arity(
            vec![0u32, 1, 2, 0, 1, 3, 0, 1, 4],
            vec![(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
            3,
        );

        assert_eq!(graph.err().unwrap(), GraphError::TopologyConflict);
    }

    // This test is a sanity check for mesh iterators, topological views, and
    // the unsafe transmutations used to coerce lifetimes.
    #[test]
    fn read_write_geometry_ref() {
        impl Attribute for f32 {}

        struct ValueGeometry;

        impl Geometry for ValueGeometry {
            type Vertex = Point3<f32>;
            type Arc = ();
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
