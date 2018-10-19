//! Half-edge graph representation of meshes.
//!
//! This module provides a flexible representation of meshes as a [half-edge
//! graph](https://en.wikipedia.org/wiki/doubly_connected_edge_list). Meshes
//! can store arbitrary geometric data associated with any topological
//! structure, including vertices, half-edges, and faces.
//!
//! # Representation
//!
//! `MeshGraph`s store topological data using associative collections. Keys are
//! exposed as strongly typed and opaque values, which can be used to refer to
//! a topological structure, such as `VertexKey`. Topology is typically
//! manipulated using a view.
//!
//! A `MeshGraph` is conceptually composed of vertices, half-edges, and faces.
//! Half-edges are directed and join vertices. Every half-edge is paired with
//! an opposite half-edge with the opposite direction.  Given a half-edge that
//! connects a vertex `a` to a vertex `b`, that half-edge will have an opposite
//! half-edge from `b` to `a`. Together, these half-edges form a composite
//! edge. When the term "edge" is used alone, it generally refers to a
//! half-edge.
//!
//! Half-edges are connected to their neighbors, known as "next" and "previous"
//! half-edges. When a face is present in the region formed by a perimeter of
//! vertices and their half-edges, the half-edges will refer to that face and
//! the face will refer to one of the half-edges in the interior circuit.
//!
//! Together with vertices and faces, the connectivity of half-edges allows for
//! effecient traversal of topology. For example, it becomes trivial to find
//! neighboring topologies, such as the faces that share a given vertex or the
//! neighboring faces of a given face.
//!
//! # Topological Views
//!
//! `MeshGraph`s expose views over their topological structures (vertices,
//! edges, and faces). Views are accessed via keys or iteration and behave
//! similarly to references. They provide the primary API for interacting with
//! a mesh's topology and geometry. There are three types summarized below:
//!
//! | Type      | Traversal | Exclusive | Geometry  | Topology  |
//! |-----------|-----------|-----------|-----------|-----------|
//! | Immutable | Yes       | No        | Immutable | Immutable |
//! | Mutable   | Yes       | Yes       | Mutable   | Mutable   |
//! | Orphan    | No        | No        | Mutable   | N/A       |
//!
//! Immutable and mutable views are much like references. Immutable views
//! cannot mutate a mesh in any way and it is possible to obtain multiple such
//! views at the same time. Mutable views are exclusive, but allow for
//! mutations.
//!
//! Orphan views are similar to mutable views, but they only have access to the
//! geometry of a specific topological structure in a mesh. Because they do not
//! know about other vertices, edges, or faces, an orphan view cannot traverse
//! the topology of a mesh in any way. These views are most useful for
//! modifying the geometry of a mesh and, unlike mutable views, multiple orphan
//! views can be obtained at the same time. Orphan views are mostly used by
//! circulators (iterators).
//!
//! Immutable and mutable views are represented by a single type constructor,
//! such as `FaceView`.  Orphan views are represented by their own type, such
//! as `OrphanFace`.
//!
//! # Circulators
//!
//! Topological views allow for traversals of a mesh's topology. One useful
//! type of traversal uses a circulator, which is a type of iterator that
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
//! Generating a mesh from a primitive:
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
//! Manipulating a face in a mesh:
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
//! let mut graph = MeshGraph::<Point2<f32>>::from_raw_buffers(
//!     vec![0, 1, 2, 3],
//!     vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
//!     4,
//! ).unwrap();
//! graph.triangulate().unwrap();
//!
//! // Traverse an edge and use a circulator to get the faces of a nearby vertex.
//! let key = graph.edges().nth(0).unwrap().key();
//! let mut vertex = graph
//!     .edge_mut(key)
//!     .unwrap()
//!     .into_opposite_edge()
//!     .into_next_edge()
//!     .into_destination_vertex();
//! for face in vertex.neighboring_orphan_faces() {
//!     // `face.geometry` is mutable here.
//! }
//! # }
//! ```

mod container;
mod geometry;
mod mesh;
mod mutation;
mod storage;
mod topology;
mod view;

use failure::Error;

pub use self::mesh::MeshGraph;
pub use self::storage::{EdgeKey, FaceKey, VertexKey};
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
    EdgeKeyTopology, EdgeView, FaceKeyTopology, FaceView, OrphanEdgeView, OrphanFaceView,
    OrphanVertexView, VertexView,
};

#[derive(Debug, Fail)]
pub enum GraphError {
    #[fail(display = "required topology not found")]
    TopologyNotFound,
    #[fail(display = "conflicting topology found")]
    TopologyConflict,
    #[fail(display = "topology malformed")]
    TopologyMalformed,
    #[fail(
        display = "conflicting arity; expected {}, but got {}",
        expected,
        actual
    )]
    ArityConflict { expected: usize, actual: usize },
    #[fail(display = "face arity is non-constant")]
    ArityNonConstant,
}

// TODO: Using `Error` so broadly is a misuse case. Look at how complex
//       `ResultExt` is! Just use `GraphError` directly. Never use `Error`
//       along "happy" or "expected" paths (this happens often in the
//       `mutation` module.

trait ResultExt<T>: Sized {
    /// If the `Result` is an error with the value
    /// `GraphError::TopologyConflict`, then the given function is executed and
    /// its `Result` is returned. Otherwise, the original `Result` is returned.
    ///
    /// Because this operates on the result of an operation, it does not cancel
    /// or negate any mutations that may have occurred.
    fn or_if_conflict<F>(self, f: F) -> Result<T, Error>
    where
        F: FnOnce() -> Result<T, Error>;
}

impl<T> ResultExt<T> for Result<T, Error> {
    fn or_if_conflict<F>(self, f: F) -> Result<T, Error>
    where
        F: FnOnce() -> Result<T, Error>,
    {
        self.map_err(|error| error.downcast::<GraphError>().unwrap())
            .or_if_conflict(f)
    }
}

impl<T> ResultExt<T> for Result<T, GraphError> {
    fn or_if_conflict<F>(self, f: F) -> Result<T, Error>
    where
        F: FnOnce() -> Result<T, Error>,
    {
        self.or_else(|error| match error {
            GraphError::TopologyConflict => f(),
            error => Err(error.into()),
        })
    }
}

trait IteratorExt: Iterator + Sized {
    /// Provides an iterator over a window of duplets that includes the first
    /// value in the sequence at the beginning and end of the iteration.
    ///
    /// Given a collection with ordered elements `a`, `b`, and `c`, this
    /// iterator yeilds the ordered items `(a, b)`, `(b, c)`, `(c, a)`.
    fn perimeter(self) -> Perimeter<Self>
    where
        Self::Item: Copy;
}

impl<I> IteratorExt for I
where
    I: Iterator,
{
    fn perimeter(self) -> Perimeter<I>
    where
        I::Item: Copy,
    {
        Perimeter::new(self)
    }
}

struct Perimeter<I>
where
    I: Iterator,
    I::Item: Copy,
{
    input: I,
    first: Option<I::Item>,
    previous: Option<I::Item>,
}

impl<I> Perimeter<I>
where
    I: Iterator,
    I::Item: Copy,
{
    fn new(mut input: I) -> Self {
        let first = input.next();
        let previous = first;
        Perimeter {
            input,
            first,
            previous,
        }
    }
}

impl<I> Iterator for Perimeter<I>
where
    I: Iterator,
    I::Item: Copy,
{
    type Item = (I::Item, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.input.next();
        match (self.previous, next.or_else(|| self.first.take())) {
            (Some(a), Some(b)) => {
                self.previous = Some(b);
                Some((a, b))
            }
            _ => None,
        }
    }
}
