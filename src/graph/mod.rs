//! Half-edge graph representation of meshes.
//!
//! This module provides a flexible representation of meshes as a [half-edge
//! graph](https://en.wikipedia.org/wiki/doubly_connected_edge_list). Meshes
//! can store arbitrary geometric data associated with any topological
//! structure, including vertices, half-edges, and faces.
//!
//! # Representation
//!
//! `Mesh`es store topological data using associative collections. Keys are
//! exposed as strongly typed and opaque values, which can be used to refer to
//! a topological structure, e.g., `VertexKey`. Topology is typically
//! manipulated using a view.
//!
//! A `Mesh` is conceptually composed of vertices, half-edges, and faces.
//! Half-edges are directed and join vertices. In a consistent mesh, every
//! half-edge is paired with an opposite half-edge with the opposite direction.
//! Given a half-edge that connects a vertex `a` to a vertex `b`, that
//! half-edge will have an opposite half-edge from `b` to `a`. Together, these
//! half-edges form a composite edge. When the term "edge" is used alone, it
//! generally refers to a half-edge.
//!
//! Half-edges are connected to their interior neighbors (a "next" and
//! "previous" half-edge). When a face is present in the region formed by a
//! perimeter of vertices and their half-edges, the half-edges will refer to
//! that face and the face will refer to one of the half-edges in the circuit.
//!
//! Together with vertices and faces, the connectivity of half-edges allows for
//! effecient traversal of topology. For example, it becomes trivial to find
//! neighboring topologies, such as the faces that share a given vertex or the
//! neighboring faces of a given face.
//!
//! # Topological Views
//!
//! Meshes expose views over their topological structures (vertices, edges, and
//! faces). Views are accessed via keys or iteration and behave similarly to
//! references. They provide the primary API for interacting with a mesh's
//! topology and geometry. There are three types summarized below:
//!
//! | Type      | Name           | Traversal | Exclusive | Geometry  | Topology  |
//! |-----------|----------------|-----------|-----------|-----------|-----------|
//! | Immutable | `...Ref`       | Yes       | No        | Immutable | Immutable |
//! | Mutable   | `...Mut`       | Yes       | Yes       | Mutable   | Mutable   |
//! | Orphan    | `Orphan...Mut` | No        | No        | Mutable   | N/A       |
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
//! views can be obtained at the same time.
//!
//! # Circulators
//!
//! Topological views allow for traversals of a mesh's topology. One useful
//! type of traversal uses a circulator, which is a type of iterator that
//! examines the neighbors of a topological structure. For example, the face
//! circulator of a vertex yields all faces that share that vertex in order.
//!
//! # Examples
//!
//! Generating a mesh from a primitive:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point3;
//! use plexus::generate::sphere::UvSphere;
//! use plexus::graph::Mesh;
//! use plexus::prelude::*;
//!
//! # fn main() {
//! let mut mesh = UvSphere::new(16, 16)
//!     .polygons_with_position()
//!     .collect::<Mesh<Point3<f32>>>();
//! # }
//! ```
//!
//! Manipulating a face in a mesh:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point3;
//! use plexus::generate::sphere::UvSphere;
//! use plexus::graph::Mesh;
//! use plexus::prelude::*;
//!
//! # fn main() {
//! let mut mesh = UvSphere::new(16, 16)
//!     .polygons_with_position()
//!     .collect::<Mesh<Point3<f32>>>();
//! let key = mesh.faces().nth(0).unwrap().key(); // Get the key of the first face.
//! mesh.face_mut(key).unwrap().extrude(1.0).unwrap(); // Extrude the face.
//! # }
//! ```
//!
//! Traversing and circulating over a mesh:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point2;
//! use plexus::graph::Mesh;
//! use plexus::prelude::*;
//!
//! # fn main() {
//! let mut mesh = Mesh::<Point2<f32>>::from_raw_buffers(
//!     vec![0, 1, 2, 3],
//!     vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
//!     4,
//! ).unwrap();
//! mesh.triangulate().unwrap();
//!
//! // Traverse an edge and use a circulator to get the faces of a nearby vertex.
//! let key = mesh.edges().nth(0).unwrap().key();
//! let mut vertex = mesh.edge_mut(key)
//!     .unwrap()
//!     .into_opposite_edge()
//!     .into_next_edge()
//!     .into_destination_vertex();
//! for face in vertex.neighboring_orphan_faces() {
//!     // ...
//! }
//! # }
//! ```

mod geometry;
mod mesh;
mod mutation;
mod storage;
mod topology;
mod view;

use failure::Error;

pub use self::mesh::Mesh;
pub use self::storage::{EdgeKey, FaceKey, VertexKey};
pub use self::view::{
    EdgeKeyTopology, EdgeMut, EdgeRef, FaceKeyTopology, FaceMut, FaceRef, OrphanEdge, OrphanFace,
    OrphanVertex, VertexMut, VertexRef,
};

// TODO: Do not re-export these types. This is only done so that they show up
//       in documentation. Client code should not interact with these types.
//       See: https://github.com/rust-lang/rust/issues/39437
pub use self::view::{
    EdgeView, FaceView, OrphanEdgeView, OrphanFaceView, OrphanVertexView, VertexView,
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

pub trait ResultExt<T>: Sized {
    fn or_if_conflict<F>(self, f: F) -> Result<T, Error>
    where
        F: FnOnce() -> Result<T, Error>;
}

impl<T> ResultExt<T> for Result<T, Error> {
    fn or_if_conflict<F>(self, f: F) -> Result<T, Error>
    where
        F: FnOnce() -> Result<T, Error>,
    {
        self.or_else(|error| match error.downcast::<GraphError>().unwrap() {
            GraphError::TopologyConflict => f(),
            error => Err(error.into()),
        })
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

/// Provides an iterator over a window of duplets that includes the first value
/// in the sequence at the beginning and end of the iteration.
trait Perimeter<'a, T, U>
where
    T: 'a + AsRef<[U]>,
    U: Copy,
{
    fn perimeter(&self) -> PerimeterIter<U>;
}

impl<'a, T, U> Perimeter<'a, T, U> for T
where
    T: 'a + AsRef<[U]>,
    U: Copy,
{
    fn perimeter(&self) -> PerimeterIter<U> {
        PerimeterIter::new(self.as_ref())
    }
}

struct PerimeterIter<'a, T>
where
    T: 'a + Copy,
{
    input: &'a [T],
    index: usize,
}

impl<'a, T> PerimeterIter<'a, T>
where
    T: 'a + Copy,
{
    fn new(input: &'a [T]) -> Self {
        PerimeterIter {
            input: input,
            index: 0,
        }
    }
}

impl<'a, T> Iterator for PerimeterIter<'a, T>
where
    T: 'a + Copy,
{
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index;
        let n = self.input.len();
        if index >= n {
            None
        }
        else {
            self.index += 1;
            Some((self.input[index], self.input[(index + 1) % n]))
        }
    }
}
