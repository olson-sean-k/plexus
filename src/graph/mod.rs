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
mod mesh;
mod mutation;
mod storage;
mod topology;
mod view;

use std::fmt::Debug;

use crate::buffer::BufferError;

pub use self::mesh::MeshGraph;
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
    ArcNeighborhood, ArcView, FaceNeighborhood, FaceView, InteriorPathView, OrphanArcView,
    OrphanFaceView, OrphanVertexView, VertexView,
};

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
