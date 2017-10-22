//! Half-edge graph representation of meshes.
//!
//! This module provides a flexible representation of meshes as a [half-edge
//! graph](https://en.wikipedia.org/wiki/doubly_connected_edge_list) exposed as
//! the `Mesh` type. Meshes can store arbitrary geometric data associated with
//! any topological structure, including vertices, half-edges, and faces.
//!
//! These structures can be difficult to construct from individual components;
//! the `generate` module can be used to produce primitive meshes that can be
//! converted into a graph.
//!
//! # Representation
//!
//! Meshes store topological data using keyed storage. That is, structures
//! representing vertices, edges, and faces, are stored using associative
//! collections. Keys are exposed as strongly typed and opaque values, which
//! can be used to refer to a topological structure, e.g., `VertexKey`.
//! Importantly, raw references are not used, which eases the construction and
//! manipulation of the graph in both user and library code by avoiding certain
//! borrowing rules.
//!
//! # Topological Views
//!
//! Meshes expose views over their topological structures (vertices, edges, and
//! faces). Views are accessed via keys or iteration and behave like
//! references, and are exposed as `...Ref`, `...Mut`, and `Orphan...Mut` types
//! (immutable, mutable, and orphan views, respectively) summarized below:
//!
//! | Type      | Name           | Traversal | Arity | Geometry  | Topology  |
//! |-----------|----------------|-----------|-------|-----------|-----------|
//! | Immutable | `...Ref`       | Yes       | Many  | Immutable | Immutable |
//! | Mutable   | `...Mut`       | Neighbors | One   | Mutable   | Mutable   |
//! | Orphan    | `Orphan...Mut` | No        | Many  | Mutable   | N/A       |
//!
//! Note that it is not possible to get mutable views from another mutable view
//! via a traversal, because a mutation may alter the topology and invalidate
//! the originating view. This also means that mutable operations will always
//! consume `self`. In general, an immutable traversal of topology can be used
//! to collect keys that are later used to query and mutate the target
//! topology.
//!
//! All views provide access to their associated geometry. Mutable views, like
//! `FaceMut`, provide topological operations, like triangulation and
//! extrusion.
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
//! use plexus::generate::sphere::UVSphere;
//! use plexus::graph::Mesh;
//! use plexus::prelude::*;
//!
//! # fn main() {
//! let mut mesh = UVSphere::<f32>::with_unit_radius(16, 16)
//!     .polygons_with_position()
//!     .map_vertices(|position| position.into_hash())
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
//! use plexus::generate::sphere::UVSphere;
//! use plexus::graph::Mesh;
//! use plexus::prelude::*;
//!
//! # fn main() {
//! let mut mesh = UVSphere::<f32>::with_unit_radius(16, 16)
//!     .polygons_with_position()
//!     .map_vertices(|position| position.into_hash())
//!     .collect::<Mesh<Point3<f32>>>();
//! let key = mesh.faces().nth(0).unwrap().key(); // Get the key of the first face.
//! mesh.face_mut(key).unwrap().extrude(1.0).unwrap(); // Extrude the face.
//! # }
//! ```

mod geometry;
mod mesh;
mod storage;
mod topology;

pub use self::mesh::Mesh;
pub use self::storage::{EdgeKey, FaceKey, VertexKey};
pub use self::topology::{EdgeKeyTopology, EdgeMut, EdgeRef, FaceKeyTopology, FaceMut, FaceRef,
                         OrphanEdgeMut, OrphanFaceMut, OrphanVertexMut, VertexMut, VertexRef};

trait VecExt<T>
where
    T: Copy,
{
    fn duplet_circuit_windows(&self) -> DupletCircuitWindows<T>;
}

impl<T> VecExt<T> for Vec<T>
where
    T: Copy,
{
    fn duplet_circuit_windows(&self) -> DupletCircuitWindows<T> {
        DupletCircuitWindows::new(self)
    }
}

struct DupletCircuitWindows<'a, T>
where
    T: 'a + Copy,
{
    input: &'a Vec<T>,
    index: usize,
}

impl<'a, T> DupletCircuitWindows<'a, T>
where
    T: 'a + Copy,
{
    fn new(input: &'a Vec<T>) -> Self {
        DupletCircuitWindows {
            input: input,
            index: 0,
        }
    }
}

impl<'a, T> Iterator for DupletCircuitWindows<'a, T>
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
