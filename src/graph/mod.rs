//! Half-edge graph representation of meshes.
//!
//! This module provides a flexible representation of meshes as a [half-edge
//! graph](https://en.wikipedia.org/wiki/doubly_connected_edge_list). Meshes
//! can store arbitrary geometric data associated with any topological
//! structure, including vertices, (bilateral) edges, and faces.
//!
//! These structures can be difficult to construct from individual components;
//! the `generate` module can be used to produce primitive meshes that can be
//! converted into a graph.
//!
//! # Examples
//!
//! Creating an empty mesh with no geometric data:
//!
//! ```rust
//! use plexus::graph::Mesh;
//!
//! let mut mesh = Mesh::<()>::new();
//! ```

mod geometry;
mod mesh;
mod storage;
mod topology;

pub use self::geometry::{AsPosition, Attribute, Cross, FromGeometry, FromInteriorGeometry,
                         Geometry, IntoGeometry, IntoInteriorGeometry, Normalize};
pub use self::mesh::Mesh;
pub use self::storage::{EdgeKey, FaceKey, VertexKey};
pub use self::topology::{EdgeMut, EdgeRef, FaceMut, FaceRef, OrphanEdgeMut, OrphanFaceMut,
                         VertexMut, VertexRef};

pub mod prelude {
    pub use super::{FromGeometry, FromInteriorGeometry, IntoGeometry, IntoInteriorGeometry};
}
