//! Primitive topological structures.
//!
//! This module provides unit primitives that can be used to form complex
//! iterator expressions to generate meshes via a stream of topology and
//! geometry. This data can be collected into simple buffers for rendering or a
//! graph (half-edge) for further manipulation.
//!
//! Iterator expressions begin with a unit primitive and manipulate its
//! components like vertices, lines, and polygons. Generation and decomposition
//! operations are exposed via traits.
//!
//! Most functionality and operations in this module are exposed via traits.
//! Many of these traits are included in the `prelude` module, and it is highly
//! recommended to import the `prelude`'s contents as seen in the examples.
//!
//! Generator traits implemented by primitives expose verbose function names
//! like `polygons_with_uv_map` or `vertices_with_position` to avoid ambiguity.
//! This is a somewhat unorthodox use of the term "with" in Rust function
//! names, but the alternatives are much less clear, especially when
//! neighboring other similar function names.
//!
//! # Examples
//!
//! Generating position data for a sphere:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point3;
//! use plexus::prelude::*;
//! use plexus::primitive::sphere::UvSphere;
//!
//! # fn main() {
//! let sphere = UvSphere::new(16, 16);
//! // Generate the unique set of positional vertices.
//! // Convert the position data into `Point3<f32>` and collect it into a buffer.
//! let positions = sphere
//!     .vertices_with_position()
//!     .map(|position| -> Point3<f32> { position.into() })
//!     .collect::<Vec<_>>();
//! // Generate polygons indexing the unique set of positional vertices.
//! // These indeces depend on the attribute (position, normal, etc.).
//! // Decompose the indexing polygons into triangles and vertices and then collect
//! // the index data into a buffer.
//! let indices = sphere
//!     .indices_for_position()
//!     .triangulate()
//!     .vertices()
//!     .collect::<Vec<_>>();
//! # }
//! ```
//! Generating position data for a cube using an indexer:
//!
//! ```rust
//! use plexus::prelude::*;
//! use plexus::primitive::cube::{Bounds, Cube};
//! use plexus::primitive::index::HashIndexer;
//!
//! let (indices, positions) = Cube::new()
//!     .polygons_with_position_from(Bounds::unit_radius())
//!     .triangulate()
//!     .index_vertices(HashIndexer::default());
//! ```

pub mod cube;
pub mod decompose;
pub mod generate;
pub mod index;
pub mod sphere;
mod topology;

pub use self::topology::*;
