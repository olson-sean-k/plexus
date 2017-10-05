//! Mesh generation from primitives like cubes and spheres.
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
//! # Examples
//!
//! Generating position and index buffers for a scaled sphere:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point3;
//! use plexus::generate::sphere::UVSphere;
//! use plexus::prelude::*;
//!
//! # fn main() {
//! let sphere = UVSphere::<f32>::with_unit_radius(16, 16);
//! let positions: Vec<_> = sphere
//!     .vertices_with_position() // Generate the unique set of positional vertices.
//!     .map(|position| -> Point3<_> { position.into() }) // Convert into a nalgebra type.
//!     .map(|position| position * 10.0) // Scale the positions by 10.
//!     .collect();
//! let indeces: Vec<_> = sphere
//!     .polygons_with_index() // Generate polygons indexing the unique set of vertices.
//!     .triangulate() // Decompose the polygons into triangles.
//!     .vertices() // Decompose the triangles into vertices (indeces).
//!     .collect();
//! # }
//! ```
//! Generating position and index buffers using an indexer:
//!
//! ```rust
//! use plexus::generate::LruIndexer;
//! use plexus::generate::cube::Cube;
//! use plexus::prelude::*;
//!
//! let (indeces, positions) = Cube::<f32>::with_unit_width()
//!     .polygons_with_position()
//!     .triangulate()
//!     .index_vertices(LruIndexer::default());
//! ```

// TODO: Primitives are parameterized over the type of scalars used for spatial
//       data. This can be interpreted as the vector space and affects the
//       internal data describing the primitive. See the `Unit` trait. Other
//       data, like texture coordinates, are not parameterized at all. It may
//       be more consistent to parameterize all of this data, either as
//       individual type parameters or via a trait (like `Geometry`). Default
//       type parameters are also provided.

pub mod cube;
mod decompose;
mod generate;
mod index;
pub mod sphere;
mod topology;
mod unit;

pub(crate) use self::decompose::{IntoTriangles, IntoVertices};
pub(crate) use self::index::{FromIndexer, Indexer};

pub use self::decompose::{Lines, Subdivide, Tetrahedrons, Triangulate, Vertices};
pub use self::generate::{PolygonGenerator, PolygonsWithIndex, PolygonsWithPosition,
                         PolygonsWithTexture, VertexGenerator, VerticesWithPosition};
pub use self::index::{CollectWithIndexer, HashIndexer, IndexVertices, LruIndexer};
pub use self::topology::{Line, MapVertices, Polygon, Polygonal, Quad, Rotate, Topological,
                         Triangle};
