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
//! ```
//! use plexus::generate::{PolygonsWithIndex, Triangulate, Vertices, VerticesWithPosition};
//! use plexus::generate::sphere::UVSphere;
//!
//! let sphere = UVSphere::<f32>::with_unit_radius(16, 16);
//! let positions: Vec<_> = sphere
//!     .vertices_with_position() // Generate the unique set of positional vertices.
//!     .map(|(x, y, z)| (x * 10.0, y * 10.0, z * 10.0)) // Scale the positions by 10.
//!     .collect();
//! let indeces: Vec<_> = sphere
//!     .polygons_with_index() // Generate polygons indexing the unique set of vertices.
//!     .triangulate() // Decompose the polygons into triangles.
//!     .vertices() // Decompose the triangles into vertices (indeces).
//!     .collect();
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
mod geometry;
mod index;
pub mod sphere;
mod topology;

pub(crate) use self::decompose::{IntoTriangles, IntoVertices};
pub(crate) use self::geometry::Unit;

pub use self::decompose::{Lines, Subdivide, Tetrahedrons, Triangulate, Vertices};
pub use self::generate::{PolygonsWithIndex, PolygonsWithPosition, PolygonsWithTexture,
                         VerticesWithPosition};
pub use self::geometry::HashConjugate;
pub use self::index::{HashIndexer, IndexVertices};
pub use self::topology::{Line, MapVertices, Polygon, Polygonal, Quad, Rotate, Topological,
                         Triangle};

pub mod prelude {
    pub use super::{MapVertices, PolygonsWithIndex, PolygonsWithPosition, PolygonsWithTexture,
                    Triangulate, Vertices, VerticesWithPosition};
}
