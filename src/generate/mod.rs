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
//! use plexus::generate::{IndexedPolygons, SpatialVertices, Triangulate, Vertices};
//! use plexus::generate::sphere::UVSphere;
//!
//! let sphere = UVSphere::<f32>::with_unit_radius(16, 16);
//! let positions: Vec<_> = sphere
//!     .spatial_vertices() // Generate the unique set of positional vertices.
//!     .map(|(x, y, z)| (x * 10.0, y * 10.0, z * 10.0)) // Scale the positions by 10.
//!     .collect();
//! let indeces: Vec<_> = sphere
//!     .indexed_polygons() // Generate polygons indexing the unique set of vertices.
//!     .triangulate() // Decompose the polygons into triangles.
//!     .vertices() // Decompose the triangles into vertices (indeces).
//!     .collect();
//! ```

// TODO: Primitives are parameterized over the type of scalars used for spatial
//       data. This can be interpreted as the vector space and affects the
//       internal data describing the primitive. Other data, like texture
//       coordinates, are not parameterized at all. It may be more consistent
//       to parameterize all of this data, either as individual type parameters
//       or via a trait (like `Geometry`).

pub mod cube;
mod decompose;
mod generate;
mod index;
pub mod sphere;
mod topology;

pub use self::decompose::{IntoLines, IntoVertices, IntoSubdivisions, IntoTetrahedrons,
                          IntoTriangles, Lines, Vertices, Subdivide, Tetrahedrons, Triangulate};
pub use self::generate::{IndexedPolygons, SpatialVertices, SpatialPolygons, TexturedPolygons};
pub use self::index::{HashIndexer, IndexVertices};
pub use self::topology::{Line, MapVertices, Polygon, Polygonal, Rotate, Topological, Triangle,
                         Quad};
