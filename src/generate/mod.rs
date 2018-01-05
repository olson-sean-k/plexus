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
//! Generator traits implemented by primitives expose verbose function names
//! like `polygons_with_index` or `vertices_with_position` to avoid ambiguity.
//! For example, a shorter name like `index_polygons` is confusing: is "index"
//! a noun or verb? Adjectives are also not too usesful: `indexed_polygons` is
//! very similar to `index_polygons`, which is an operation exposed by
//! unrelated traits. For normal vectors, obvious adjectives would be synthetic
//! or confusing, such as "normaled". Instead, longer but clearer names are
//! used.
//!
//! # Examples
//!
//! Generating position and index buffers for a scaled sphere:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point3;
//! use plexus::generate::sphere;
//! use plexus::prelude::*;
//!
//! # fn main() {
//! let sphere = sphere::UvSphere::new(16, 16);
//! let positions = sphere
//!     .vertices_with_position() // Generate the unique set of positional vertices.
//!     .map(|position| -> Point3<f32> { position.into() }) // Convert into a nalgebra type.
//!     .map(|position| position * 10.0) // Scale the positions by 10.
//!     .collect::<Vec<_>>();
//! let indeces = sphere
//!     .polygons_with_index() // Generate polygons indexing the unique set of vertices.
//!     .triangulate() // Decompose the polygons into triangles.
//!     .vertices() // Decompose the triangles into vertices (indeces).
//!     .collect::<Vec<_>>();
//! # }
//! ```
//! Generating position and index buffers using an indexer:
//!
//! ```rust
//! use plexus::generate::LruIndexer;
//! use plexus::generate::cube;
//! use plexus::prelude::*;
//!
//! let (indeces, positions) = cube::Cube::new()
//!     .polygons_with_position_with(cube::Bounds::unit_radius())
//!     .triangulate()
//!     .index_vertices(LruIndexer::default());
//! ```

pub mod cube;
mod decompose;
#[allow(module_inception)]
mod generate;
mod index;
pub mod sphere;
mod topology;

use decorum::Real;
use num::{One, Zero};
use std::ops::Div;

pub(crate) use self::decompose::IntoVertices;
pub(crate) use self::index::{FromIndexer, Indexer};
pub(crate) use self::topology::{Arity, MapVerticesInto};

pub use self::decompose::{Edges, Subdivide, Tetrahedrons, Triangulate, Vertices};
pub use self::generate::{PolygonGenerator, PolygonsWithIndex, PolygonsWithPosition,
                         PolygonsWithTexture, VertexGenerator, VerticesWithPosition};
pub use self::index::{CollectWithIndexer, FlatIndexVertices, HashIndexer, IndexVertices,
                      LruIndexer};
pub use self::topology::{zip_vertices, Edge, MapVertices, Polygon, Polygonal, Quad, Rotate,
                         Topological, Triangle};

trait Half {
    fn half() -> Self;
}

impl<T> Half for T
where
    T: Div<T, Output = T> + One + Real + Zero,
{
    fn half() -> Self {
        let one = T::one();
        one / (one + one)
    }
}
