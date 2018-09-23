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
//! Most functionality and operations in this module are exposed via traits.
//! Many of these traits are included in the `prelude` module, and it is highly
//! recommended to import the `prelude`'s contents as seen in the examples.
//!
//! Generator traits implemented by primitives expose verbose function names
//! like `polygons_with_index` or `vertices_with_position` to avoid ambiguity.
//! This is a somewhat unorthodox use of the term "with" in Rust function
//! names, but the alternatives are much less clear, especially when
//! neighboring other similar function names.
//!
//! # Examples
//!
//! Generating position and index buffers for a scaled sphere:
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
//! use plexus::prelude::*;
//! use plexus::primitive::cube::{Bounds, Cube};
//! use plexus::primitive::LruIndexer;
//!
//! let (indeces, positions) = Cube::new()
//!     .polygons_with_position_from(Bounds::unit_radius())
//!     .triangulate()
//!     .index_vertices(LruIndexer::default());
//! ```

pub mod cube;
mod decompose;
mod generate;
mod index;
pub mod sphere;
mod topology;

use decorum::Real;
use num::{One, Zero};
use std::ops::Div;

pub(crate) use self::index::{FromIndexer, Indexer};

pub use self::decompose::{
    Edges, IntoEdges, IntoSubdivisions, IntoTetrahedrons, IntoTriangles, IntoVertices, Subdivide,
    Tetrahedrons, Triangulate, Vertices,
};
pub use self::generate::{
    PolygonGenerator, PolygonsWithIndex, PolygonsWithPosition, PolygonsWithTexture,
    VertexGenerator, VerticesWithPosition,
};
pub use self::index::{
    CollectWithIndexer, FlatIndexVertices, HashIndexer, IndexVertices, LruIndexer,
};
pub use self::topology::{
    zip_vertices, Arity, Converged, Edge, Map, MapVertices, Polygon, Polygonal, Quad, Rotate,
    Topological, Triangle, Zip,
};

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
