//! This module provides tools for generating meshes for simple shapes like
//! cubes and spheres.
//!
//! The interface is iterator-based and begins with a unit shape and
//! manipulates its constituent primitives like points, lines, and polygons.
//! All shapes provide position information and some can additionally generate
//! index, texture, and conjoint point information as well.
//!
//! # Examples
//!
//! Generating position and index data for a scaled sphere mesh:
//!
//! ```
//! use plexus::generate::{IndexedPolygons, Points, SpatialPoints, Triangulate};
//! use plexus::generate::sphere::UVSphere;
//!
//! let sphere = UVSphere::with_unit_radius(16, 16);
//! let positions: Vec<_> = sphere
//!     .spatial_points() // Get the unique set of points.
//!     .map(|point| point * 10.0) // Scale the points by 10.
//!     .collect();
//! let indeces: Vec<_> = sphere
//!     .indexed_polygons() // Get indeces into the unique set of points as polygons.
//!     .triangulate() // Decompose the polygons into triangles.
//!     .points() // Decompose the triangles into points (indeces).
//!     .collect();
//! ```

pub mod cube;
mod decompose;
mod generate;
mod index;
pub mod sphere;
mod topology;

pub use self::decompose::{IntoLines, IntoVertices, IntoSubdivisions, IntoTetrahedrons,
                          IntoTriangles, Lines, Vertices, Subdivide, Tetrahedrons, Triangulate};
pub use self::generate::{IndexedPolygons, SpatialVertices, SpatialPolygons, TexturedPolygons};
pub use self::index::{HashIndexer, IndexTopology};
pub use self::topology::{Line, MapVertices, Polygon, Polygonal, Rotate, Topological, Triangle,
                         Quad};
