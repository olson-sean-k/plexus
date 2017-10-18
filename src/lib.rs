//! **Plexus** is a library for generating and manipulating 3D mesh data.
//!
//! Please note that versions in the `0.0.*` series are experimental and
//! extremely unstable!
#![allow(unknown_lints)] // Allow clippy lints.

#[cfg(feature = "geometry-nalgebra")]
extern crate alga;
extern crate arrayvec;
#[cfg(feature = "geometry-cgmath")]
extern crate cgmath;
#[macro_use]
extern crate derivative;
extern crate itertools;
#[cfg(feature = "geometry-nalgebra")]
extern crate nalgebra;
extern crate num;
extern crate ordered_float;

pub mod buffer;
pub mod generate;
pub mod geometry;
pub mod graph;
pub mod ordered;

pub mod prelude {
    pub use generate::{CollectWithIndexer, IndexVertices, MapVertices, PolygonGenerator,
                       PolygonsWithIndex, PolygonsWithPosition, PolygonsWithTexture, Triangulate,
                       VertexGenerator, Vertices, VerticesWithPosition};
    pub use geometry::{Duplet, Triplet};
    pub use ordered::HashConjugate;
}
