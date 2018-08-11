//! **Plexus** is a library for generating and manipulating 3D mesh data.
//!
//! Please note that versions in the `0.0.*` series are experimental and
//! extremely unstable!
#![allow(unknown_lints)] // Allow clippy lints.

extern crate arrayvec;
#[cfg(feature = "geometry-cgmath")]
extern crate cgmath;
extern crate decorum;
#[macro_use]
extern crate derivative;
#[macro_use]
extern crate failure;
#[macro_use]
extern crate itertools;
#[cfg(feature = "geometry-nalgebra")]
extern crate nalgebra;
extern crate num;

pub mod buffer;
pub mod generate;
pub mod geometry;
pub mod graph;

pub use decorum::{R32, R64};

pub mod prelude {
    pub use generate::{
        CollectWithIndexer, FlatIndexVertices, IndexVertices, MapVertices, PolygonGenerator,
        PolygonsWithIndex, PolygonsWithPosition, PolygonsWithTexture, Triangulate, VertexGenerator,
        Vertices, VerticesWithPosition,
    };
    pub use geometry::{Duplet, Triplet};
}

trait BoolExt: Sized {
    fn into_some<T>(self, some: T) -> Option<T>;
}

impl BoolExt for bool {
    fn into_some<T>(self, some: T) -> Option<T> {
        if self {
            Some(some)
        }
        else {
            None
        }
    }
}

trait OptionExt<T> {
    fn and_if<F>(self, f: F) -> Self
    where
        F: Fn(&T) -> bool;
}

impl<T> OptionExt<T> for Option<T> {
    fn and_if<F>(mut self, f: F) -> Self
    where
        F: Fn(&T) -> bool,
    {
        match self.take() {
            Some(value) => if f(&value) {
                Some(value)
            }
            else {
                None
            },
            _ => None,
        }
    }
}
