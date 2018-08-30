//! **Plexus** is a library for generating and manipulating 2D and 3D mesh
//! data.
//!
//! Please note that versions in the `0.0.*` series are experimental and
//! unstable! Use exact version constraints when specifying a dependency to
//! avoid spurious breakage.
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
pub mod geometry;
pub mod graph;
pub mod primitive;

// Re-exported to avoid requiring a direct dependency on decorum.
pub use decorum::{R32, R64};

pub mod prelude {
    pub use geometry::{Duplet, Triplet};
    pub use primitive::{
        CollectWithIndexer, FlatIndexVertices, IndexVertices, IntoEdges, IntoSubdivisions,
        IntoTetrahedrons, IntoTriangles, IntoVertices, MapVertices, PolygonGenerator,
        PolygonsWithIndex, PolygonsWithPosition, PolygonsWithTexture, Triangulate, VertexGenerator,
        Vertices, VerticesWithPosition,
    };
}

trait BoolExt: Sized {
    fn option(self) -> Option<()>;

    fn some<T>(self, some: T) -> Option<T>;

    fn some_with<T, F>(self, f: F) -> Option<T>
    where
        F: Fn() -> T;
}

impl BoolExt for bool {
    fn option(self) -> Option<()> {
        if self {
            Some(())
        }
        else {
            None
        }
    }

    fn some<T>(self, some: T) -> Option<T> {
        if self {
            Some(some)
        }
        else {
            None
        }
    }

    fn some_with<T, F>(self, f: F) -> Option<T>
    where
        F: Fn() -> T,
    {
        if self {
            Some(f())
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
