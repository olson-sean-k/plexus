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
extern crate fnv;
extern crate fool;
#[macro_use]
extern crate itertools;
#[cfg(feature = "geometry-mint")]
extern crate mint;
#[cfg(feature = "geometry-nalgebra")]
extern crate nalgebra;
extern crate num;
extern crate typenum;

use decorum::Real;
use num::{One, Zero};
use std::ops::Div;

pub mod buffer;
pub mod geometry;
pub mod graph;
pub mod primitive;

// Re-exported to avoid requiring a direct dependency on decorum.
pub use decorum::{R32, R64};

// TODO: Documentation comments include static image content from the GitHub
//       repository. This is fragile and difficult to maintain. Use a mechanism
//       provided by rustdoc or doxidize for this instead.

pub mod prelude {
    pub use crate::geometry::{Duplet, Triplet};
    pub use crate::primitive::decompose::{
        Edges, IntoEdges, IntoSubdivisions, IntoTetrahedrons, IntoTriangles, IntoVertices,
        Subdivide, Tetrahedrons, Triangulate, Vertices,
    };
    pub use crate::primitive::generate::{
        IndicesForPosition, PolygonGenerator, PolygonsWithPosition, PolygonsWithTexture,
        VertexGenerator, VerticesWithPosition,
    };
    pub use crate::primitive::index::{CollectWithIndexer, FlatIndexVertices, IndexVertices};
    pub use crate::primitive::{Converged, Map, MapVertices, Zip};
}

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
