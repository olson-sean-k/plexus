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
#[cfg(feature = "geometry-nalgebra")]
extern crate nalgebra;
extern crate num;
extern crate typenum;

pub mod buffer;
pub mod geometry;
pub mod graph;
pub mod primitive;

// Re-exported to avoid requiring a direct dependency on decorum.
pub use decorum::{R32, R64};

pub mod prelude {
    pub use geometry::{Duplet, Triplet};
    pub use primitive::decompose::{
        Edges, IntoEdges, IntoSubdivisions, IntoTetrahedrons, IntoTriangles, IntoVertices,
        Subdivide, Tetrahedrons, Triangulate, Vertices,
    };
    pub use primitive::generate::{
        PolygonGenerator, PolygonsWithIndex, PolygonsWithPosition, PolygonsWithTexture,
        VertexGenerator, VerticesWithPosition,
    };
    pub use primitive::index::{CollectWithIndexer, FlatIndexVertices, IndexVertices};
    pub use primitive::{Converged, Map, MapVertices, Zip};
}
