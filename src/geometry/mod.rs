//! Geometric traits and primitives.
//!
//! These traits are used to support geometric operations in generators and
//! graphs. Implementing these traits implicitly implements internal traits,
//! which in turn enable geometric features.
//!
//! To use types as geometry in a `MeshGraph` only requires implementing the
//! `Geometry` trait. Implementing operations like `Cross` and `Normalize` for
//! the `Vertex` attribute enables geometric features like extrusion,
//! centroids, midpoints, etc.

pub mod convert;

// Feature modules. These are empty unless `geometry-*` features are enabled.
mod cgmath;
mod mint;
mod nalgebra;

/// Graph geometry.
///
/// Specifies the types used to represent geometry for vertices, arcs, edges,
/// and faces in a graph. Arbitrary types can be used, including `()` for no
/// geometry at all.
///
/// Geometry is vertex-based. Geometric operations depend on understanding the
/// positional data in vertices, and operational traits mostly apply to such
/// positional data.
///
/// # Examples
///
/// ```rust
/// # extern crate decorum;
/// # extern crate nalgebra;
/// # extern crate num;
/// # extern crate plexus;
/// use decorum::R64;
/// use nalgebra::{Point3, Vector4};
/// use num::One;
/// use plexus::geometry::convert::{AsPosition, IntoGeometry};
/// use plexus::geometry::Geometry;
/// use plexus::graph::MeshGraph;
/// use plexus::prelude::*;
/// use plexus::primitive::sphere::UvSphere;
///
/// // Vertex-only geometry with a position and color.
/// #[derive(Clone, Copy, Eq, Hash, PartialEq)]
/// pub struct VertexGeometry {
///     pub position: Point3<R64>,
///     pub color: Vector4<R64>,
/// }
///
/// impl Geometry for VertexGeometry {
///     type Vertex = Self;
///     type Arc = ();
///     type Edge = ();
///     type Face = ();
/// }
///
/// impl AsPosition for VertexGeometry {
///     type Target = Point3<R64>;
///
///     fn as_position(&self) -> &Self::Target {
///         &self.position
///     }
///
///     fn as_position_mut(&mut self) -> &mut Self::Target {
///         &mut self.position
///     }
/// }
///
/// # fn main() {
/// // Create a mesh from a sphere primitive and map the geometry data.
/// let mut graph = UvSphere::new(8, 8)
///     .polygons_with_position()
///     .map_vertices(|position| VertexGeometry {
///         position: position.into_geometry(),
///         color: Vector4::new(One::one(), One::one(), One::one(), One::one()),
///     })
///     .collect::<MeshGraph<VertexGeometry>>();
/// # }
/// ```
pub trait Geometry: Sized {
    type Vertex: Clone;
    type Arc: Clone + Default;
    type Edge: Clone + Default;
    type Face: Clone + Default;
}

impl Geometry for () {
    type Vertex = ();
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> Geometry for (T, T, T)
where
    T: Clone,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

pub mod alias {
    use crate::geometry::convert::AsPosition;
    use crate::geometry::Geometry;

    pub type VertexPosition<G> = <<G as Geometry>::Vertex as AsPosition>::Target;
}
