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

// Feature modules. These are empty unless `geometry-*` features are enabled.
mod cgmath;
mod mint;
mod nalgebra;

pub type VertexPosition<G> = <<G as Geometry>::Vertex as AsPosition>::Target;

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

pub trait FromGeometry<T> {
    fn from_geometry(other: T) -> Self;
}

pub trait IntoGeometry<T> {
    fn into_geometry(self) -> T;
}

impl<T, U> IntoGeometry<U> for T
where
    U: FromGeometry<T>,
{
    fn into_geometry(self) -> U {
        U::from_geometry(self)
    }
}

impl<T> FromGeometry<T> for T {
    fn from_geometry(other: T) -> Self {
        other
    }
}

// TODO: The interior versions of geometry conversion traits do not have a
//       reflexive implementation. This allows for conversions from `Mesh<T>`
//       to `Mesh<U>`, where `T` and `U` may be the same.
//
//       This is a bit confusing; consider removing these if they aren't
//       useful.
pub trait FromInteriorGeometry<T> {
    fn from_interior_geometry(other: T) -> Self;
}

pub trait IntoInteriorGeometry<T> {
    fn into_interior_geometry(self) -> T;
}

impl<T, U> IntoInteriorGeometry<U> for T
where
    U: FromInteriorGeometry<T>,
{
    fn into_interior_geometry(self) -> U {
        U::from_interior_geometry(self)
    }
}

/// Exposes a reference to positional vertex data.
///
/// To enable geometric features, this trait must be implemented for the type
/// representing vertex data. Additionally, geometric operations should be
/// implemented for the `Target` type.
pub trait AsPosition {
    type Target;

    fn as_position(&self) -> &Self::Target;
    fn as_position_mut(&mut self) -> &mut Self::Target;
}
