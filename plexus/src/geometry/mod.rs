//! Geometric traits and computational geometry.
//!
//! Plexus uses the [`theon`] crate to abstract over types that represent
//! Euclidean spaces and implement linear algebra. Types and traits are
//! re-exported from [`theon`] in this module, but it may be necessary to import
//! additional types from [`theon`]. See the [`integration`] module.
//!
//! [`theon`]: https://crates.io/crates/theon
//!
//! [`integration`]: crate::integration

use num::{One, Zero};

pub mod partition;

pub use theon::query::*;
pub use theon::space::{Scalar, Vector};
pub use theon::{AsPosition, AsPositionMut, Position};

pub trait FromGeometry<T> {
    fn from_geometry(other: T) -> Self;
}

impl<T> FromGeometry<T> for T {
    fn from_geometry(other: T) -> Self {
        other
    }
}

/// Geometry elision into `()`.
impl<T> FromGeometry<T> for ()
where
    T: UnitGeometry,
{
    fn from_geometry(_: T) -> Self {}
}

/// Geometry elision from `()`.
impl<T> FromGeometry<()> for T
where
    T: UnitGeometry + Default,
{
    fn from_geometry(_: ()) -> Self {
        T::default()
    }
}

/// Geometry elision.
///
/// Geometric types that implement this trait may be elided. In particular,
/// these types may be converted into and from `()` via the [`FromGeometry`] and
/// [`IntoGeometry`] traits.
///
/// For a geometric type `T`, the following table illustrates the elisions in
/// which `T` may participate:
///
/// | Bounds on `T`            | From | Into |
/// |--------------------------|------|------|
/// | `UnitGeometry`           | `T`  | `()` |
/// | `Default + UnitGeometry` | `()` | `T`  |
///
/// These conversions are useful when converting between mesh data structures
/// with incompatible geometry, such as from a [`MeshGraph`] with face geometry
/// to a [`MeshBuffer`] that cannot support such geometry.
///
/// When geometry features are enabled, `UnitGeometry` is implemented for
/// integrated foreign types.
///
/// [`FromGeometry`]: crate::geometry::FromGeometry
/// [`IntoGeometry`]: crate::geometry::IntoGeometry
/// [`MeshBuffer`]: crate::buffer::MeshBuffer
/// [`MeshGraph`]: crate::graph::MeshGraph
pub trait UnitGeometry {}

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

pub trait Metric: Eq + One + Ord + Zero {}

impl<Q> Metric for Q where Q: Eq + One + Ord + Zero {}
