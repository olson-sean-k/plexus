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
pub mod ops;
pub mod query;
pub mod space;

// Feature modules. These are empty unless `geometry-*` features are enabled.
mod cgmath;
mod mint;
mod nalgebra;

use decorum::{Real, R64};
use num::{self, Num, NumCast, One, ToPrimitive, Zero};
use std::ops::Div;

use crate::geometry::convert::FromGeometry;
use crate::geometry::ops::Interpolate;

trait Half {
    fn half() -> Self;
}

impl<T> Half for T
where
    T: Div<T, Output = T> + One + Real,
{
    fn half() -> Self {
        let one = T::one();
        one / (one + one)
    }
}

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

/// Homogeneous duplet.
///
/// Provides basic vertex geometry and a grouping of values emitted by
/// generators. Conversions into commonly used types from commonly used
/// libraries are supported. See feature flags.
#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Duplet<T>(pub T, pub T);

impl<T> Duplet<T> {
    pub fn one() -> Self
    where
        T: One,
    {
        Duplet(One::one(), One::one())
    }

    pub fn zero() -> Self
    where
        T: Zero,
    {
        Duplet(Zero::zero(), Zero::zero())
    }
}

impl<T, U> FromGeometry<(U, U)> for Duplet<T>
where
    T: NumCast,
    U: ToPrimitive,
{
    fn from_geometry(other: (U, U)) -> Self {
        Duplet(T::from(other.0).unwrap(), T::from(other.1).unwrap())
    }
}

impl<T> Geometry for Duplet<T>
where
    T: Clone,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> Interpolate for Duplet<T>
where
    T: Copy + Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Duplet(lerp(self.0, other.0, f), lerp(self.1, other.1, f))
    }
}

/// Homogeneous triplet.
///
/// Provides basic vertex geometry and a grouping of values emitted by
/// generators. Conversions into commonly used types from commonly used
/// libraries are supported. See feature flags.
#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Triplet<T>(pub T, pub T, pub T);

impl<T> Triplet<T> {
    pub fn one() -> Self
    where
        T: One,
    {
        Triplet(One::one(), One::one(), One::one())
    }

    pub fn zero() -> Self
    where
        T: Zero,
    {
        Triplet(Zero::zero(), Zero::zero(), Zero::zero())
    }
}

impl<T, U> FromGeometry<(U, U, U)> for Triplet<T>
where
    T: NumCast,
    U: ToPrimitive,
{
    fn from_geometry(other: (U, U, U)) -> Self {
        Triplet(
            T::from(other.0).unwrap(),
            T::from(other.1).unwrap(),
            T::from(other.2).unwrap(),
        )
    }
}

impl<T> Geometry for Triplet<T>
where
    T: Clone,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> Interpolate for Triplet<T>
where
    T: Copy + Num + NumCast,
{
    type Output = Self;

    fn lerp(self, other: Self, f: R64) -> Self::Output {
        Triplet(
            lerp(self.0, other.0, f),
            lerp(self.1, other.1, f),
            lerp(self.2, other.2, f),
        )
    }
}

pub fn lerp<T>(a: T, b: T, f: R64) -> T
where
    T: Num + NumCast,
{
    let f = num::clamp(f, Zero::zero(), One::one());
    let af = <R64 as NumCast>::from(a).unwrap() * (R64::one() - f);
    let bf = <R64 as NumCast>::from(b).unwrap() * f;
    <T as NumCast>::from(af + bf).unwrap()
}

pub mod alias {
    use crate::geometry::convert::AsPosition;
    use crate::geometry::space::{EuclideanSpace, VectorSpace};
    use crate::geometry::Geometry;

    pub type Scalar<S> = <Vector<S> as VectorSpace>::Scalar;
    pub type Vector<S> = <S as EuclideanSpace>::Difference;
    pub type VertexPosition<G> = <<G as Geometry>::Vertex as AsPosition>::Target;
}
