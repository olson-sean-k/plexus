//! Geometric traits and primitives.
//!
//! These traits are used to support high-order operations in generators and
//! graphs. Implementing these traits implicitly implements internal traits,
//! which in turn enable geometric features.
//!
//! To use types as geometry in a `MeshGraph` only requires implementing the
//! `Geometry` and `Attribute` traits. Implementing operations like `Cross`,
//! `Normalize`, etc., enable features like extrusion and splitting.

use decorum::R64;
use num::{self, Num, NumCast, One, Zero};

pub mod compose;
pub mod convert;
pub mod ops;

/// Geometric attribute.
pub trait Attribute: Clone {}

/// Graph geometry.
///
/// Specifies the types used to represent geometry for vertices, half-edges,
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
/// use plexus::geometry::{Attribute, Geometry};
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
/// impl Attribute for VertexGeometry {}
///
/// impl Geometry for VertexGeometry {
///     type Vertex = Self;
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
    type Vertex: Attribute;
    type Edge: Attribute + Default;
    type Face: Attribute + Default;
}

impl Attribute for () {}

impl Geometry for () {
    type Vertex = ();
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

impl<T> Attribute for Duplet<T> where T: Clone {}

impl<T> Geometry for Duplet<T>
where
    T: Clone,
{
    type Vertex = Self;
    type Edge = ();
    type Face = ();
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

impl<T> Attribute for Triplet<T> where T: Clone {}

impl<T> Geometry for Triplet<T>
where
    T: Clone,
{
    type Vertex = Self;
    type Edge = ();
    type Face = ();
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
    use std::ops::Mul;

    use crate::geometry::compose::*;
    use crate::geometry::convert::*;
    use crate::geometry::*;

    pub type VertexPosition<G> = <<G as Geometry>::Vertex as AsPosition>::Target;
    pub type ScaledFaceNormal<G, T> = <<G as FaceNormal>::Normal as Mul<T>>::Output;
    pub type ScaledEdgeLateral<G, T> = <<G as EdgeLateral>::Lateral as Mul<T>>::Output;
}

#[cfg(feature = "geometry-cgmath")]
mod feature_geometry_cgmath {
    use cgmath::{Point2, Point3, Vector2, Vector3};
    use num::{NumCast, ToPrimitive};

    use crate::geometry::*;

    impl<T, U> From<Point2<U>> for Duplet<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from(other: Point2<U>) -> Self {
            Duplet(T::from(other.x).unwrap(), T::from(other.y).unwrap())
        }
    }

    impl<T, U> From<Vector2<U>> for Duplet<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from(other: Vector2<U>) -> Self {
            Duplet(T::from(other.x).unwrap(), T::from(other.y).unwrap())
        }
    }

    impl<T, U> Into<Point2<T>> for Duplet<U>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn into(self) -> Point2<T> {
            Point2::new(T::from(self.0).unwrap(), T::from(self.1).unwrap())
        }
    }

    impl<T, U> Into<Vector2<T>> for Duplet<U>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn into(self) -> Vector2<T> {
            Vector2::new(T::from(self.0).unwrap(), T::from(self.1).unwrap())
        }
    }

    impl<T, U> From<Point3<U>> for Triplet<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from(other: Point3<U>) -> Self {
            Triplet(
                T::from(other.x).unwrap(),
                T::from(other.y).unwrap(),
                T::from(other.z).unwrap(),
            )
        }
    }

    impl<T, U> From<Vector3<U>> for Triplet<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from(other: Vector3<U>) -> Self {
            Triplet(
                T::from(other.x).unwrap(),
                T::from(other.y).unwrap(),
                T::from(other.z).unwrap(),
            )
        }
    }

    impl<T, U> Into<Point3<T>> for Triplet<U>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn into(self) -> Point3<T> {
            Point3::new(
                T::from(self.0).unwrap(),
                T::from(self.1).unwrap(),
                T::from(self.2).unwrap(),
            )
        }
    }

    impl<T, U> Into<Vector3<T>> for Triplet<U>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn into(self) -> Vector3<T> {
            Vector3::new(
                T::from(self.0).unwrap(),
                T::from(self.1).unwrap(),
                T::from(self.2).unwrap(),
            )
        }
    }

    impl<T> Attribute for Point2<T> where T: Clone {}

    impl<T> Attribute for Point3<T> where T: Clone {}

    impl<T> Geometry for Point2<T>
    where
        T: Clone,
    {
        type Vertex = Self;
        type Edge = ();
        type Face = ();
    }

    impl<T> Geometry for Point3<T>
    where
        T: Clone,
    {
        type Vertex = Self;
        type Edge = ();
        type Face = ();
    }
}

#[cfg(feature = "geometry-mint")]
mod feature_geometry_mint {
    use mint::{Point2, Point3, Vector2, Vector3};
    use num::{NumCast, ToPrimitive};

    use crate::geometry::*;

    impl<T, U> From<Point2<U>> for Duplet<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from(other: Point2<U>) -> Self {
            Duplet(T::from(other.x).unwrap(), T::from(other.y).unwrap())
        }
    }

    impl<T, U> From<Vector2<U>> for Duplet<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from(other: Vector2<U>) -> Self {
            Duplet(T::from(other.x).unwrap(), T::from(other.y).unwrap())
        }
    }

    impl<T, U> Into<Point2<T>> for Duplet<U>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn into(self) -> Point2<T> {
            Point2 {
                x: T::from(self.0).unwrap(),
                y: T::from(self.1).unwrap(),
            }
        }
    }

    impl<T, U> Into<Vector2<T>> for Duplet<U>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn into(self) -> Vector2<T> {
            Vector2 {
                x: T::from(self.0).unwrap(),
                y: T::from(self.1).unwrap(),
            }
        }
    }

    impl<T, U> From<Point3<U>> for Triplet<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from(other: Point3<U>) -> Self {
            Triplet(
                T::from(other.x).unwrap(),
                T::from(other.y).unwrap(),
                T::from(other.z).unwrap(),
            )
        }
    }

    impl<T, U> From<Vector3<U>> for Triplet<T>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn from(other: Vector3<U>) -> Self {
            Triplet(
                T::from(other.x).unwrap(),
                T::from(other.y).unwrap(),
                T::from(other.z).unwrap(),
            )
        }
    }

    impl<T, U> Into<Point3<T>> for Triplet<U>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn into(self) -> Point3<T> {
            Point3 {
                x: T::from(self.0).unwrap(),
                y: T::from(self.1).unwrap(),
                z: T::from(self.2).unwrap(),
            }
        }
    }

    impl<T, U> Into<Vector3<T>> for Triplet<U>
    where
        T: NumCast,
        U: ToPrimitive,
    {
        fn into(self) -> Vector3<T> {
            Vector3 {
                x: T::from(self.0).unwrap(),
                y: T::from(self.1).unwrap(),
                z: T::from(self.2).unwrap(),
            }
        }
    }

    impl<T> Attribute for Point2<T> where T: Clone {}

    impl<T> Attribute for Point3<T> where T: Clone {}

    impl<T> Geometry for Point2<T>
    where
        T: Clone,
    {
        type Vertex = Self;
        type Edge = ();
        type Face = ();
    }

    impl<T> Geometry for Point3<T>
    where
        T: Clone,
    {
        type Vertex = Self;
        type Edge = ();
        type Face = ();
    }
}

#[cfg(feature = "geometry-nalgebra")]
mod feature_geometry_nalgebra {
    use nalgebra::{Point2, Point3, Scalar, Vector2, Vector3};
    use num::{NumCast, ToPrimitive};

    use crate::geometry::*;

    impl<T, U> From<Point2<U>> for Duplet<T>
    where
        T: NumCast,
        U: Scalar + ToPrimitive,
    {
        fn from(other: Point2<U>) -> Self {
            Duplet(T::from(other.x).unwrap(), T::from(other.y).unwrap())
        }
    }

    impl<T, U> From<Vector2<U>> for Duplet<T>
    where
        T: NumCast,
        U: Scalar + ToPrimitive,
    {
        fn from(other: Vector2<U>) -> Self {
            Duplet(T::from(other.x).unwrap(), T::from(other.y).unwrap())
        }
    }

    impl<T, U> Into<Point2<T>> for Duplet<U>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn into(self) -> Point2<T> {
            Point2::new(T::from(self.0).unwrap(), T::from(self.1).unwrap())
        }
    }

    impl<T, U> Into<Vector2<T>> for Duplet<U>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn into(self) -> Vector2<T> {
            Vector2::new(T::from(self.0).unwrap(), T::from(self.1).unwrap())
        }
    }

    impl<T, U> From<Point3<U>> for Triplet<T>
    where
        T: NumCast,
        U: Scalar + ToPrimitive,
    {
        fn from(other: Point3<U>) -> Self {
            Triplet(
                T::from(other.x).unwrap(),
                T::from(other.y).unwrap(),
                T::from(other.z).unwrap(),
            )
        }
    }

    impl<T, U> From<Vector3<U>> for Triplet<T>
    where
        T: NumCast,
        U: Scalar + ToPrimitive,
    {
        fn from(other: Vector3<U>) -> Self {
            Triplet(
                T::from(other.x).unwrap(),
                T::from(other.y).unwrap(),
                T::from(other.z).unwrap(),
            )
        }
    }

    impl<T, U> Into<Point3<T>> for Triplet<U>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn into(self) -> Point3<T> {
            Point3::new(
                T::from(self.0).unwrap(),
                T::from(self.1).unwrap(),
                T::from(self.2).unwrap(),
            )
        }
    }

    impl<T, U> Into<Vector3<T>> for Triplet<U>
    where
        T: NumCast + Scalar,
        U: ToPrimitive,
    {
        fn into(self) -> Vector3<T> {
            Vector3::new(
                T::from(self.0).unwrap(),
                T::from(self.1).unwrap(),
                T::from(self.2).unwrap(),
            )
        }
    }

    impl<T> Attribute for Point2<T> where T: Scalar {}

    impl<T> Attribute for Point3<T> where T: Scalar {}

    impl<T> Geometry for Point2<T>
    where
        T: Scalar,
    {
        type Vertex = Self;
        type Edge = ();
        type Face = ();
    }

    impl<T> Geometry for Point3<T>
    where
        T: Scalar,
    {
        type Vertex = Self;
        type Edge = ();
        type Face = ();
    }
}
