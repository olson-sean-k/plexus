use num::Float;

use generate::{Duplet, Triplet};
use graph::mesh::Mesh;

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

// TODO: This reflexive implementation disallows interior conversions. For
//       example, it is not possible to extend geometry conversion from
//       `Vertex<T>` to `Vertex<U>` or `Mesh<T>` to `Mesh<U>`, because there is
//       no way to constrain `T` and `U` such that `T != U`. See
//       `FromInteriorGeometry`.
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

// This trait does not require `Default`, because supporting nalgebra and mesh
// generation requires being explicit about vertex geometry. The types exposed
// by nalgebra do not support `Default`, but it is convenient to support them
// directly and emit them from the `generate` module.
//
// Instead, the associated `Edge` and `Face` types of `Geometry` require
// `Default` on their own.
pub trait Attribute {}

pub trait Geometry: Sized {
    type Vertex: Attribute;
    type Edge: Attribute + Default;
    type Face: Attribute + Default;

    fn compute(_: &mut Mesh<Self>) {}
}


impl Attribute for () {}

impl<T> Attribute for Triplet<T>
where
    T: Default,
{
}

impl Geometry for () {
    type Vertex = ();
    type Edge = ();
    type Face = ();
}

impl<T> Geometry for Triplet<T>
where
    T: Default,
{
    type Vertex = Self;
    type Edge = ();
    type Face = ();
}

pub trait Normalize {
    fn normalize(self) -> Self;
}

pub trait Cross<T = Self> {
    type Output;

    fn cross(self, other: T) -> Self::Output;
}

pub trait AsPosition {
    type Target;

    fn as_position(&self) -> &Self::Target;
    fn as_position_mut(&mut self) -> &mut Self::Target;
}

#[cfg(feature = "geometry-cgmath")]
mod feature_geometry_cgmath {}
#[cfg(not(feature = "geometry-cgmath"))]
mod feature_geometry_cgmath {}

#[cfg(feature = "geometry-nalgebra")]
mod feature_geometry_nalgebra {
    use alga::general::Real;
    use nalgebra::{Point2, Point3, Scalar, Vector2, Vector3};
    use nalgebra::core::Matrix;

    use super::*;

    impl<T> FromGeometry<Duplet<T>> for Point2<T>
    where
        T: Scalar,
    {
        fn from_geometry(other: Duplet<T>) -> Self {
            Point2::new(other.0, other.1)
        }
    }

    impl<T> FromGeometry<Triplet<T>> for Point3<T>
    where
        T: Scalar,
    {
        fn from_geometry(other: Triplet<T>) -> Self {
            Point3::new(other.0, other.1, other.2)
        }
    }

    impl<T> FromGeometry<Duplet<T>> for Vector2<T>
    where
        T: Scalar,
    {
        fn from_geometry(other: Duplet<T>) -> Self {
            Vector2::new(other.0, other.1)
        }
    }

    impl<T> FromGeometry<Triplet<T>> for Vector3<T>
    where
        T: Scalar,
    {
        fn from_geometry(other: Triplet<T>) -> Self {
            Vector3::new(other.0, other.1, other.2)
        }
    }

    impl<T> Attribute for Point3<T>
    where
        T: Scalar,
    {
    }

    impl<T> Geometry for Point3<T>
    where
        T: Scalar,
    {
        type Vertex = Self;
        type Edge = ();
        type Face = ();
    }

    impl<T> Normalize for Vector3<T>
    where
        T: Float + Real + Scalar,
    {
        fn normalize(self) -> Self {
            Matrix::normalize(&self)
        }
    }

    impl<T> Cross for Vector3<T>
    where
        T: Float + Real + Scalar,
    {
        type Output = Self;

        fn cross(self, other: Self) -> Self::Output {
            Matrix::cross(&self, &other)
        }
    }

    impl<T> AsPosition for Point3<T>
    where
        T: Scalar,
    {
        type Target = Self;

        fn as_position(&self) -> &Self::Target {
            self
        }

        fn as_position_mut(&mut self) -> &mut Self::Target {
            self
        }
    }
}
#[cfg(not(feature = "geometry-nalgebra"))]
mod feature_geometry_nalgebra {}

pub use self::feature_geometry_cgmath::*;
pub use self::feature_geometry_nalgebra::*;
