use num::Float;
use ordered_float::OrderedFloat;

use generate::{FromOrderedFloat, Unit, Vector};
use graph::topology::FaceMut;

pub trait FromGeometry<T> {
    fn from_geometry(_: T) -> Self;
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

// TODO: The implementation of this trait for `Mesh` would conflict with a
//       blanket reflexive implementation. Is there some way to express to the
//       compiler "where `Mesh<T> != Mesh<U>`"?
impl FromGeometry<()> for () {
    fn from_geometry(_: ()) -> Self {
        ()
    }
}

impl<T, U> FromGeometry<U> for T
where
    T: FromOrderedFloat<U> + Geometry + Vector,
    T::Scalar: Float + Unit,
    U: Geometry + Vector<Scalar = OrderedFloat<T::Scalar>>,
{
    fn from_geometry(geometry: U) -> Self {
        Self::from_ordered_float(geometry)
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

    fn compute_vertex_geometry() {}

    fn compute_edge_geometry() {}

    fn compute_face_geometry(_: FaceMut<Self>) {}
}


impl Attribute for () {}

impl<T> Attribute for (T, T, T)
where
    T: Default + Unit,
{
}

impl<T> Attribute for (OrderedFloat<T>, OrderedFloat<T>, OrderedFloat<T>)
where
    T: Default + Float + Unit,
{
}

impl Geometry for () {
    type Vertex = ();
    type Edge = ();
    type Face = ();
}

impl<T> Geometry for (T, T, T)
where
    T: Default + Unit,
{
    type Vertex = Self;
    type Edge = ();
    type Face = ();
}

impl<T> Geometry for (OrderedFloat<T>, OrderedFloat<T>, OrderedFloat<T>)
where
    T: Default + Float + Unit,
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

#[cfg(feature = "geometry-nalgebra")]
mod feature {
    use alga::general::Real;
    use nalgebra::{Point3, Scalar, Vector3};
    use nalgebra::core::Matrix;

    use super::*;

    impl<T> Attribute for Point3<T>
    where
        T: Scalar + Unit,
    {
    }

    impl<T> Geometry for Point3<T>
    where
        T: Scalar + Unit,
    {
        type Vertex = Self;
        type Edge = ();
        type Face = ();
    }

    impl<T> Normalize for Vector3<T>
    where
        T: Float + Real + Scalar + Unit,
    {
        fn normalize(self) -> Self {
            Matrix::normalize(&self)
        }
    }

    impl<T> Cross for Vector3<T>
    where
        T: Float + Real + Scalar + Unit,
    {
        type Output = Self;

        fn cross(self, other: Self) -> Self::Output {
            Matrix::cross(&self, &other)
        }
    }

    impl<T> AsPosition for Point3<T>
    where
        T: Scalar + Unit,
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
mod feature {}

pub use self::feature::*;
