use num::{Float, Num, NumCast};
use std::ops::{AddAssign, MulAssign, Sub};

use generate::{Duplet, Triplet};
use graph::mesh::Mesh;
use graph::topology::FaceRef;
use self::alias::*;

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
pub trait Attribute: Clone {}

impl Attribute for () {}

impl<T> Attribute for Triplet<T>
where
    T: Clone + Default,
{
}

pub trait Geometry: Sized {
    type Vertex: Attribute;
    type Edge: Attribute + Default;
    type Face: Attribute + Default;

    fn compute(_: &mut Mesh<Self>) {}
}

impl Geometry for () {
    type Vertex = ();
    type Edge = ();
    type Face = ();
}

impl<T> Geometry for Triplet<T>
where
    T: Clone + Default,
{
    type Vertex = Self;
    type Edge = ();
    type Face = ();
}

pub trait FaceNormal: Geometry {
    type Normal;

    fn normal(face: FaceRef<Self>) -> Result<Self::Normal, ()>;
}

impl<G> FaceNormal for G
where
    G: Geometry,
    G::Vertex: AsPosition,
    VertexPosition<G>: Clone + Sub,
    <VertexPosition<G> as Sub>::Output: Cross,
    <<VertexPosition<G> as Sub>::Output as Cross>::Output: Normalize,
{
    type Normal = <<<G::Vertex as AsPosition>::Target as Sub>::Output as Cross>::Output;

    fn normal(face: FaceRef<Self>) -> Result<Self::Normal, ()> {
        let positions = face.vertices()
            .take(3)
            .map(|vertex| vertex.geometry.as_position().clone())
            .collect::<Vec<_>>();
        let (a, b, c) = (&positions[0], &positions[1], &positions[2]);
        let ab = a.clone() - b.clone();
        let bc = b.clone() - c.clone();
        Ok(ab.cross(bc).normalize())
    }
}

pub trait FaceCentroid: Geometry {
    type Centroid;

    fn centroid(face: FaceRef<Self>) -> Result<Self::Centroid, ()>;
}

impl<G> FaceCentroid for G
where
    G: Geometry,
    G::Vertex: Average,
{
    type Centroid = G::Vertex;

    fn centroid(face: FaceRef<Self>) -> Result<Self::Centroid, ()> {
        Ok(G::Vertex::average(
            face.vertices().map(|vertex| vertex.geometry.clone()),
        ))
    }
}

pub trait Normalize {
    fn normalize(self) -> Self;
}

pub trait Average: Sized {
    fn average<I>(values: I) -> Self
    where
        I: IntoIterator<Item = Self>;
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

pub mod alias {
    use std::ops::Mul;

    use super::*;

    pub type VertexPosition<G> =
        <<G as Geometry>::Vertex as AsPosition>::Target;
    pub type ScaledFaceNormal<G, F> =
        <<G as FaceNormal>::Normal as Mul<F>>::Output;
}

#[cfg(feature = "geometry-cgmath")]
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

    // TODO: Implement `Average` for points and vectors of arbitrary dimension.
    impl<T> Average for Point3<T>
    where
        T: AddAssign + MulAssign + Num + NumCast + Scalar,
    {
        fn average<I>(values: I) -> Self
        where
            I: IntoIterator<Item = Self>,
        {
            let values = values.into_iter().collect::<Vec<_>>();
            let n = <T as NumCast>::from(values.len()).unwrap();
            let sum = {
                let mut sum = Point3::origin();
                for point in values {
                    sum += Vector3::<T>::new(point.x, point.y, point.z);
                }
                sum
            };
            sum * (T::one() / n)
        }
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
