// Feature modules. These are empty unless features are enabled.
mod cgmath;
mod mint;
mod nalgebra;

/// Exposes positional data in vertex geometry.
///
/// To enable geometric features, this trait must be implemented for the type
/// representing vertex data. Additionally, geometric operations should be
/// implemented for the `Target` type. See the
/// [theon](https://crates.io/crates/theon) crate.
pub trait AsPosition {
    type Target;

    fn as_position(&self) -> &Self::Target;
    fn as_position_mut(&mut self) -> &mut Self::Target;
}

pub trait FromGeometry<T> {
    fn from_geometry(other: T) -> Self;
}

impl<T> FromGeometry<T> for T {
    fn from_geometry(other: T) -> Self {
        other
    }
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
