#![cfg(feature = "geometry-ultraviolet")]

use theon::integration::ultraviolet;

use ultraviolet::vec::{Vec2, Vec3};

use crate::geometry::{FromGeometry, UnitGeometry};
use crate::graph::GraphData;

#[doc(hidden)]
pub use self::ultraviolet::*;

impl FromGeometry<(f32, f32)> for Vec2 {
    fn from_geometry(other: (f32, f32)) -> Self {
        Self::from(other)
    }
}

impl FromGeometry<(f32, f32, f32)> for Vec3 {
    fn from_geometry(other: (f32, f32, f32)) -> Self {
        Self::from(other)
    }
}

impl GraphData for Vec2 {
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl GraphData for Vec3 {
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl UnitGeometry for Vec2 {}

impl UnitGeometry for Vec3 {}
