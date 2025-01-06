#![cfg(feature = "geometry-glam")]

use glam::{Vec2, Vec3, Vec3A};

use crate::geometry::{FromGeometry, UnitGeometry};
use crate::graph::GraphData;

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

impl FromGeometry<(f32, f32, f32)> for Vec3A {
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

impl GraphData for Vec3A {
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl UnitGeometry for Vec2 {}

impl UnitGeometry for Vec3 {}

impl UnitGeometry for Vec3A {}
