//! Cube primitives.
//!
//! # Examples
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point3;
//! use plexus::graph::MeshGraph;
//! use plexus::index::HashIndexer;
//! use plexus::prelude::*;
//! use plexus::primitive::cube::Cube;
//!
//! # fn main() {
//! let graph = Cube::new()
//!     .polygons_with_position()
//!     .collect_with_indexer::<MeshGraph<Point3<f32>>, _>(HashIndexer::default())
//!     .unwrap();
//! # }
//! ```

use decorum::R64;
use num::{One, Zero};

use crate::geometry::{Duplet, Triplet};
use crate::primitive::generate::{
    Generate, NormalGenerator, NormalIndexGenerator, NormalPolygonGenerator, NormalVertexGenerator,
    PolygonGenerator, PositionGenerator, PositionIndexGenerator, PositionPolygonGenerator,
    PositionVertexGenerator, UvMapGenerator, UvMapPolygonGenerator,
};
use crate::primitive::{Converged, Map, Quad};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Plane {
    XY,
    NXY,
    ZY,
    NZY,
    XZ,
    XNZ,
}

impl Plane {
    pub fn normal(&self) -> Triplet<R64> {
        let zero = R64::zero();
        let one = R64::one();
        match *self {
            Plane::XY => Triplet(zero, zero, one),   // front
            Plane::NXY => Triplet(zero, one, zero),  // right
            Plane::ZY => Triplet(-one, zero, zero),  // top
            Plane::NZY => Triplet(zero, -one, zero), // left
            Plane::XZ => Triplet(one, zero, zero),   // bottom
            Plane::XNZ => Triplet(zero, zero, -one), // back
        }
    }
}

#[derive(Clone, Copy)]
pub struct Bounds {
    lower: R64,
    upper: R64,
}

impl Bounds {
    pub fn with_radius(radius: R64) -> Self {
        Bounds {
            lower: -radius,
            upper: radius,
        }
    }

    pub fn with_width(width: R64) -> Self {
        Self::with_radius(width / 2.0)
    }

    pub fn unit_radius() -> Self {
        Self::with_radius(One::one())
    }

    pub fn unit_width() -> Self {
        Self::with_width(One::one())
    }
}

impl Default for Bounds {
    fn default() -> Self {
        Self::unit_width()
    }
}

#[derive(Clone, Copy)]
pub struct Cube;

impl Cube {
    pub fn new() -> Self {
        Cube
    }

    pub fn polygons_with_plane(&self) -> Generate<Self, (), Quad<Plane>> {
        Generate::new(self, (), self.polygon_count(), |generator, _, index| {
            generator.polygon_with_plane(index)
        })
    }

    pub fn vertex_with_plane_count(&self) -> usize {
        self.polygon_count()
    }

    pub fn vertex_with_plane(&self, index: usize) -> Plane {
        match index {
            0 => Plane::XY,  // front
            1 => Plane::NZY, // right
            2 => Plane::XNZ, // top
            3 => Plane::ZY,  // left
            4 => Plane::XZ,  // bottom
            5 => Plane::NXY, // back
            _ => panic!(),
        }
    }

    pub fn polygon_with_plane(&self, index: usize) -> Quad<Plane> {
        Quad::converged(self.vertex_with_plane(index))
    }
}

impl Default for Cube {
    fn default() -> Self {
        Cube::new()
    }
}

impl PolygonGenerator for Cube {
    fn polygon_count(&self) -> usize {
        6
    }
}

impl NormalGenerator for Cube {
    type State = ();
}

impl NormalVertexGenerator for Cube {
    type Output = Triplet<R64>;

    fn vertex_with_normal_from(&self, _: &Self::State, index: usize) -> Self::Output {
        // There is a unique normal for each face (plane).
        self.vertex_with_plane(index).normal()
    }

    fn vertex_with_normal_count(&self) -> usize {
        self.polygon_count()
    }
}

impl NormalPolygonGenerator for Cube {
    type Output = Quad<<Self as NormalVertexGenerator>::Output>;

    fn polygon_with_normal_from(&self, state: &Self::State, index: usize) -> Self::Output {
        self.index_for_normal(index)
            .map(|index| self.vertex_with_normal_from(state, index))
    }
}

impl NormalIndexGenerator for Cube {
    type Output = Quad<usize>;

    fn index_for_normal(&self, index: usize) -> <Self as PositionIndexGenerator>::Output {
        assert!(index < self.polygon_count());
        Quad::converged(index)
    }
}

impl PositionGenerator for Cube {
    type State = Bounds;
}

impl PositionVertexGenerator for Cube {
    type Output = Triplet<R64>;

    fn vertex_with_position_from(&self, state: &Self::State, index: usize) -> Self::Output {
        let x = if index & 0b100 == 0b100 {
            state.upper
        }
        else {
            state.lower
        };
        let y = if index & 0b010 == 0b010 {
            state.upper
        }
        else {
            state.lower
        };
        let z = if index & 0b001 == 0b001 {
            state.upper
        }
        else {
            state.lower
        };
        Triplet(x, y, z)
    }

    fn vertex_with_position_count(&self) -> usize {
        8
    }
}

impl PositionPolygonGenerator for Cube {
    type Output = Quad<Triplet<R64>>;

    fn polygon_with_position_from(&self, state: &Self::State, index: usize) -> Self::Output {
        self.index_for_position(index)
            .map(|index| self.vertex_with_position_from(state, index))
    }
}

impl PositionIndexGenerator for Cube {
    type Output = Quad<usize>;

    fn index_for_position(&self, index: usize) -> <Self as PositionIndexGenerator>::Output {
        match index {
            0 => Quad::new(5, 7, 3, 1), // front
            1 => Quad::new(6, 7, 5, 4), // right
            2 => Quad::new(3, 7, 6, 2), // top
            3 => Quad::new(0, 1, 3, 2), // left
            4 => Quad::new(4, 5, 1, 0), // bottom
            5 => Quad::new(0, 2, 6, 4), // back
            _ => panic!(),
        }
    }
}

impl UvMapGenerator for Cube {
    type State = ();
}

impl UvMapPolygonGenerator for Cube {
    type Output = Quad<Duplet<R64>>;

    fn polygon_with_uv_map_from(&self, _: &Self::State, index: usize) -> Self::Output {
        let uu = Duplet::one();
        let ul = Duplet(One::one(), Zero::zero());
        let ll = Duplet::zero();
        let lu = Duplet(Zero::zero(), One::one());
        match index {
            0 | 4 | 5 => Quad::new(uu, ul, ll, lu), // front | bottom | back
            1 => Quad::new(ul, ll, lu, uu),         // right
            2 | 3 => Quad::new(lu, uu, ul, ll),     // top | left
            _ => panic!(),
        }
    }
}
