//! Sphere primitives.
//!
//! # Examples
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point3;
//! use plexus::graph::MeshGraph;
//! use plexus::prelude::*;
//! use plexus::primitive::index::HashIndexer;
//! use plexus::primitive::sphere::UvSphere;
//!
//! # fn main() {
//! let graph = UvSphere::new(16, 8)
//!     .polygons_with_position()
//!     .collect_with_indexer::<MeshGraph<Point3<f32>>, _>(HashIndexer::default())
//!     .unwrap();
//! # }
//! ```

use decorum::{Real, R64};
use num::traits::FloatConst;
use num::{NumCast, One};
use std::cmp;

use crate::geometry::Triplet;
use crate::primitive::generate::{
    PolygonGenerator, PositionGenerator, PositionIndexGenerator, PositionPolygonGenerator,
    PositionVertexGenerator, VertexGenerator,
};
use crate::primitive::topology::{Polygon, Quad, Triangle};

#[derive(Clone, Copy)]
pub struct Bounds {
    radius: R64,
}

impl Bounds {
    pub fn with_radius(radius: R64) -> Self {
        Bounds { radius }
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
        Self::unit_radius()
    }
}

#[derive(Clone, Copy)]
pub struct UvSphere {
    nu: usize, // Meridians.
    nv: usize, // Parallels.
}

impl UvSphere {
    pub fn new(nu: usize, nv: usize) -> Self {
        UvSphere {
            nu: cmp::max(3, nu),
            nv: cmp::max(2, nv),
        }
    }

    fn vertex_with_position_from(
        &self,
        state: &<Self as PositionGenerator>::State,
        u: usize,
        v: usize,
    ) -> <Self as PositionVertexGenerator>::Output {
        let u = (<R64 as NumCast>::from(u).unwrap() / <R64 as NumCast>::from(self.nu).unwrap())
            * R64::PI()
            * 2.0;
        let v = (<R64 as NumCast>::from(v).unwrap() / <R64 as NumCast>::from(self.nv).unwrap())
            * R64::PI();
        Triplet(
            state.radius * u.cos() * v.sin(),
            state.radius * u.sin() * v.sin(),
            state.radius * v.cos(),
        )
    }

    fn index_for_position(&self, u: usize, v: usize) -> usize {
        if v == 0 {
            0
        }
        else if v == self.nv {
            ((self.nv - 1) * self.nu) + 1
        }
        else {
            ((v - 1) * self.nu) + (u % self.nu) + 1
        }
    }

    fn map_polygon_index(&self, index: usize) -> (usize, usize) {
        (index % self.nu, index / self.nu)
    }
}

impl Default for UvSphere {
    fn default() -> Self {
        UvSphere::new(16, 16)
    }
}

impl VertexGenerator for UvSphere {}

impl PolygonGenerator for UvSphere {
    fn polygon_count(&self) -> usize {
        self.nu * self.nv
    }
}

impl PositionGenerator for UvSphere {
    type State = Bounds;
}

impl PositionVertexGenerator for UvSphere {
    type Output = Triplet<R64>;

    fn vertex_with_position_from(&self, state: &Self::State, index: usize) -> Self::Output {
        if index == 0 {
            self.vertex_with_position_from(state, 0, 0)
        }
        else if index == self.vertex_with_position_count() - 1 {
            self.vertex_with_position_from(state, 0, self.nv)
        }
        else {
            let index = index - 1;
            self.vertex_with_position_from(state, index % self.nu, (index / self.nu) + 1)
        }
    }

    fn vertex_with_position_count(&self) -> usize {
        (self.nv - 1) * self.nu + 2
    }
}

impl PositionPolygonGenerator for UvSphere {
    type Output = Polygon<Triplet<R64>>;

    fn polygon_with_position_from(&self, state: &Self::State, index: usize) -> Self::Output {
        // Prevent floating point rounding errors by wrapping the incremented
        // values for `(u, v)` into `(p, q)`. This is important for indexing
        // geometry, because small differences in the computation of spatial
        // vertices will produce redundant output vertices. There should be
        // exactly `(nv - 1) * nu + 2` unique values of `(u, v)` used to
        // generate positions.
        //
        // There are two important observations:
        //
        //   1. `u` must wrap, but `v` need not. There are `nu` meridians of
        //      points and polygons, but there are `nv` parallels of polygons
        //      and `nv + 1` parallels of points.
        //   2. `u`, which represents a meridian, is meaningless at the poles,
        //      and can be normalized to zero.
        let (u, v) = self.map_polygon_index(index);
        let (p, q) = ((u + 1) % self.nu, v + 1);

        // Generate the vertices at the requested meridian and parallel. The
        // lower bound of `(u, v)` is always used, so compute that in advance
        // (`lower`). Emit triangles at the poles, otherwise quads.
        let lower = self.vertex_with_position_from(state, u, v);
        if v == 0 {
            Polygon::Triangle(Triangle::new(
                lower,
                self.vertex_with_position_from(state, u, q),
                self.vertex_with_position_from(state, p, q),
            ))
        }
        else if v == self.nv - 1 {
            Polygon::Triangle(Triangle::new(
                // Normalize `u` at the pole, using `(0, nv)` in place of
                // `(p, q)`.
                self.vertex_with_position_from(state, 0, self.nv),
                self.vertex_with_position_from(state, p, v),
                lower,
            ))
        }
        else {
            Polygon::Quad(Quad::new(
                lower,
                self.vertex_with_position_from(state, u, q),
                self.vertex_with_position_from(state, p, q),
                self.vertex_with_position_from(state, p, v),
            ))
        }
    }
}

impl PositionIndexGenerator for UvSphere {
    type Output = Polygon<usize>;

    fn index_for_position(&self, index: usize) -> <Self as PositionIndexGenerator>::Output {
        let (u, v) = self.map_polygon_index(index);
        let (p, q) = (u + 1, v + 1);

        let low = self.index_for_position(u, v);
        let high = self.index_for_position(p, q);
        if v == 0 {
            Polygon::Triangle(Triangle::new(low, self.index_for_position(u, q), high))
        }
        else if v == self.nv - 1 {
            Polygon::Triangle(Triangle::new(high, self.index_for_position(p, v), low))
        }
        else {
            Polygon::Quad(Quad::new(
                low,
                self.index_for_position(u, q),
                high,
                self.index_for_position(p, v),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::iter::FromIterator;

    use crate::primitive::decompose::*;
    use crate::primitive::generate::*;
    use crate::primitive::sphere::UvSphere;

    #[test]
    fn vertex_count() {
        assert_eq!(
            5,
            UvSphere::new(3, 2)
                .vertices_with_position() // 5 conjoint vertices.
                .count()
        );
    }

    #[test]
    fn polygon_vertex_count() {
        assert_eq!(
            18,
            UvSphere::new(3, 2)
                .polygons_with_position() // 6 triangles, 18 vertices.
                .vertices()
                .count()
        );
    }

    #[test]
    fn position_index_to_vertex_mapping() {
        assert_eq!(
            5,
            BTreeSet::from_iter(
                UvSphere::new(3, 2)
                    .indices_for_position() // 18 vertices, 5 indices.
                    .vertices()
            )
            .len()
        )
    }
}
