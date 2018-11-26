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

use geometry::Triplet;
use primitive::generate::{
    IndexGenerator, IndexPolygonGenerator, PolygonGenerator, PositionGenerator,
    PositionPolygonGenerator, PositionVertexGenerator, VertexGenerator,
};
use primitive::topology::{Polygon, Quad, Triangle};
use primitive::Half;

#[derive(Clone, Copy)]
pub struct Bounds {
    radius: R64,
}

impl Bounds {
    pub fn unit_radius() -> Self {
        Bounds { radius: R64::one() }
    }

    pub fn unit_width() -> Self {
        Bounds {
            radius: R64::half(),
        }
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
    ) -> Triplet<R64> {
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

    fn vertex_with_index_from(
        &self,
        _: &<Self as IndexGenerator>::State,
        u: usize,
        v: usize,
    ) -> usize {
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

impl VertexGenerator for UvSphere {
    fn vertex_count(&self) -> usize {
        (self.nv - 1) * self.nu + 2
    }
}

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
        else if index == self.vertex_count() - 1 {
            self.vertex_with_position_from(state, 0, self.nv)
        }
        else {
            let index = index - 1;
            self.vertex_with_position_from(state, index % self.nu, (index / self.nu) + 1)
        }
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

impl IndexGenerator for UvSphere {
    type State = ();
}

impl IndexPolygonGenerator for UvSphere {
    type Output = Polygon<usize>;

    fn polygon_with_index_from(
        &self,
        state: &Self::State,
        index: usize,
    ) -> <Self as IndexPolygonGenerator>::Output {
        let (u, v) = self.map_polygon_index(index);
        let (p, q) = (u + 1, v + 1);

        let low = self.vertex_with_index_from(state, u, v);
        let high = self.vertex_with_index_from(state, p, q);
        if v == 0 {
            Polygon::Triangle(Triangle::new(
                low,
                self.vertex_with_index_from(state, u, q),
                high,
            ))
        }
        else if v == self.nv - 1 {
            Polygon::Triangle(Triangle::new(
                high,
                self.vertex_with_index_from(state, p, v),
                low,
            ))
        }
        else {
            Polygon::Quad(Quad::new(
                low,
                self.vertex_with_index_from(state, u, q),
                high,
                self.vertex_with_index_from(state, p, v),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::iter::FromIterator;

    use primitive::decompose::*;
    use primitive::generate::*;
    use primitive::sphere::UvSphere;

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
    fn index_to_vertex_mapping() {
        assert_eq!(
            5,
            BTreeSet::from_iter(
                UvSphere::new(3, 2)
                    .polygons_with_index() // 18 vertices, 5 indices.
                    .vertices()
            )
            .len()
        )
    }
}
