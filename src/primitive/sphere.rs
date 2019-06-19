//! Sphere primitives.
//!
//! # Examples
//!
//! ```rust
//! # extern crate decorum;
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use decorum::N64;
//! use nalgebra::Point3;
//! use plexus::graph::MeshGraph;
//! use plexus::index::HashIndexer;
//! use plexus::prelude::*;
//! use plexus::primitive::sphere::UvSphere;
//!
//! # fn main() {
//! let mut graph = UvSphere::new(16, 8)
//!     .polygons_with_position::<Point3<N64>>()
//!     .collect_with_indexer::<MeshGraph<Point3<N64>>, _>(HashIndexer::default())
//!     .unwrap();
//! # }
//! ```

use decorum::Real;
use num::traits::FloatConst;
use num::{NumCast, One, ToPrimitive};
use std::cmp;
use theon::space::{EuclideanSpace, FiniteDimensional, Scalar};
use typenum::U3;

use crate::primitive::generate::{
    PolygonGenerator, PolygonsWithPosition, PositionGenerator, PositionIndexGenerator,
    PositionPolygonGenerator, PositionVertexGenerator, VerticesWithPosition,
};
use crate::primitive::Polygon;

#[derive(Clone, Copy)]
pub struct Bounds<S>
where
    S: EuclideanSpace,
{
    radius: Scalar<S>,
}

impl<S> Bounds<S>
where
    S: EuclideanSpace,
{
    pub fn with_radius(radius: Scalar<S>) -> Self {
        Bounds { radius }
    }

    pub fn with_width(width: Scalar<S>) -> Self {
        Self::with_radius(width / (Scalar::<S>::one() + One::one()))
    }

    pub fn unit_radius() -> Self {
        Self::with_radius(One::one())
    }

    pub fn unit_width() -> Self {
        Self::with_width(One::one())
    }
}

impl<S> Default for Bounds<S>
where
    S: EuclideanSpace,
{
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

    fn vertex_with_position_from<S>(
        &self,
        state: &<Self as PositionGenerator<S>>::State,
        u: usize,
        v: usize,
    ) -> S
    where
        Self: PositionGenerator<S, State = Bounds<S>>,
        S: EuclideanSpace + FiniteDimensional<N = U3>,
        Scalar<S>: FloatConst,
    {
        let one = Scalar::<S>::one();
        let pi = Scalar::<S>::PI();
        let u = (into_scalar::<_, S>(u) / into_scalar::<_, S>(self.nu)) * pi * (one + one);
        let v = (into_scalar::<_, S>(v) / into_scalar::<_, S>(self.nv)) * pi;
        S::from_xyz(
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

impl PolygonGenerator for UvSphere {
    fn polygon_count(&self) -> usize {
        self.nu * self.nv
    }
}

impl<S> PositionGenerator<S> for UvSphere
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
{
    type State = Bounds<S>;
}

impl<S> PositionVertexGenerator<S> for UvSphere
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
    Scalar<S>: FloatConst,
{
    fn vertex_with_position_from(&self, state: &Self::State, index: usize) -> S {
        let count = <Self as PositionVertexGenerator<S>>::vertex_with_position_count(self);
        if index == 0 {
            self.vertex_with_position_from::<S>(state, 0, 0)
        }
        else if index == (count - 1) {
            self.vertex_with_position_from::<S>(state, 0, self.nv)
        }
        else {
            let index = index - 1;
            self.vertex_with_position_from::<S>(state, index % self.nu, (index / self.nu) + 1)
        }
    }

    fn vertex_with_position_count(&self) -> usize {
        (self.nv - 1) * self.nu + 2
    }
}

impl<S> PositionPolygonGenerator<S> for UvSphere
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
    Scalar<S>: FloatConst,
{
    type Output = Polygon<S>;

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
            Polygon::triangle(
                lower,
                self.vertex_with_position_from(state, u, q),
                self.vertex_with_position_from(state, p, q),
            )
        }
        else if v == self.nv - 1 {
            Polygon::triangle(
                // Normalize `u` at the pole, using `(0, nv)` in place of
                // `(p, q)`.
                self.vertex_with_position_from(state, 0, self.nv),
                self.vertex_with_position_from(state, p, v),
                lower,
            )
        }
        else {
            Polygon::quad(
                lower,
                self.vertex_with_position_from(state, u, q),
                self.vertex_with_position_from(state, p, q),
                self.vertex_with_position_from(state, p, v),
            )
        }
    }
}

impl PositionIndexGenerator for UvSphere {
    type Output = Polygon<usize>;

    fn index_for_position(&self, index: usize) -> Self::Output {
        let (u, v) = self.map_polygon_index(index);
        let (p, q) = (u + 1, v + 1);

        let low = self.index_for_position(u, v);
        let high = self.index_for_position(p, q);
        if v == 0 {
            Polygon::triangle(low, self.index_for_position(u, q), high)
        }
        else if v == self.nv - 1 {
            Polygon::triangle(high, self.index_for_position(p, v), low)
        }
        else {
            Polygon::quad(
                low,
                self.index_for_position(u, q),
                high,
                self.index_for_position(p, v),
            )
        }
    }
}

impl VerticesWithPosition for UvSphere {}

impl PolygonsWithPosition for UvSphere {}

fn into_scalar<T, S>(value: T) -> Scalar<S>
where
    T: ToPrimitive,
    S: EuclideanSpace,
{
    <Scalar<S> as NumCast>::from(value).unwrap()
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;
    use std::collections::BTreeSet;
    use std::iter::FromIterator;

    use crate::prelude::*;
    use crate::primitive::sphere::UvSphere;

    type E3 = Point3<f64>;

    #[test]
    fn vertex_count() {
        assert_eq!(
            5,
            UvSphere::new(3, 2)
                .vertices_with_position::<E3>() // 5 conjoint vertices.
                .count()
        );
    }

    #[test]
    fn polygon_vertex_count() {
        assert_eq!(
            18,
            UvSphere::new(3, 2)
                .polygons_with_position::<E3>() // 6 triangles, 18 vertices.
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
