//! Sphere primitives.
//!
//! # Examples
//!
//! Generating a graph from the positional data of a $uv$-sphere.
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
//! use plexus::primitive::generate::Position;
//! use plexus::primitive::sphere::UvSphere;
//!
//! type E3 = Point3<N64>;
//!
//! let mut graph = UvSphere::new(16, 8)
//!     .polygons::<Position<E3>>()
//!     .collect_with_indexer::<MeshGraph<E3>, _>(HashIndexer::default())
//!     .unwrap();
//! ```

use decorum::Real;
use num::traits::FloatConst;
use num::{NumCast, One, ToPrimitive};
use std::cmp;
use theon::ops::Map;
use theon::query::Unit;
use theon::space::{EuclideanSpace, FiniteDimensional, Scalar, Vector};
use typenum::U3;

use crate::primitive::generate::{
    AttributeGenerator, AttributePolygonGenerator, AttributeVertexGenerator, Generator,
    IndexingPolygonGenerator, Normal, PolygonGenerator, Position,
};
use crate::primitive::{Polygon, Tetragon, Trigon};

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
        state: &<Self as AttributeGenerator<Position<S>>>::State,
        u: usize,
        v: usize,
    ) -> S
    where
        Self: AttributeGenerator<Position<S>, State = Bounds<S>>,
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

impl<S> AttributeGenerator<Normal<S>> for UvSphere
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
{
    type State = ();
}

impl<S> AttributeVertexGenerator<Normal<S>> for UvSphere
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
    Scalar<S>: FloatConst,
{
    type Output = Unit<Vector<S>>;

    fn vertex_count(&self) -> usize {
        (self.nv - 1) * self.nu + 2
    }

    fn vertex_from(&self, _: &Self::State, index: usize) -> Self::Output {
        let position =
            AttributeVertexGenerator::<Position<S>>::vertex_from(self, &Default::default(), index);
        Unit::try_from_inner(position.into_coordinates()).expect("non-zero vector")
    }
}

impl<S> AttributePolygonGenerator<Normal<S>> for UvSphere
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
    Scalar<S>: FloatConst,
{
    type Output = Polygon<Unit<Vector<S>>>;

    fn polygon_from(&self, _: &Self::State, index: usize) -> Self::Output {
        AttributePolygonGenerator::<Position<S>>::polygon_from(self, &Default::default(), index)
            .map(|position| {
                Unit::try_from_inner(position.into_coordinates()).expect("non-zero vector")
            })
    }
}

impl<S> IndexingPolygonGenerator<Normal<S>> for UvSphere {
    type Output = Polygon<usize>;

    fn indexing_polygon(&self, index: usize) -> Self::Output {
        IndexingPolygonGenerator::<Position<S>>::indexing_polygon(self, index)
    }
}

impl<S> AttributeGenerator<Position<S>> for UvSphere
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
{
    type State = Bounds<S>;
}

impl<S> AttributeVertexGenerator<Position<S>> for UvSphere
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
    Scalar<S>: FloatConst,
{
    type Output = S;

    fn vertex_count(&self) -> usize {
        (self.nv - 1) * self.nu + 2
    }

    fn vertex_from(&self, state: &Self::State, index: usize) -> Self::Output {
        let count = AttributeVertexGenerator::<Position<S>>::vertex_count(self);
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
}

impl<S> AttributePolygonGenerator<Position<S>> for UvSphere
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
    Scalar<S>: FloatConst,
{
    type Output = Polygon<S>;

    fn polygon_from(&self, state: &Self::State, index: usize) -> Self::Output {
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
        // (`lower`). Emit triangles at the poles, otherwise quadrilaterals.
        let lower = self.vertex_with_position_from(state, u, v);
        if v == 0 {
            Trigon::new(
                lower,
                self.vertex_with_position_from(state, u, q),
                self.vertex_with_position_from(state, p, q),
            )
            .into()
        }
        else if v == self.nv - 1 {
            Trigon::new(
                // Normalize `u` at the pole, using `(0, nv)` in place of
                // `(p, q)`.
                self.vertex_with_position_from(state, 0, self.nv),
                self.vertex_with_position_from(state, p, v),
                lower,
            )
            .into()
        }
        else {
            Tetragon::new(
                lower,
                self.vertex_with_position_from(state, u, q),
                self.vertex_with_position_from(state, p, q),
                self.vertex_with_position_from(state, p, v),
            )
            .into()
        }
    }
}

impl<S> IndexingPolygonGenerator<Position<S>> for UvSphere {
    type Output = Polygon<usize>;

    fn indexing_polygon(&self, index: usize) -> Self::Output {
        let (u, v) = self.map_polygon_index(index);
        let (p, q) = (u + 1, v + 1);

        let low = self.index_for_position(u, v);
        let high = self.index_for_position(p, q);
        if v == 0 {
            Trigon::new(low, self.index_for_position(u, q), high).into()
        }
        else if v == self.nv - 1 {
            Trigon::new(high, self.index_for_position(p, v), low).into()
        }
        else {
            Tetragon::new(
                low,
                self.index_for_position(u, q),
                high,
                self.index_for_position(p, v),
            )
            .into()
        }
    }
}

impl Generator for UvSphere {}

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
    use crate::primitive::generate::Position;
    use crate::primitive::sphere::UvSphere;

    type E3 = Point3<f64>;

    #[test]
    fn vertex_count() {
        assert_eq!(
            5,
            UvSphere::new(3, 2)
                .vertices::<Position<E3>>() // 5 conjoint vertices.
                .count()
        );
    }

    #[test]
    fn polygon_vertex_count() {
        assert_eq!(
            18,
            UvSphere::new(3, 2)
                .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
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
                    .indexing_polygons::<Position>() // 18 vertices, 5 indices.
                    .vertices()
            )
            .len()
        )
    }
}
