//! Cube primitives.
//!
//! # Examples
//!
//! ```rust
//! # extern crate decorum;
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use decorum::R32;
//! use nalgebra::Point3;
//! use plexus::graph::MeshGraph;
//! use plexus::prelude::*;
//! use plexus::primitive::cube::Cube;
//! use plexus::primitive::generate::Position;
//!
//! let mut graph = Cube::new()
//!     .polygons::<Position<Point3<R32>>>()
//!     .collect::<MeshGraph<Point3<R32>>>();
//! ```

use num::One;
use theon::adjunct::{Converged, Map};
use theon::query::Unit;
use theon::space::{Basis, EuclideanSpace, FiniteDimensional, InnerSpace, Scalar, Vector};
use typenum::U3;

use crate::primitive::generate::{
    Attribute, AttributeGenerator, AttributePolygonGenerator, AttributeVertexGenerator, Generator,
    IndexingPolygonGenerator, Normal, PolygonGenerator, Position,
};
use crate::primitive::Tetragon;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Plane {
    Xy,  // front
    Nxy, // back
    Yz,  // right
    Nyz, // left
    Xz,  // bottom
    Nxz, // top
}

impl Plane {
    pub fn normal<S>(self) -> Unit<S>
    where
        S: Basis + FiniteDimensional<N = U3> + InnerSpace,
    {
        match self {
            Plane::Xy => Unit::<S>::z(),   // front
            Plane::Nxy => -Unit::<S>::z(), // back
            Plane::Yz => Unit::<S>::x(),   // right
            Plane::Nyz => -Unit::<S>::x(), // left
            Plane::Xz => -Unit::<S>::y(),  // bottom
            Plane::Nxz => Unit::<S>::y(),  // top
        }
    }
}

impl Attribute for Plane {}

#[derive(Clone, Copy)]
pub struct Bounds<S>
where
    S: EuclideanSpace,
{
    lower: Scalar<S>,
    upper: Scalar<S>,
}

impl<S> Bounds<S>
where
    S: EuclideanSpace,
{
    pub fn with_radius(radius: Scalar<S>) -> Self {
        Bounds {
            lower: -radius,
            upper: radius,
        }
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
        Self::unit_width()
    }
}

#[derive(Clone, Copy)]
pub struct Cube;

impl Cube {
    pub fn new() -> Self {
        Cube
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

impl<S> AttributeGenerator<Normal<S>> for Cube
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
{
    type State = ();
}

impl<S> AttributeVertexGenerator<Normal<S>> for Cube
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
{
    type Output = Unit<Vector<S>>;

    fn vertex_count(&self) -> usize {
        self.polygon_count()
    }

    fn vertex_from(&self, _: &Self::State, index: usize) -> Self::Output {
        AttributeVertexGenerator::<Plane>::vertex_from(self, &(), index).normal::<Vector<S>>()
    }
}

impl<S> AttributePolygonGenerator<Normal<S>> for Cube
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
{
    type Output = Tetragon<Unit<Vector<S>>>;

    fn polygon_from(&self, state: &Self::State, index: usize) -> Self::Output {
        IndexingPolygonGenerator::<Normal<S>>::indexing_polygon(self, index)
            .map(|index| AttributeVertexGenerator::<Normal<S>>::vertex_from(self, state, index))
    }
}

impl<S> IndexingPolygonGenerator<Normal<S>> for Cube {
    type Output = Tetragon<usize>;

    fn indexing_polygon(&self, index: usize) -> Self::Output {
        assert!(index < self.polygon_count());
        Tetragon::converged(index)
    }
}

impl<S> AttributeGenerator<Position<S>> for Cube
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
{
    type State = Bounds<S>;
}

impl<S> AttributeVertexGenerator<Position<S>> for Cube
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
{
    type Output = S;

    fn vertex_count(&self) -> usize {
        8
    }

    fn vertex_from(&self, state: &Self::State, index: usize) -> Self::Output {
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
        S::from_xyz(x, y, z)
    }
}

impl<S> AttributePolygonGenerator<Position<S>> for Cube
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
{
    type Output = Tetragon<S>;

    fn polygon_from(&self, state: &Self::State, index: usize) -> Self::Output {
        IndexingPolygonGenerator::<Position<S>>::indexing_polygon(self, index)
            .map(|index| AttributeVertexGenerator::<Position<S>>::vertex_from(self, state, index))
    }
}

impl<S> IndexingPolygonGenerator<Position<S>> for Cube {
    type Output = Tetragon<usize>;

    fn indexing_polygon(&self, index: usize) -> Self::Output {
        match index {
            0 => Tetragon::new(5, 7, 3, 1), // front
            1 => Tetragon::new(6, 7, 5, 4), // right
            2 => Tetragon::new(3, 7, 6, 2), // top
            3 => Tetragon::new(0, 1, 3, 2), // left
            4 => Tetragon::new(4, 5, 1, 0), // bottom
            5 => Tetragon::new(0, 2, 6, 4), // back
            _ => panic!(),
        }
    }
}

impl AttributeGenerator<Plane> for Cube {
    type State = ();
}

impl AttributeVertexGenerator<Plane> for Cube {
    type Output = Plane;

    fn vertex_count(&self) -> usize {
        self.polygon_count()
    }

    fn vertex_from(&self, _: &Self::State, index: usize) -> Self::Output {
        match index {
            0 => Plane::Xy,  // front
            1 => Plane::Yz,  // right
            2 => Plane::Nxz, // top
            3 => Plane::Nyz, // left
            4 => Plane::Xz,  // bottom
            5 => Plane::Nxy, // back
            _ => panic!(),
        }
    }
}

impl AttributePolygonGenerator<Plane> for Cube {
    type Output = Tetragon<Plane>;

    fn polygon_from(&self, state: &Self::State, index: usize) -> Self::Output {
        IndexingPolygonGenerator::<Plane>::indexing_polygon(self, index)
            .map(|index| AttributeVertexGenerator::<Plane>::vertex_from(self, state, index))
    }
}

impl IndexingPolygonGenerator<Plane> for Cube {
    type Output = Tetragon<usize>;

    fn indexing_polygon(&self, index: usize) -> Self::Output {
        match index {
            0 => Tetragon::converged(0), // front
            1 => Tetragon::converged(1), // right
            2 => Tetragon::converged(2), // top
            3 => Tetragon::converged(3), // left
            4 => Tetragon::converged(4), // bottom
            5 => Tetragon::converged(5), // back
            _ => panic!(),
        }
    }
}

impl Generator for Cube {}
