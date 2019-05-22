//! Primitive generation.
//!
//! This module provides a generic iterator and traits for generating streams
//! of geometric and topological data for primitives like cubes and spheres.

use std::ops::Range;
use theon::space::{EuclideanSpace, FiniteDimensional, Vector};
use typenum::{U2, U3};

use crate::primitive::Polygonal;

pub struct Generate<'a, G, S, P>
where
    G: 'a,
{
    generator: &'a G,
    state: S,
    range: Range<usize>,
    f: fn(&'a G, &S, usize) -> P,
}

impl<'a, G, S, P> Generate<'a, G, S, P>
where
    G: 'a,
{
    pub(in crate::primitive) fn new(
        generator: &'a G,
        state: S,
        n: usize,
        f: fn(&'a G, &S, usize) -> P,
    ) -> Self {
        Generate {
            generator,
            state,
            range: 0..n,
            f,
        }
    }
}

impl<'a, G, S, P> Iterator for Generate<'a, G, S, P>
where
    G: 'a,
{
    type Item = P;

    fn next(&mut self) -> Option<Self::Item> {
        self.range
            .next()
            .map(|index| (self.f)(self.generator, &self.state, index))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

pub trait PolygonGenerator {
    fn polygon_count(&self) -> usize;
}

pub trait NormalGenerator<S>
where
    S: EuclideanSpace,
    <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U3>,
{
    type State: Default;
}

pub trait NormalVertexGenerator<S>: NormalGenerator<S>
where
    S: EuclideanSpace,
    <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U3>,
{
    fn vertex_with_normal_from(&self, state: &Self::State, index: usize) -> Vector<S>;

    /// Gets the number of unique vertices with normal data that comprise a primitive.
    fn vertex_with_normal_count(&self) -> usize;
}

/// Functions for generating vertices with normal data.
pub trait VerticesWithNormal: Sized {
    fn vertices_with_normal<S>(&self) -> Generate<Self, Self::State, Vector<S>>
    where
        Self: NormalVertexGenerator<S>,
        S: EuclideanSpace,
        <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U3>,
    {
        self.vertices_with_normal_from(Default::default())
    }

    fn vertices_with_normal_from<S>(
        &self,
        state: Self::State,
    ) -> Generate<Self, Self::State, Vector<S>>
    where
        Self: NormalVertexGenerator<S>,
        S: EuclideanSpace,
        <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U3>,
    {
        Generate::new(
            self,
            state,
            self.vertex_with_normal_count(),
            Self::vertex_with_normal_from,
        )
    }
}

pub trait NormalIndexGenerator: PolygonGenerator {
    type Output: Polygonal<Vertex = usize>;

    fn index_for_normal(&self, index: usize) -> Self::Output;
}

pub trait IndicesForNormal: NormalIndexGenerator + Sized {
    fn indices_for_normal(&self) -> Generate<Self, (), Self::Output> {
        Generate::new(self, (), self.polygon_count(), |generator, _, index| {
            generator.index_for_normal(index)
        })
    }
}

impl<G> IndicesForNormal for G where G: NormalIndexGenerator + Sized {}

pub trait NormalPolygonGenerator<S>: PolygonGenerator + NormalGenerator<S>
where
    S: EuclideanSpace,
    <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U3>,
{
    type Output: Polygonal<Vertex = Vector<S>>;

    fn polygon_with_normal_from(&self, state: &Self::State, index: usize) -> Self::Output;
}

/// Functions for generating polygons with normal data.
pub trait PolygonsWithNormal: Sized {
    fn polygons_with_normal<S>(
        &self,
    ) -> Generate<Self, Self::State, <Self as NormalPolygonGenerator<S>>::Output>
    where
        Self: NormalPolygonGenerator<S>,
        S: EuclideanSpace,
        <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U3>,
    {
        self.polygons_with_normal_from(Default::default())
    }

    fn polygons_with_normal_from<S>(
        &self,
        state: Self::State,
    ) -> Generate<Self, Self::State, <Self as NormalPolygonGenerator<S>>::Output>
    where
        Self: NormalPolygonGenerator<S>,
        S: EuclideanSpace,
        <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U3>,
    {
        Generate::new(
            self,
            state,
            self.polygon_count(),
            Self::polygon_with_normal_from,
        )
    }
}

pub trait PositionGenerator<S>
where
    S: EuclideanSpace,
{
    type State: Default;
}

pub trait PositionVertexGenerator<S>: PositionGenerator<S>
where
    S: EuclideanSpace,
{
    fn vertex_with_position_from(&self, state: &Self::State, index: usize) -> S;

    /// Gets the number of unique vertices with position data that comprise a primitive.
    fn vertex_with_position_count(&self) -> usize;
}

/// Functions for generating vertices with position data.
pub trait VerticesWithPosition: Sized {
    /// Provides an iterator over the set of unique vertices with position
    /// data.
    ///
    /// This can be paired with functions from `IndicesForPosition` to index
    /// the set of positions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point3;
    /// use plexus::prelude::*;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// # fn main() {
    /// let sphere = UvSphere::new(8, 8);
    ///
    /// let positions = sphere
    ///     .vertices_with_position::<Point3<f64>>()
    ///     .collect::<Vec<_>>();
    /// let indices = sphere
    ///     .indices_for_position()
    ///     .triangulate()
    ///     .vertices()
    ///     .collect::<Vec<_>>();
    /// # }
    /// ```
    fn vertices_with_position<S>(&self) -> Generate<Self, Self::State, S>
    where
        Self: PositionVertexGenerator<S>,
        S: EuclideanSpace,
    {
        self.vertices_with_position_from(Default::default())
    }

    /// Provides an iterator over the set of unique vertices with position data
    /// using the provided state.
    ///
    /// This can be paired with functions from `IndicesForPosition` to index
    /// the set of positions.
    ///
    /// State typically dictates the scale of the generated positions, using a
    /// unit width, radius, etc.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point3;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::{Bounds, Cube};
    ///
    /// # fn main() {
    /// let cube = Cube::new();
    ///
    /// let positions = cube
    ///     .vertices_with_position_from::<Point3<f64>>(Bounds::with_width(3.0))
    ///     .collect::<Vec<_>>();
    /// let indices = cube
    ///     .indices_for_position()
    ///     .triangulate()
    ///     .vertices()
    ///     .collect::<Vec<_>>();
    /// # }
    /// ```
    fn vertices_with_position_from<S>(&self, state: Self::State) -> Generate<Self, Self::State, S>
    where
        Self: PositionVertexGenerator<S>,
        S: EuclideanSpace,
    {
        Generate::new(
            self,
            state,
            self.vertex_with_position_count(),
            Self::vertex_with_position_from,
        )
    }
}

pub trait PositionIndexGenerator: PolygonGenerator {
    type Output: Polygonal<Vertex = usize>;

    fn index_for_position(&self, index: usize) -> Self::Output;
}

pub trait IndicesForPosition: PositionIndexGenerator + Sized {
    fn indices_for_position(&self) -> Generate<Self, (), Self::Output> {
        Generate::new(self, (), self.polygon_count(), |generator, _, index| {
            generator.index_for_position(index)
        })
    }
}

impl<G> IndicesForPosition for G where G: PositionIndexGenerator + Sized {}

pub trait PositionPolygonGenerator<S>: PolygonGenerator + PositionGenerator<S>
where
    S: EuclideanSpace,
{
    type Output: Polygonal<Vertex = S>;

    fn polygon_with_position_from(&self, state: &Self::State, index: usize) -> Self::Output;
}

/// Functions for generating polygons with position data.
pub trait PolygonsWithPosition: Sized {
    /// Provides an iterator over the set of unique polygons with position
    /// data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::N64;
    /// use nalgebra::Point3;
    /// use plexus::index::{HashIndexer, StructuredN};
    /// use plexus::prelude::*;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// # fn main() {
    /// let (indices, positions) = UvSphere::new(8, 8)
    ///     .polygons_with_position::<Point3<N64>>()
    ///     .index_vertices::<StructuredN, _>(HashIndexer::default());
    /// # }
    /// ```
    fn polygons_with_position<S>(
        &self,
    ) -> Generate<Self, Self::State, <Self as PositionPolygonGenerator<S>>::Output>
    where
        Self: PositionPolygonGenerator<S>,
        S: EuclideanSpace,
    {
        self.polygons_with_position_from(Default::default())
    }

    /// Provides an iterator over the set of unique polygons with position data
    /// using the provided state.
    ///
    /// State typically dictates the scale of the generated positions, using a
    /// unit width, radius, etc.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::N64;
    /// use nalgebra::Point3;
    /// use plexus::index::{HashIndexer, Structured4};
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::{Bounds, Cube};
    ///
    /// # fn main() {
    /// let (indices, positions) = Cube::new()
    ///     .polygons_with_position_from::<Point3<N64>>(Bounds::unit_radius())
    ///     .index_vertices::<Structured4, _>(HashIndexer::default());
    /// # }
    /// ```
    fn polygons_with_position_from<S>(
        &self,
        state: Self::State,
    ) -> Generate<Self, Self::State, <Self as PositionPolygonGenerator<S>>::Output>
    where
        Self: PositionPolygonGenerator<S>,
        S: EuclideanSpace,
    {
        Generate::new(
            self,
            state,
            self.polygon_count(),
            Self::polygon_with_position_from,
        )
    }
}

pub trait UvMapGenerator<S>
where
    S: EuclideanSpace,
    <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U2>,
{
    type State: Default;
}

pub trait UvMapPolygonGenerator<S>: PolygonGenerator + UvMapGenerator<S>
where
    S: EuclideanSpace,
    <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U2>,
{
    type Output: Polygonal<Vertex = Vector<S>>;

    fn polygon_with_uv_map_from(&self, state: &Self::State, index: usize) -> Self::Output;
}

/// Functions for generating polygons with UV-mapping (texture coordinate)
/// data.
pub trait PolygonsWithUvMap: Sized {
    /// Provides an iterator over the set of unique polygons with UV-mapping
    /// data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::N64;
    /// use nalgebra::{Point2, Point3};
    /// use plexus::index::{Flat4, HashIndexer};
    /// use plexus::prelude::*;
    /// use plexus::primitive;
    /// use plexus::primitive::cube::Cube;
    ///
    /// # fn main() {
    /// type E2 = Point2<N64>;
    /// type E3 = Point3<N64>;
    ///
    /// let cube = Cube::new();
    /// let (indices, positions) = primitive::zip_vertices((
    ///     cube.polygons_with_position::<E3>(),
    ///     cube.polygons_with_uv_map::<E2>(),
    /// ))
    /// .index_vertices::<Flat4, _>(HashIndexer::default());
    /// # }
    /// ```
    fn polygons_with_uv_map<S>(
        &self,
    ) -> Generate<Self, Self::State, <Self as UvMapPolygonGenerator<S>>::Output>
    where
        Self: UvMapPolygonGenerator<S>,
        S: EuclideanSpace,
        <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U2>,
    {
        self.polygons_with_uv_map_from(Default::default())
    }

    /// Provides an iterator over the set of unique polygons with UV-mapping
    /// data using the provided state.
    fn polygons_with_uv_map_from<S>(
        &self,
        state: Self::State,
    ) -> Generate<Self, Self::State, <Self as UvMapPolygonGenerator<S>>::Output>
    where
        Self: UvMapPolygonGenerator<S>,
        S: EuclideanSpace,
        <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U2>,
    {
        Generate::new(
            self,
            state,
            self.polygon_count(),
            Self::polygon_with_uv_map_from,
        )
    }
}
