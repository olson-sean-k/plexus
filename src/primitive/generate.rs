//! Primitive generation.
//!
//! This module provides a generic iterator and traits for generating streams
//! of geometric and topological data for primitives like cubes and spheres.

use std::ops::Range;

use crate::primitive::topology::Polygonal;

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
}

/// Vertex generating primitive.
pub trait VertexGenerator {}

/// Polygon generating primitive.
pub trait PolygonGenerator {
    /// Gets the number of unique polygons that comprise a primitive.
    fn polygon_count(&self) -> usize;
}

pub trait PositionGenerator {
    type State: Default;
}

pub trait PositionVertexGenerator: PositionGenerator + VertexGenerator {
    type Output;

    fn vertex_with_position_from(&self, state: &Self::State, index: usize) -> Self::Output;

    /// Gets the number of unique vertices with position data that comprise a primitive.
    fn vertex_with_position_count(&self) -> usize;
}

/// Functions for generating vertices with position data.
pub trait VerticesWithPosition<P>: PositionGenerator + Sized {
    /// Provides an iterator over the set of unique vertices with position
    /// data.
    ///
    /// This can be paired with functions from `IndicesForPosition` to index
    /// the set of positions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::prelude::*;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// let sphere = UvSphere::new(8, 8);
    /// let vertices = sphere.vertices_with_position().collect::<Vec<_>>();
    /// let indices = sphere
    ///     .indices_for_position()
    ///     .triangulate()
    ///     .vertices()
    ///     .collect::<Vec<_>>();
    /// ```
    fn vertices_with_position(&self) -> Generate<Self, Self::State, P> {
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
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::{Bounds, Cube};
    ///
    /// let cube = Cube::new();
    /// let vertices = cube
    ///     .vertices_with_position_from(Bounds::with_width(3.0.into()))
    ///     .collect::<Vec<_>>();
    /// let indices = cube
    ///     .indices_for_position()
    ///     .triangulate()
    ///     .vertices()
    ///     .collect::<Vec<_>>();
    /// ```
    fn vertices_with_position_from(&self, state: Self::State) -> Generate<Self, Self::State, P>;
}

impl<G, P> VerticesWithPosition<P> for G
where
    G: PositionVertexGenerator<Output = P>,
{
    fn vertices_with_position_from(&self, state: Self::State) -> Generate<Self, Self::State, P> {
        Generate::new(
            self,
            state,
            self.vertex_with_position_count(),
            G::vertex_with_position_from,
        )
    }
}

pub trait PositionIndexGenerator: PolygonGenerator + PositionVertexGenerator {
    type Output: Polygonal<Vertex = usize>;

    fn index_for_position(&self, index: usize) -> <Self as PositionIndexGenerator>::Output;
}

pub trait IndicesForPosition: PositionIndexGenerator + Sized {
    fn indices_for_position(&self) -> Generate<Self, (), <Self as PositionIndexGenerator>::Output> {
        Generate::new(self, (), self.polygon_count(), |generator, _, index| {
            generator.index_for_position(index)
        })
    }
}

impl<G> IndicesForPosition for G where G: PositionIndexGenerator + Sized {}

pub trait PositionPolygonGenerator: PolygonGenerator + PositionGenerator {
    type Output: Polygonal;

    fn polygon_with_position_from(&self, state: &Self::State, index: usize) -> Self::Output;
}

/// Functions for generating polygons with position data.
pub trait PolygonsWithPosition<P>: PositionGenerator + Sized {
    /// Provides an iterator over the set of unique polygons with position
    /// data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::prelude::*;
    /// use plexus::primitive::index::HashIndexer;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// let (indices, positions) = UvSphere::new(8, 8)
    ///     .polygons_with_position()
    ///     .index_vertices(HashIndexer::default());
    /// ```
    fn polygons_with_position(&self) -> Generate<Self, Self::State, P> {
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
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::{Bounds, Cube};
    /// use plexus::primitive::index::HashIndexer;
    ///
    /// let (indices, positions) = Cube::new()
    ///     .polygons_with_position_from(Bounds::unit_radius())
    ///     .index_vertices(HashIndexer::default());
    /// ```
    fn polygons_with_position_from(&self, state: Self::State) -> Generate<Self, Self::State, P>;
}

impl<G, P> PolygonsWithPosition<P> for G
where
    G: PositionPolygonGenerator<Output = P>,
    P: Polygonal,
{
    fn polygons_with_position_from(&self, state: Self::State) -> Generate<Self, Self::State, P> {
        Generate::new(
            self,
            state,
            self.polygon_count(),
            G::polygon_with_position_from,
        )
    }
}

pub trait UvMapGenerator {
    type State: Default;
}

pub trait UvMapPolygonGenerator: PolygonGenerator + UvMapGenerator {
    type Output: Polygonal;

    fn polygon_with_uv_map_from(
        &self,
        state: &Self::State,
        index: usize,
    ) -> <Self as UvMapPolygonGenerator>::Output;
}

/// Functions for generating polygons with UV-mapping (texture coordinate)
/// data.
pub trait PolygonsWithUvMap<P>: Sized + UvMapGenerator {
    /// Provides an iterator over the set of unique polygons with UV-mapping
    /// data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::prelude::*;
    /// use plexus::primitive;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::index::HashIndexer;
    ///
    /// let cube = Cube::new();
    /// let (indices, positions) =
    ///     primitive::zip_vertices((cube.polygons_with_position(), cube.polygons_with_uv_map()))
    ///         .index_vertices(HashIndexer::default());
    /// ```
    fn polygons_with_uv_map(&self) -> Generate<Self, Self::State, P> {
        self.polygons_with_uv_map_from(Default::default())
    }

    /// Provides an iterator over the set of unique polygons with UV-mapping
    /// data using the provided state.
    fn polygons_with_uv_map_from(&self, state: Self::State) -> Generate<Self, Self::State, P>;
}

impl<G, P> PolygonsWithUvMap<P> for G
where
    G: PolygonGenerator + UvMapPolygonGenerator<Output = P>,
    P: Polygonal,
{
    fn polygons_with_uv_map_from(&self, state: Self::State) -> Generate<Self, Self::State, P> {
        Generate::new(
            self,
            state,
            self.polygon_count(),
            G::polygon_with_uv_map_from,
        )
    }
}
