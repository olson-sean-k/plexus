//! Primitive generation.
//!
//! This module provides a generic iterator and traits for generating streams
//! of geometric and topological data for primitives like cubes and spheres.

use std::ops::Range;

use primitive::topology::Polygonal;

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
    pub(in primitive) fn new(
        generator: &'a G,
        state: S,
        range: Range<usize>,
        f: fn(&'a G, &S, usize) -> P,
    ) -> Self {
        Generate {
            generator,
            state,
            range,
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
pub trait VertexGenerator {
    /// Gets the number of unique vertices that comprise a primitive.
    fn vertex_count(&self) -> usize;
}

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
}

/// Functions for generating vertices with position data.
pub trait VerticesWithPosition<P>: PositionGenerator + Sized {
    /// Provides an iterator over the set of unique vertices with position
    /// data.
    ///
    /// This can be paired with functions from `PolygonsWithIndex` to index the
    /// set of vertices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::prelude::*;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// let sphere = UvSphere::new(8, 8);
    /// let vertices = sphere
    ///     .vertices_with_position()
    ///     .collect::<Vec<_>>();
    /// let indices = sphere
    ///     .polygons_with_index()
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
    /// This can be paired with functions from `PolygonsWithIndex` to index the
    /// set of vertices.
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
    ///     .vertices_with_position_from(Bounds::unit_radius())
    ///     .collect::<Vec<_>>();
    /// let indices = cube
    ///     .polygons_with_index()
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
            0..self.vertex_count(),
            G::vertex_with_position_from,
        )
    }
}

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
            0..self.polygon_count(),
            G::polygon_with_position_from,
        )
    }
}

pub trait IndexGenerator {
    type State: Default;
}

pub trait IndexPolygonGenerator: IndexGenerator + PolygonGenerator + VertexGenerator {
    type Output: Polygonal;

    fn polygon_with_index_from(
        &self,
        state: &Self::State,
        index: usize,
    ) -> <Self as IndexPolygonGenerator>::Output;
}

/// Functions for generating polygons with index data.
///
/// The indices generated by these functions map to the unique set of vertices
/// produced by a primitive's vertex generators, such as
/// `VerticesWithPosition`.
pub trait PolygonsWithIndex<P>: IndexGenerator + Sized {
    /// Provides an iterator over the set of unique polygons with index data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    ///
    /// let cube = Cube::new();
    /// let vertices = cube
    ///     .vertices_with_position()
    ///     .collect::<Vec<_>>();
    /// let indices = cube
    ///     .polygons_with_index()
    ///     .triangulate()
    ///     .vertices()
    ///     .collect::<Vec<_>>();
    /// ```
    fn polygons_with_index(&self) -> Generate<Self, Self::State, P> {
        self.polygons_with_index_from(Default::default())
    }

    /// Provides an iterator over the set of unique polygons with index data
    /// using the provided state.
    fn polygons_with_index_from(&self, state: Self::State) -> Generate<Self, Self::State, P>;
}

impl<G, P> PolygonsWithIndex<P> for G
where
    G: IndexPolygonGenerator<Output = P> + VertexGenerator + PolygonGenerator,
    P: Polygonal,
{
    fn polygons_with_index_from(&self, state: Self::State) -> Generate<Self, Self::State, P> {
        Generate::new(
            self,
            state,
            0..self.polygon_count(),
            G::polygon_with_index_from,
        )
    }
}

pub trait TextureGenerator {
    type State: Default;
}

pub trait TexturePolygonGenerator: PolygonGenerator + TextureGenerator {
    type Output: Polygonal;

    fn polygon_with_texture_from(
        &self,
        state: &Self::State,
        index: usize,
    ) -> <Self as TexturePolygonGenerator>::Output;
}

/// Functions for generating polygons with texture coordinate data.
pub trait PolygonsWithTexture<P>: Sized + TextureGenerator {
    /// Provides an iterator over the set of unique polygons with texture
    /// coordinate data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::index::HashIndexer;
    /// use plexus::primitive;
    ///
    /// let cube = Cube::new();
    /// let (indices, positions) = primitive::zip_vertices((
    ///         cube.polygons_with_position(),
    ///         cube.polygons_with_texture(),
    ///     ))
    ///     .index_vertices(HashIndexer::default());
    /// ```
    fn polygons_with_texture(&self) -> Generate<Self, Self::State, P> {
        self.polygons_with_texture_from(Default::default())
    }

    /// Provides an iterator over the set of unique polygons with texture
    /// coordinate data using the provided state.
    fn polygons_with_texture_from(&self, state: Self::State) -> Generate<Self, Self::State, P>;
}

impl<G, P> PolygonsWithTexture<P> for G
where
    G: PolygonGenerator + TexturePolygonGenerator<Output = P>,
    P: Polygonal,
{
    fn polygons_with_texture_from(&self, state: Self::State) -> Generate<Self, Self::State, P> {
        Generate::new(
            self,
            state,
            0..self.polygon_count(),
            G::polygon_with_texture_from,
        )
    }
}
