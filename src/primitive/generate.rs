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

pub trait NormalGenerator {
    type State: Default;
}

pub trait NormalVertexGenerator: NormalGenerator + VertexGenerator {
    type Output;

    fn vertex_with_normal_from(&self, state: &Self::State, index: usize) -> Self::Output;

    /// Gets the number of unique vertices with normal data that comprise a primitive.
    fn vertex_with_normal_count(&self) -> usize;
}

/// Functions for generating vertices with normal data.
pub trait VerticesWithNormal: NormalVertexGenerator + Sized {
    fn vertices_with_normal(
        &self,
    ) -> Generate<Self, Self::State, <Self as NormalVertexGenerator>::Output> {
        self.vertices_with_normal_from(Default::default())
    }

    fn vertices_with_normal_from(
        &self,
        state: Self::State,
    ) -> Generate<Self, Self::State, <Self as NormalVertexGenerator>::Output> {
        Generate::new(
            self,
            state,
            self.vertex_with_normal_count(),
            Self::vertex_with_normal_from,
        )
    }
}

impl<G> VerticesWithNormal for G where G: NormalVertexGenerator + Sized {}

pub trait NormalIndexGenerator: PolygonGenerator + NormalVertexGenerator {
    type Output: Polygonal<Vertex = usize>;

    fn index_for_normal(&self, index: usize) -> <Self as NormalIndexGenerator>::Output;
}

pub trait IndicesForNormal: NormalIndexGenerator + Sized {
    fn indices_for_normal(&self) -> Generate<Self, (), <Self as NormalIndexGenerator>::Output> {
        Generate::new(self, (), self.polygon_count(), |generator, _, index| {
            generator.index_for_normal(index)
        })
    }
}

impl<G> IndicesForNormal for G where G: NormalIndexGenerator + Sized {}

pub trait NormalPolygonGenerator: PolygonGenerator + NormalGenerator {
    type Output: Polygonal;

    fn polygon_with_normal_from(&self, state: &Self::State, index: usize) -> Self::Output;
}

/// Functions for generating polygons with normal data.
pub trait PolygonsWithNormal: NormalPolygonGenerator + Sized {
    fn polygons_with_normal(
        &self,
    ) -> Generate<Self, Self::State, <Self as NormalPolygonGenerator>::Output> {
        self.polygons_with_normal_from(Default::default())
    }

    fn polygons_with_normal_from(
        &self,
        state: Self::State,
    ) -> Generate<Self, Self::State, <Self as NormalPolygonGenerator>::Output> {
        Generate::new(
            self,
            state,
            self.polygon_count(),
            Self::polygon_with_normal_from,
        )
    }
}

impl<G> PolygonsWithNormal for G where G: NormalPolygonGenerator {}

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
pub trait VerticesWithPosition: PositionVertexGenerator + Sized {
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
    fn vertices_with_position(
        &self,
    ) -> Generate<Self, Self::State, <Self as PositionVertexGenerator>::Output> {
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
    fn vertices_with_position_from(
        &self,
        state: Self::State,
    ) -> Generate<Self, Self::State, <Self as PositionVertexGenerator>::Output> {
        Generate::new(
            self,
            state,
            self.vertex_with_position_count(),
            Self::vertex_with_position_from,
        )
    }
}

impl<G> VerticesWithPosition for G where G: PositionVertexGenerator + Sized {}

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
pub trait PolygonsWithPosition: PositionPolygonGenerator + Sized {
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
    fn polygons_with_position(
        &self,
    ) -> Generate<Self, Self::State, <Self as PositionPolygonGenerator>::Output> {
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
    fn polygons_with_position_from(
        &self,
        state: Self::State,
    ) -> Generate<Self, Self::State, <Self as PositionPolygonGenerator>::Output> {
        Generate::new(
            self,
            state,
            self.polygon_count(),
            Self::polygon_with_position_from,
        )
    }
}

impl<G> PolygonsWithPosition for G where G: PositionPolygonGenerator {}

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
pub trait PolygonsWithUvMap: Sized + UvMapPolygonGenerator {
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
    fn polygons_with_uv_map(
        &self,
    ) -> Generate<Self, Self::State, <Self as UvMapPolygonGenerator>::Output> {
        self.polygons_with_uv_map_from(Default::default())
    }

    /// Provides an iterator over the set of unique polygons with UV-mapping
    /// data using the provided state.
    fn polygons_with_uv_map_from(
        &self,
        state: Self::State,
    ) -> Generate<Self, Self::State, <Self as UvMapPolygonGenerator>::Output> {
        Generate::new(
            self,
            state,
            self.polygon_count(),
            Self::polygon_with_uv_map_from,
        )
    }
}

impl<G> PolygonsWithUvMap for G where G: UvMapPolygonGenerator {}
