//! This module provides a generic iterator and traits for mapping from an
//! index to a topology of a primitive.

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
    pub(super) fn new(
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

pub trait VertexGenerator {
    fn vertex_count(&self) -> usize;
}

pub trait PolygonGenerator {
    fn polygon_count(&self) -> usize;
}

pub trait PositionGenerator {
    type State: Default;
}

pub trait PositionVertexGenerator: PositionGenerator + VertexGenerator {
    type Output;

    fn vertex_with_position_from(&self, state: &Self::State, index: usize) -> Self::Output;
}

pub trait VerticesWithPosition<P>: PositionGenerator + Sized {
    fn vertices_with_position(&self) -> Generate<Self, Self::State, P> {
        self.vertices_with_position_from(Default::default())
    }

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

pub trait PolygonsWithPosition<P>: PositionGenerator + Sized {
    fn polygons_with_position(&self) -> Generate<Self, Self::State, P> {
        self.polygons_with_position_from(Default::default())
    }

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

pub trait PolygonsWithIndex<P>: IndexGenerator + Sized {
    fn polygons_with_index(&self) -> Generate<Self, Self::State, P> {
        self.polygons_with_index_from(Default::default())
    }

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

pub trait PolygonsWithTexture<P>: Sized + TextureGenerator {
    fn polygons_with_texture(&self) -> Generate<Self, Self::State, P> {
        self.polygons_with_texture_from(Default::default())
    }

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
