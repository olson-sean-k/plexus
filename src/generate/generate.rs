//! This module provides a generic iterator and traits for mapping from an
//! index to a topology of a primitive.

use std::ops::Range;

use generate::topology::Polygonal;

pub struct Generate<'a, G, P>
where
    G: 'a,
{
    generator: &'a G,
    range: Range<usize>,
    f: fn(&'a G, usize) -> P,
}

impl<'a, G, P> Generate<'a, G, P>
where
    G: 'a,
{
    pub(super) fn new(generator: &'a G, range: Range<usize>, f: fn(&'a G, usize) -> P) -> Self {
        Generate {
            generator: generator,
            range: range,
            f: f,
        }
    }
}

impl<'a, G, P> Iterator for Generate<'a, G, P>
where
    G: 'a,
{
    type Item = P;

    fn next(&mut self) -> Option<Self::Item> {
        self.range
            .next()
            .map(|index| (self.f)(self.generator, index))
    }
}

pub trait VertexGenerator {
    fn vertex_count(&self) -> usize;
}

pub trait PositionVertexGenerator: VertexGenerator {
    type Output;

    fn vertex_with_position(&self, index: usize) -> Self::Output;
}

pub trait VerticesWithPosition<P>: Sized {
    fn vertices_with_position(&self) -> Generate<Self, P>;
}

impl<G, P> VerticesWithPosition<P> for G
where
    G: PositionVertexGenerator<Output = P>,
{
    fn vertices_with_position(&self) -> Generate<Self, P> {
        Generate::new(self, 0..self.vertex_count(), G::vertex_with_position)
    }
}

pub trait PolygonGenerator {
    fn polygon_count(&self) -> usize;
}

pub trait PositionPolygonGenerator: PolygonGenerator {
    type Output: Polygonal;

    fn polygon_with_position(&self, index: usize) -> Self::Output;
}

pub trait PolygonsWithPosition<P>: Sized {
    fn polygons_with_position(&self) -> Generate<Self, P>;
}

impl<G, P> PolygonsWithPosition<P> for G
where
    G: PositionPolygonGenerator<Output = P>,
    P: Polygonal,
{
    fn polygons_with_position(&self) -> Generate<Self, P> {
        Generate::new(self, 0..self.polygon_count(), G::polygon_with_position)
    }
}

pub trait IndexPolygonGenerator: VertexGenerator + PolygonGenerator {
    type Output: Polygonal;

    fn polygon_with_index(&self, index: usize) -> <Self as IndexPolygonGenerator>::Output;
}

pub trait PolygonsWithIndex<P>: Sized {
    fn polygons_with_index(&self) -> Generate<Self, P>;
}

impl<G, P> PolygonsWithIndex<P> for G
where
    G: IndexPolygonGenerator<Output = P> + VertexGenerator + PolygonGenerator,
    P: Polygonal,
{
    fn polygons_with_index(&self) -> Generate<Self, P> {
        Generate::new(self, 0..self.polygon_count(), G::polygon_with_index)
    }
}

pub trait TexturePolygonGenerator: PolygonGenerator {
    type Output: Polygonal;

    fn polygon_with_texture(&self, index: usize) -> <Self as TexturePolygonGenerator>::Output;
}

pub trait PolygonsWithTexture<P>: Sized {
    fn polygons_with_texture(&self) -> Generate<Self, P>;
}

impl<G, P> PolygonsWithTexture<P> for G
where
    G: PolygonGenerator + TexturePolygonGenerator<Output = P>,
    P: Polygonal,
{
    fn polygons_with_texture(&self) -> Generate<Self, P> {
        Generate::new(self, 0..self.polygon_count(), G::polygon_with_texture)
    }
}
