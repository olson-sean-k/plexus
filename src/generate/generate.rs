//! This module provides a generic iterator and traits for mapping from an
//! index to a topology from some shape.

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

pub trait SpatialVertexGenerator: VertexGenerator {
    type Output;

    fn spatial_vertex(&self, index: usize) -> Self::Output;
}

pub trait SpatialVertices<P>: Sized {
    fn spatial_vertices(&self) -> Generate<Self, P>;
}

impl<G, P> SpatialVertices<P> for G
where
    G: SpatialVertexGenerator<Output = P>,
{
    fn spatial_vertices(&self) -> Generate<Self, P> {
        Generate::new(self, 0..self.vertex_count(), G::spatial_vertex)
    }
}

pub trait PolygonGenerator {
    fn polygon_count(&self) -> usize;
}

pub trait SpatialPolygonGenerator: PolygonGenerator {
    type Output: Polygonal;

    fn spatial_polygon(&self, index: usize) -> Self::Output;
}

pub trait SpatialPolygons<P>: Sized {
    fn spatial_polygons(&self) -> Generate<Self, P>;
}

impl<G, P> SpatialPolygons<P> for G
where
    G: SpatialPolygonGenerator<Output = P>,
    P: Polygonal,
{
    fn spatial_polygons(&self) -> Generate<Self, P> {
        Generate::new(self, 0..self.polygon_count(), G::spatial_polygon)
    }
}

pub trait IndexedPolygonGenerator: VertexGenerator + PolygonGenerator {
    type Output: Polygonal;

    fn indexed_polygon(&self, index: usize) -> <Self as IndexedPolygonGenerator>::Output;
}

pub trait IndexedPolygons<P>: Sized {
    fn indexed_polygons(&self) -> Generate<Self, P>;
}

impl<G, P> IndexedPolygons<P> for G
where
    G: IndexedPolygonGenerator<Output = P> + VertexGenerator + PolygonGenerator,
    P: Polygonal,
{
    fn indexed_polygons(&self) -> Generate<Self, P> {
        Generate::new(self, 0..self.polygon_count(), G::indexed_polygon)
    }
}

pub trait TexturedPolygonGenerator: PolygonGenerator {
    type Output: Polygonal;

    fn textured_polygon(&self, index: usize) -> <Self as TexturedPolygonGenerator>::Output;
}

pub trait TexturedPolygons<P>: Sized {
    fn textured_polygons(&self) -> Generate<Self, P>;
}

impl<G, P> TexturedPolygons<P> for G
where
    G: PolygonGenerator + TexturedPolygonGenerator<Output = P>,
    P: Polygonal,
{
    fn textured_polygons(&self) -> Generate<Self, P> {
        Generate::new(self, 0..self.polygon_count(), G::textured_polygon)
    }
}
