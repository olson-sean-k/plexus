//! Topological decomposition and tessellation.
//!
//! The `Decompose` iterator uses various traits to decompose and tessellate
//! streams of topological structures.

use arrayvec::ArrayVec;
use std::collections::VecDeque;
use std::iter::IntoIterator;
use theon::ops::Interpolate;
use theon::IntoItems;

use crate::primitive::{Edge, Polygon, Polygonal, Quad, Topological, Triangle};

pub struct Decompose<I, P, Q, R>
where
    R: IntoIterator<Item = Q>,
{
    input: I,
    output: VecDeque<Q>,
    f: fn(P) -> R,
}

impl<I, P, Q, R> Decompose<I, P, Q, R>
where
    R: IntoIterator<Item = Q>,
{
    pub(in crate::primitive) fn new(input: I, f: fn(P) -> R) -> Self {
        Decompose {
            input,
            output: VecDeque::new(),
            f,
        }
    }
}

impl<I, P, R> Decompose<I, P, P, R>
where
    I: Iterator<Item = P>,
    R: IntoIterator<Item = P>,
{
    /// Reapplies a congruent decomposition.
    ///
    /// A decomposition is _congruent_ if its input and output types are the
    /// same. This is useful when the number of applications is somewhat large
    /// or variable, in which case chaining calls is impractical or impossible.
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
    /// use plexus::index::{Flat4, HashIndexer};
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    ///
    /// # fn main() {
    /// let (indices, positions) = Cube::new()
    ///     .polygons_with_position::<Point3<N64>>()
    ///     .subdivide()
    ///     .remap(7) // 8 subdivision operations are applied.
    ///     .index_vertices::<Flat4, _>(HashIndexer::default());
    /// # }
    /// ```
    pub fn remap(self, n: usize) -> Decompose<impl Iterator<Item = P>, P, P, R> {
        let Decompose { input, output, f } = self;
        Decompose::new(output.into_iter().rev().chain(remap(n, input, f)), f)
    }
}

impl<I, P, Q, R> Iterator for Decompose<I, P, Q, R>
where
    I: Iterator<Item = P>,
    R: IntoIterator<Item = Q>,
{
    type Item = Q;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(topology) = self.output.pop_front() {
                return Some(topology);
            }
            if let Some(topology) = self.input.next() {
                self.output.extend((self.f)(topology));
            }
            else {
                return None;
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, _) = self.input.size_hint();
        (lower, None)
    }
}

pub trait IntoVertices: Topological {
    type Output: IntoIterator<Item = Self::Vertex>;

    fn into_vertices(self) -> Self::Output;
}

impl<T> IntoVertices for T
where
    T: IntoItems + Topological,
{
    type Output = <T as IntoItems>::Output;

    fn into_vertices(self) -> Self::Output {
        self.into_items()
    }
}

pub trait IntoEdges: Topological {
    type Output: IntoIterator<Item = Edge<Self::Vertex>>;

    fn into_edges(self) -> Self::Output;
}

pub trait IntoTriangles: Polygonal {
    type Output: IntoIterator<Item = Triangle<Self::Vertex>>;

    fn into_triangles(self) -> Self::Output;
}

pub trait IntoSubdivisions: Polygonal {
    type Output: IntoIterator<Item = Self>;

    fn into_subdivisions(self) -> Self::Output;
}

pub trait IntoTetrahedrons: Polygonal {
    fn into_tetrahedrons(self) -> ArrayVec<[Triangle<Self::Vertex>; 4]>;
}

impl<T> IntoEdges for Edge<T> {
    type Output = ArrayVec<[Edge<Self::Vertex>; 1]>;

    fn into_edges(self) -> Self::Output {
        ArrayVec::from([self])
    }
}

impl<T> IntoEdges for Triangle<T>
where
    T: Clone,
{
    type Output = ArrayVec<[Edge<Self::Vertex>; 3]>;

    fn into_edges(self) -> Self::Output {
        let Triangle { a, b, c } = self;
        ArrayVec::from([
            Edge::new(a.clone(), b.clone()),
            Edge::new(b, c.clone()),
            Edge::new(c, a),
        ])
    }
}

impl<T> IntoEdges for Quad<T>
where
    T: Clone,
{
    type Output = ArrayVec<[Edge<Self::Vertex>; 4]>;

    fn into_edges(self) -> Self::Output {
        let Quad { a, b, c, d } = self;
        ArrayVec::from([
            Edge::new(a.clone(), b.clone()),
            Edge::new(b, c.clone()),
            Edge::new(c, d.clone()),
            Edge::new(d, a),
        ])
    }
}

impl<T> IntoEdges for Polygon<T>
where
    T: Clone,
{
    type Output = Vec<Edge<Self::Vertex>>;

    fn into_edges(self) -> Self::Output {
        match self {
            Polygon::Triangle(triangle) => triangle.into_edges().into_iter().collect(),
            Polygon::Quad(quad) => quad.into_edges().into_iter().collect(),
        }
    }
}

impl<T> IntoTriangles for Triangle<T> {
    type Output = ArrayVec<[Triangle<Self::Vertex>; 1]>;

    fn into_triangles(self) -> Self::Output {
        ArrayVec::from([self])
    }
}

impl<T> IntoTriangles for Quad<T>
where
    T: Clone,
{
    type Output = ArrayVec<[Triangle<Self::Vertex>; 2]>;

    fn into_triangles(self) -> Self::Output {
        let Quad { a, b, c, d } = self;
        ArrayVec::from([
            Triangle::new(a.clone(), b, c.clone()),
            Triangle::new(c, d, a),
        ])
    }
}

impl<T> IntoTriangles for Polygon<T>
where
    T: Clone,
{
    type Output = Vec<Triangle<Self::Vertex>>;

    fn into_triangles(self) -> Self::Output {
        match self {
            Polygon::Triangle(triangle) => triangle.into_triangles().into_iter().collect(),
            Polygon::Quad(quad) => quad.into_triangles().into_iter().collect(),
        }
    }
}

impl<T> IntoSubdivisions for Triangle<T>
where
    T: Clone + Interpolate<Output = T>,
{
    type Output = ArrayVec<[Triangle<Self::Vertex>; 2]>;

    fn into_subdivisions(self) -> Self::Output {
        let Triangle { a, b, c } = self;
        let ac = a.clone().midpoint(c.clone());
        ArrayVec::from([
            Triangle::new(b.clone(), ac.clone(), a),
            Triangle::new(c, ac, b),
        ])
    }
}

impl<T> IntoSubdivisions for Quad<T>
where
    T: Clone + Interpolate<Output = T>,
{
    type Output = ArrayVec<[Quad<Self::Vertex>; 4]>;

    fn into_subdivisions(self) -> Self::Output {
        let Quad { a, b, c, d } = self;
        let ab = a.clone().midpoint(b.clone());
        let bc = b.clone().midpoint(c.clone());
        let cd = c.clone().midpoint(d.clone());
        let da = d.clone().midpoint(a.clone());
        let ac = a.clone().midpoint(c.clone()); // Diagonal.
        ArrayVec::from([
            Quad::new(a, ab.clone(), ac.clone(), da.clone()),
            Quad::new(ab, b, bc.clone(), ac.clone()),
            Quad::new(ac.clone(), bc, c, cd.clone()),
            Quad::new(da, ac, cd, d),
        ])
    }
}

impl<T> IntoTetrahedrons for Quad<T>
where
    T: Clone + Interpolate<Output = T>,
{
    fn into_tetrahedrons(self) -> ArrayVec<[Triangle<Self::Vertex>; 4]> {
        let Quad { a, b, c, d } = self;
        let ac = a.clone().midpoint(c.clone()); // Diagonal.
        ArrayVec::from([
            Triangle::new(a.clone(), b.clone(), ac.clone()),
            Triangle::new(b, c.clone(), ac.clone()),
            Triangle::new(c, d.clone(), ac.clone()),
            Triangle::new(d, a, ac),
        ])
    }
}

impl<T> IntoSubdivisions for Polygon<T>
where
    T: Clone + Interpolate<Output = T>,
{
    type Output = Vec<Self>;

    fn into_subdivisions(self) -> Self::Output {
        match self {
            Polygon::Triangle(triangle) => triangle
                .into_subdivisions()
                .into_iter()
                .map(|triangle| triangle.into())
                .collect(),
            Polygon::Quad(quad) => quad
                .into_subdivisions()
                .into_iter()
                .map(|quad| quad.into())
                .collect(),
        }
    }
}

pub trait Vertices<P>: Sized
where
    P: IntoVertices,
{
    fn vertices(self) -> Decompose<Self, P, P::Vertex, P::Output>;
}

impl<I, P> Vertices<P> for I
where
    I: Iterator<Item = P>,
    P: IntoVertices,
{
    fn vertices(self) -> Decompose<Self, P, P::Vertex, P::Output> {
        Decompose::new(self, P::into_vertices)
    }
}

pub trait Edges<P>: Sized
where
    P: IntoEdges,
{
    fn edges(self) -> Decompose<Self, P, Edge<P::Vertex>, P::Output>;
}

impl<I, P> Edges<P> for I
where
    I: Iterator<Item = P>,
    P: IntoEdges,
    P::Vertex: Clone,
{
    fn edges(self) -> Decompose<Self, P, Edge<P::Vertex>, P::Output> {
        Decompose::new(self, P::into_edges)
    }
}

pub trait Triangulate<P>: Sized
where
    P: IntoTriangles,
{
    fn triangulate(self) -> Decompose<Self, P, Triangle<P::Vertex>, P::Output>;
}

impl<I, P> Triangulate<P> for I
where
    I: Iterator<Item = P>,
    P: IntoTriangles,
{
    fn triangulate(self) -> Decompose<Self, P, Triangle<P::Vertex>, P::Output> {
        Decompose::new(self, P::into_triangles)
    }
}

pub trait Subdivide<P>: Sized
where
    P: IntoSubdivisions,
{
    fn subdivide(self) -> Decompose<Self, P, P, P::Output>;
}

impl<I, P> Subdivide<P> for I
where
    I: Iterator<Item = P>,
    P: IntoSubdivisions,
{
    fn subdivide(self) -> Decompose<Self, P, P, P::Output> {
        Decompose::new(self, P::into_subdivisions)
    }
}

pub trait Tetrahedrons<T>: Sized {
    #[allow(type_complexity)]
    fn tetrahedrons(self) -> Decompose<Self, Quad<T>, Triangle<T>, ArrayVec<[Triangle<T>; 4]>>;
}

impl<I, T> Tetrahedrons<T> for I
where
    I: Iterator<Item = Quad<T>>,
    T: Clone + Interpolate<Output = T>,
{
    #[allow(type_complexity)]
    fn tetrahedrons(self) -> Decompose<Self, Quad<T>, Triangle<T>, ArrayVec<[Triangle<T>; 4]>> {
        Decompose::new(self, Quad::into_tetrahedrons)
    }
}

fn remap<I, P, R, F>(n: usize, topologies: I, f: F) -> Vec<P>
where
    I: IntoIterator<Item = P>,
    R: IntoIterator<Item = P>,
    F: Fn(P) -> R,
{
    let mut topologies: Vec<_> = topologies.into_iter().collect();
    for _ in 0..n {
        topologies = topologies.into_iter().flat_map(&f).collect();
    }
    topologies
}
