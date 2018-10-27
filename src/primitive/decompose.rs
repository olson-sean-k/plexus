//! Topological decomposition and tessellation.
//!
//! The `Decompose` iterator uses various traits to decompose and tessellate
//! streams of topological structures.

use arrayvec::ArrayVec;
use std::collections::{vec_deque, VecDeque};
use std::iter::{Chain, IntoIterator, Rev};
use std::vec;

use geometry::ops::Interpolate;
use primitive::topology::{Edge, Polygon, Polygonal, Quad, Topological, Triangle};

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
    pub(in primitive) fn new(input: I, f: fn(P) -> R) -> Self {
        Decompose {
            input,
            output: VecDeque::new(),
            f,
        }
    }
}

// TODO: Use `impl Iterator<Item = P>` instead of this alias after
//       https://github.com/rust-lang/rust/issues/50823 is fixed.
// Names the iterator fed into the `Decompose` adapter in `remap`.
type Remap<P> = Chain<Rev<vec_deque::IntoIter<P>>, vec::IntoIter<P>>;

impl<I, P, R> Decompose<I, P, P, R>
where
    I: Iterator<Item = P>,
    R: IntoIterator<Item = P>,
{
    /// Reapplies a congruent decomposition.
    ///
    /// A decomposition is congruent if its input and output types are the
    /// same. This is useful when the number of applications is somewhat large
    /// or variable, in which case chaining calls is impractical or impossible.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::index::HashIndexer;
    ///
    /// let (indeces, positions) = Cube::new()
    ///     .polygons_with_position()
    ///     .subdivide()
    ///     .remap(7) // 8 subdivision operations are applied.
    ///     .flat_index_vertices(HashIndexer::default());
    /// ```
    pub fn remap(self, n: usize) -> Decompose<Remap<P>, P, P, R> {
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
}

pub trait IntoVertices: Topological {
    type Output: IntoIterator<Item = Self::Vertex>;

    fn into_vertices(self) -> Self::Output;
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

impl<T> IntoVertices for Edge<T> {
    type Output = ArrayVec<[Self::Vertex; 2]>;

    fn into_vertices(self) -> Self::Output {
        let Edge { a, b } = self;
        ArrayVec::from([a, b])
    }
}

impl<T> IntoVertices for Triangle<T> {
    type Output = ArrayVec<[Self::Vertex; 3]>;

    fn into_vertices(self) -> Self::Output {
        let Triangle { a, b, c } = self;
        ArrayVec::from([a, b, c])
    }
}

impl<T> IntoVertices for Quad<T> {
    type Output = ArrayVec<[Self::Vertex; 4]>;

    fn into_vertices(self) -> Self::Output {
        let Quad { a, b, c, d } = self;
        ArrayVec::from([a, b, c, d])
    }
}

impl<T> IntoVertices for Polygon<T> {
    type Output = Vec<Self::Vertex>;

    fn into_vertices(self) -> Self::Output {
        match self {
            Polygon::Triangle(triangle) => triangle.into_vertices().into_iter().collect(),
            Polygon::Quad(quad) => quad.into_vertices().into_iter().collect(),
        }
    }
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
