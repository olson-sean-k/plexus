//! This module provides a generic iterator and traits for decomposing
//! topologies and tessellating polygons.

use arrayvec::ArrayVec;
use num::{self, Num, NumCast};
use std::collections::{vec_deque, VecDeque};
use std::iter::{Chain, IntoIterator, Rev};
use std::vec;

use generate::topology::{Line, Polygon, Polygonal, Topological, Triangle, Quad};

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
    pub(super) fn new(input: I, f: fn(P) -> R) -> Self {
        Decompose {
            input: input,
            output: VecDeque::new(),
            f: f,
        }
    }
}

// Names the iterator fed into the `Decompose` adapter in `remap`.
type Remap<P> = Chain<Rev<vec_deque::IntoIter<P>>, vec::IntoIter<P>>;

impl<I, P, R> Decompose<I, P, P, R>
where
    I: Iterator<Item = P>,
    R: IntoIterator<Item = P>,
{
    // TODO: This is questionable, but acts as a replacement for optional
    //       parameters used by the `Into*` traits. In particular,
    //       `into_subdivisions` no longer accepts a parameter `n`, and `remap`
    //       can be used to emulate that behavior. This is especially useful
    //       for larger `n` values, where chaining calls to `subdivide` is not
    //       practical.
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

pub trait IntoLines: Topological {
    type Output: IntoIterator<Item = Line<Self::Vertex>>;

    fn into_lines(self) -> Self::Output;
}

pub trait IntoTriangles: Polygonal {
    type Output: IntoIterator<Item = Triangle<Self::Vertex>>;

    fn into_triangles(self) -> Self::Output;
}

pub trait IntoSubdivisions: Polygonal
where
    Self::Vertex: Clone + Interpolate,
{
    type Output: IntoIterator<Item = Self>;

    fn into_subdivisions(self) -> Self::Output;
}

pub trait IntoTetrahedrons: Polygonal
where
    Self::Vertex: Clone + Interpolate,
{
    fn into_tetrahedrons(self) -> ArrayVec<[Triangle<Self::Vertex>; 4]>;
}

impl<T> IntoVertices for Line<T>
where
    T: Clone,
{
    type Output = ArrayVec<[Self::Vertex; 2]>;

    fn into_vertices(self) -> Self::Output {
        let Line { a, b } = self;
        ArrayVec::from([a, b])
    }
}

impl<T> IntoVertices for Triangle<T>
where
    T: Clone,
{
    type Output = ArrayVec<[Self::Vertex; 3]>;

    fn into_vertices(self) -> Self::Output {
        let Triangle { a, b, c } = self;
        ArrayVec::from([a, b, c])
    }
}

impl<T> IntoVertices for Quad<T>
where
    T: Clone,
{
    type Output = ArrayVec<[Self::Vertex; 4]>;

    fn into_vertices(self) -> Self::Output {
        let Quad { a, b, c, d } = self;
        ArrayVec::from([a, b, c, d])
    }
}

impl<T> IntoVertices for Polygon<T>
where
    T: Clone,
{
    type Output = Vec<Self::Vertex>;

    fn into_vertices(self) -> Self::Output {
        match self {
            Polygon::Triangle(triangle) => triangle.into_vertices().into_iter().collect(),
            Polygon::Quad(quad) => quad.into_vertices().into_iter().collect(),
        }
    }
}

impl<T> IntoLines for Line<T>
where
    T: Clone,
{
    type Output = ArrayVec<[Line<Self::Vertex>; 1]>;

    fn into_lines(self) -> Self::Output {
        ArrayVec::from([self])
    }
}

impl<T> IntoLines for Triangle<T>
where
    T: Clone,
{
    type Output = ArrayVec<[Line<Self::Vertex>; 3]>;

    fn into_lines(self) -> Self::Output {
        let Triangle { a, b, c } = self;
        ArrayVec::from([
            Line::new(a.clone(), b.clone()),
            Line::new(b, c.clone()),
            Line::new(c, a),
        ])
    }
}

impl<T> IntoLines for Quad<T>
where
    T: Clone,
{
    type Output = ArrayVec<[Line<Self::Vertex>; 4]>;

    fn into_lines(self) -> Self::Output {
        let Quad { a, b, c, d } = self;
        ArrayVec::from([
            Line::new(a.clone(), b.clone()),
            Line::new(b, c.clone()),
            Line::new(c, d.clone()),
            Line::new(d, a),
        ])
    }
}

impl<T> IntoLines for Polygon<T>
where
    T: Clone,
{
    type Output = Vec<Line<Self::Vertex>>;

    fn into_lines(self) -> Self::Output {
        match self {
            Polygon::Triangle(triangle) => triangle.into_lines().into_iter().collect(),
            Polygon::Quad(quad) => quad.into_lines().into_iter().collect(),
        }
    }
}

impl<T> IntoTriangles for Triangle<T>
where
    T: Clone,
{
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
    T: Clone + Interpolate,
{
    type Output = ArrayVec<[Triangle<Self::Vertex>; 2]>;

    fn into_subdivisions(self) -> Self::Output {
        let Triangle { a, b, c } = self;
        let ac = a.midpoint(&c);
        ArrayVec::from([
            Triangle::new(b.clone(), ac.clone(), a),
            Triangle::new(c, ac, b),
        ])
    }
}

impl<T> IntoSubdivisions for Quad<T>
where
    T: Clone + Interpolate,
{
    type Output = ArrayVec<[Quad<Self::Vertex>; 4]>;

    fn into_subdivisions(self) -> Self::Output {
        let Quad { a, b, c, d } = self;
        let ab = a.midpoint(&b);
        let bc = b.midpoint(&c);
        let cd = c.midpoint(&d);
        let da = d.midpoint(&a);
        let ac = a.midpoint(&c); // Diagonal.
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
    T: Clone + Interpolate,
{
    fn into_tetrahedrons(self) -> ArrayVec<[Triangle<Self::Vertex>; 4]> {
        let Quad { a, b, c, d } = self;
        let ac = a.midpoint(&c); // Diagonal.
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
    T: Clone + Interpolate,
{
    type Output = Vec<Self>;

    fn into_subdivisions(self) -> Self::Output {
        match self {
            Polygon::Triangle(triangle) => {
                triangle
                    .into_subdivisions()
                    .into_iter()
                    .map(|triangle| triangle.into())
                    .collect()
            }
            Polygon::Quad(quad) => {
                quad.into_subdivisions()
                    .into_iter()
                    .map(|quad| quad.into())
                    .collect()
            }
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
    P::Vertex: Clone,
{
    fn vertices(self) -> Decompose<Self, P, P::Vertex, P::Output> {
        Decompose::new(self, P::into_vertices)
    }
}

pub trait Lines<P>: Sized
where
    P: IntoLines,
{
    fn lines(self) -> Decompose<Self, P, Line<P::Vertex>, P::Output>;
}

impl<I, P> Lines<P> for I
where
    I: Iterator<Item = P>,
    P: IntoLines,
    P::Vertex: Clone,
{
    fn lines(self) -> Decompose<Self, P, Line<P::Vertex>, P::Output> {
        Decompose::new(self, P::into_lines)
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
    P::Vertex: Clone,
{
    fn triangulate(self) -> Decompose<Self, P, Triangle<P::Vertex>, P::Output> {
        Decompose::new(self, P::into_triangles)
    }
}

pub trait Subdivide<P>: Sized
where
    P: IntoSubdivisions,
    P::Vertex: Clone + Interpolate,
{
    fn subdivide(self) -> Decompose<Self, P, P, P::Output>;
}

impl<I, P> Subdivide<P> for I
where
    I: Iterator<Item = P>,
    P: IntoSubdivisions,
    P::Vertex: Clone + Interpolate,
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
    T: Clone + Interpolate,
{
    #[allow(type_complexity)]
    fn tetrahedrons(self) -> Decompose<Self, Quad<T>, Triangle<T>, ArrayVec<[Triangle<T>; 4]>> {
        Decompose::new(self, Quad::into_tetrahedrons)
    }
}

pub trait Interpolate: Sized {
    fn lerp(&self, other: &Self, f: f32) -> Self;

    fn midpoint(&self, other: &Self) -> Self {
        self.lerp(other, 0.5)
    }
}

impl<T> Interpolate for (T, T)
where
    T: Copy + Num + NumCast,
{
    fn lerp(&self, other: &Self, f: f32) -> Self {
        (lerp(self.0, other.0, f), lerp(self.1, other.1, f))
    }
}

impl<T> Interpolate for (T, T, T)
where
    T: Copy + Num + NumCast,
{
    fn lerp(&self, other: &Self, f: f32) -> Self {
        (lerp(self.0, other.0, f), lerp(self.1, other.1, f), lerp(self.2, other.2, f))
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

fn lerp<T>(a: T, b: T, f: f32) -> T
where
    T: Num + NumCast,
{
    let f = num::clamp(f, 0.0, 1.0);
    let af = <f32 as NumCast>::from(a).unwrap() * (1.0 - f);
    let bf = <f32 as NumCast>::from(b).unwrap() * f;
    <T as NumCast>::from(af + bf).unwrap()
}
