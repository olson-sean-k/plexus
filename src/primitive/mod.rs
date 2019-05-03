//! Primitive topological structures.
//!
//! This module provides primitives that can be composed to form more complex
//! structures.  For example, primitives are used by generators, iterator
//! expressions, and index buffers.
//!
//! Much functionality in this module are exposed via traits.  Many of these
//! traits are included in the `prelude` module, and it is highly recommended
//! to import the `prelude`'s contents as seen in the examples.
//!
//! Traits implemented by generators expose verbose function names like
//! `polygons_with_uv_map` or `vertices_with_position` to avoid ambiguity.
//! This is a somewhat unorthodox use of the term "with" in Rust function
//! names, but the alternatives are much less clear, especially when other
//! similar function names exist in the same crate.
//!
//! # Examples
//!
//! Generating position data for a sphere:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point3;
//! use plexus::prelude::*;
//! use plexus::primitive::sphere::UvSphere;
//!
//! # fn main() {
//! let sphere = UvSphere::new(16, 16);
//! // Generate the unique set of positional vertices.
//! // Convert the position data into `Point3<f32>` and collect it into a buffer.
//! let positions = sphere
//!     .vertices_with_position()
//!     .map(|position| -> Point3<f32> { position.into() })
//!     .collect::<Vec<_>>();
//! // Generate polygons indexing the unique set of positional vertices.
//! // These indeces depend on the attribute (position, normal, etc.).
//! // Decompose the indexing polygons into triangles and vertices and then collect
//! // the index data into a buffer.
//! let indices = sphere
//!     .indices_for_position()
//!     .triangulate()
//!     .vertices()
//!     .collect::<Vec<_>>();
//! # }
//! ```
//! Generating position data for a cube using an indexer:
//!
//! ```rust
//! use plexus::index::{Flat3, HashIndexer};
//! use plexus::prelude::*;
//! use plexus::primitive::cube::{Bounds, Cube};
//!
//! let (indices, positions) = Cube::new()
//!     .polygons_with_position_from(Bounds::unit_radius())
//!     .triangulate()
//!     .index_vertices::<Flat3, _>(HashIndexer::default());
//! ```

pub mod cube;
pub mod decompose;
pub mod generate;
pub mod sphere;

use arrayvec::ArrayVec;
use itertools::structs::Zip as OuterZip; // Avoid collision with `Zip`.
use num::Integer;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::{Index, Range};

use crate::primitive::decompose::IntoVertices;

pub trait Topological: Sized {
    type Vertex;

    fn arity(&self) -> usize;
}

pub trait Polygonal: Topological {}

pub trait ConstantArity {
    const ARITY: usize;
}

pub trait Converged: Topological {
    fn converged(value: Self::Vertex) -> Self;
}

pub trait Rotate {
    fn rotate(self, n: isize) -> Self;
}

pub trait Map<U = <Self as Topological>::Vertex>: Topological {
    type Output: Topological<Vertex = U>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Vertex) -> U;
}

pub trait Zip {
    type Output: FromIterator<<Self::Output as Topological>::Vertex> + Topological;

    fn zip(self) -> Self::Output;
}

macro_rules! zip {
    (topology => $t:ident, geometries => ($($g:ident),*)) => (
        #[allow(non_snake_case)]
        impl<$($g),*> Zip for ($($t<$g>),*) {
            type Output = $t<($($g),*)>;

            fn zip(self) -> Self::Output {
                let ($($g,)*) = self;
                izip!($($g.into_vertices()),*).collect()
            }
        }
    );
}

pub trait MapVertices<T, U>: Sized {
    fn map_vertices<F>(self, f: F) -> InteriorMap<Self, T, U, F>
    where
        F: FnMut(T) -> U;
}

impl<I, T, U, P, Q> MapVertices<T, U> for I
where
    I: Iterator<Item = P>,
    P: Map<U, Output = Q> + Topological<Vertex = T>,
    Q: Topological<Vertex = U>,
{
    fn map_vertices<F>(self, f: F) -> InteriorMap<Self, T, U, F>
    where
        F: FnMut(T) -> U,
    {
        InteriorMap::new(self, f)
    }
}

pub struct InteriorMap<I, T, U, F> {
    input: I,
    f: F,
    phantom: PhantomData<(T, U)>,
}

impl<I, T, U, F> InteriorMap<I, T, U, F> {
    fn new(input: I, f: F) -> Self {
        InteriorMap {
            input,
            f,
            phantom: PhantomData,
        }
    }
}

impl<I, T, U, F, P, Q> Iterator for InteriorMap<I, T, U, F>
where
    I: Iterator<Item = P>,
    F: FnMut(T) -> U,
    P: Map<U, Output = Q> + Topological<Vertex = T>,
    Q: Topological<Vertex = U>,
{
    type Item = Q;

    fn next(&mut self) -> Option<Self::Item> {
        self.input.next().map(|topology| topology.map(&mut self.f))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.input.size_hint()
    }
}

struct Iter<'a, T>
where
    T: 'a + Index<usize, Output = <T as Topological>::Vertex> + Topological,
{
    topology: &'a T,
    range: Range<usize>,
}

impl<'a, T> Iter<'a, T>
where
    T: 'a + Index<usize, Output = <T as Topological>::Vertex> + Topological,
{
    fn new(topology: &'a T, n: usize) -> Self {
        Iter {
            topology,
            range: 0..n,
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T>
where
    T: 'a + Index<usize, Output = <T as Topological>::Vertex> + Topological,
{
    type Item = &'a T::Vertex;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|index| self.topology.index(index))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Edge<T> {
    pub a: T,
    pub b: T,
}

impl<T> Edge<T> {
    pub fn new(a: T, b: T) -> Self {
        Edge { a, b }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        // For polygons (including digons), the vertex count and edge count
        // (arity) are the same. For monogons, these values diverge: an edge
        // connects two vertices with only one edge (itself).
        Iter::new(self, 2)
    }
}

impl<T> ConstantArity for Edge<T> {
    const ARITY: usize = 1;
}

impl<T> Converged for Edge<T>
where
    T: Clone,
{
    fn converged(value: T) -> Self {
        Edge::new(value.clone(), value)
    }
}

impl<T> FromIterator<T> for Edge<T> {
    fn from_iter<I>(input: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut input = input.into_iter();
        Edge::new(input.next().unwrap(), input.next().unwrap())
    }
}
zip!(topology => Edge, geometries => (A, B));
zip!(topology => Edge, geometries => (A, B, C));
zip!(topology => Edge, geometries => (A, B, C, D));

impl<T> Index<usize> for Edge<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.a,
            1 => &self.b,
            _ => panic!(),
        }
    }
}

impl<T, U> Map<U> for Edge<T> {
    type Output = Edge<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(T) -> U,
    {
        let Edge { a, b } = self;
        Edge::new(f(a), f(b))
    }
}

impl<T> Topological for Edge<T> {
    type Vertex = T;

    fn arity(&self) -> usize {
        Self::ARITY
    }
}

impl<T> Rotate for Edge<T> {
    fn rotate(self, n: isize) -> Self {
        if n % 2 != 0 {
            let Edge { a, b } = self;
            Edge { b, a }
        }
        else {
            self
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Triangle<T> {
    pub a: T,
    pub b: T,
    pub c: T,
}

impl<T> Triangle<T> {
    pub fn new(a: T, b: T, c: T) -> Self {
        Triangle { a, b, c }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        Iter::new(self, self.arity())
    }
}

impl<T> ConstantArity for Triangle<T> {
    const ARITY: usize = 3;
}

impl<T> Converged for Triangle<T>
where
    T: Clone,
{
    fn converged(value: T) -> Self {
        Triangle::new(value.clone(), value.clone(), value)
    }
}

impl<T> FromIterator<T> for Triangle<T> {
    fn from_iter<I>(input: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut input = input.into_iter();
        Triangle::new(
            input.next().unwrap(),
            input.next().unwrap(),
            input.next().unwrap(),
        )
    }
}
zip!(topology => Triangle, geometries => (A, B));
zip!(topology => Triangle, geometries => (A, B, C));
zip!(topology => Triangle, geometries => (A, B, C, D));

impl<T> Index<usize> for Triangle<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.a,
            1 => &self.b,
            2 => &self.c,
            _ => panic!(),
        }
    }
}

impl<T, U> Map<U> for Triangle<T> {
    type Output = Triangle<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(T) -> U,
    {
        let Triangle { a, b, c } = self;
        Triangle::new(f(a), f(b), f(c))
    }
}

impl<T> Topological for Triangle<T> {
    type Vertex = T;

    fn arity(&self) -> usize {
        Self::ARITY
    }
}

impl<T> Polygonal for Triangle<T> {}

impl<T> Rotate for Triangle<T> {
    fn rotate(self, n: isize) -> Self {
        let n = umod(n, Self::ARITY as isize);
        if n == 1 {
            let Triangle { a, b, c } = self;
            Triangle { b, c, a }
        }
        else if n == 2 {
            let Triangle { a, b, c } = self;
            Triangle { c, a, b }
        }
        else {
            self
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Quad<T> {
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
}

impl<T> Quad<T> {
    pub fn new(a: T, b: T, c: T, d: T) -> Self {
        Quad { a, b, c, d }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        Iter::new(self, self.arity())
    }
}

impl<T> ConstantArity for Quad<T> {
    const ARITY: usize = 4;
}

impl<T> Converged for Quad<T>
where
    T: Clone,
{
    fn converged(value: T) -> Self {
        Quad::new(value.clone(), value.clone(), value.clone(), value)
    }
}

impl<T> FromIterator<T> for Quad<T> {
    fn from_iter<I>(input: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut input = input.into_iter();
        Quad::new(
            input.next().unwrap(),
            input.next().unwrap(),
            input.next().unwrap(),
            input.next().unwrap(),
        )
    }
}
zip!(topology => Quad, geometries => (A, B));
zip!(topology => Quad, geometries => (A, B, C));
zip!(topology => Quad, geometries => (A, B, C, D));

impl<T> Index<usize> for Quad<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.a,
            1 => &self.b,
            2 => &self.c,
            3 => &self.d,
            _ => panic!(),
        }
    }
}

impl<T, U> Map<U> for Quad<T> {
    type Output = Quad<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(T) -> U,
    {
        let Quad { a, b, c, d } = self;
        Quad::new(f(a), f(b), f(c), f(d))
    }
}

impl<T> Topological for Quad<T> {
    type Vertex = T;

    fn arity(&self) -> usize {
        Self::ARITY
    }
}

impl<T> Polygonal for Quad<T> {}

impl<T> Rotate for Quad<T> {
    fn rotate(self, n: isize) -> Self {
        let n = umod(n, Self::ARITY as isize);
        if n == 1 {
            let Quad { a, b, c, d } = self;
            Quad { b, c, d, a }
        }
        else if n == 2 {
            let Quad { a, b, c, d } = self;
            Quad { c, d, a, b }
        }
        else if n == 3 {
            let Quad { a, b, c, d } = self;
            Quad { d, a, b, c }
        }
        else {
            self
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Polygon<T> {
    Triangle(Triangle<T>),
    Quad(Quad<T>),
}

impl<T> Polygon<T> {
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        Iter::new(self, self.arity())
    }
}

impl<T> From<Triangle<T>> for Polygon<T> {
    fn from(triangle: Triangle<T>) -> Self {
        Polygon::Triangle(triangle)
    }
}

impl<T> From<Quad<T>> for Polygon<T> {
    fn from(quad: Quad<T>) -> Self {
        Polygon::Quad(quad)
    }
}

impl<T> FromIterator<T> for Polygon<T> {
    fn from_iter<I>(input: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let input = input
            .into_iter()
            .take(Quad::<T>::ARITY)
            .collect::<ArrayVec<[T; 4]>>();
        match input.len() {
            Triangle::<T>::ARITY => Polygon::Triangle(Triangle::from_iter(input)),
            Quad::<T>::ARITY => Polygon::Quad(Quad::from_iter(input)),
            _ => panic!(),
        }
    }
}
zip!(topology => Polygon, geometries => (A, B));
zip!(topology => Polygon, geometries => (A, B, C));
zip!(topology => Polygon, geometries => (A, B, C, D));

impl<T> Index<usize> for Polygon<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match *self {
            Polygon::Triangle(ref triangle) => triangle.index(index),
            Polygon::Quad(ref quad) => quad.index(index),
        }
    }
}

impl<T, U> Map<U> for Polygon<T> {
    type Output = Polygon<U>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(T) -> U,
    {
        match self {
            Polygon::Triangle(triangle) => Polygon::Triangle(triangle.map(f)),
            Polygon::Quad(quad) => Polygon::Quad(quad.map(f)),
        }
    }
}

impl<T> Topological for Polygon<T> {
    type Vertex = T;

    fn arity(&self) -> usize {
        match *self {
            Polygon::Triangle(..) => Triangle::<T>::ARITY,
            Polygon::Quad(..) => Quad::<T>::ARITY,
        }
    }
}

impl<T> Polygonal for Polygon<T> {}

impl<T> Rotate for Polygon<T> {
    fn rotate(self, n: isize) -> Self {
        match self {
            Polygon::Triangle(triangle) => Polygon::Triangle(triangle.rotate(n)),
            Polygon::Quad(quad) => Polygon::Quad(quad.rotate(n)),
        }
    }
}

/// Zips the vertices and topologies from multiple iterators into a single
/// iterator.
///
/// This is useful for zipping different attributes of a primitive generator.
/// For example, it can be used to combine position, plane, and UV-mapping data
/// data of a cube into a single topology stream.
///
/// # Examples
///
/// Create a topological stream of position and UV-mapping data for a cube:
///
/// ```rust
/// # extern crate decorum;
/// # extern crate num;
/// # extern crate plexus;
/// #
/// use decorum::R64;
/// use plexus::prelude::*;
/// use plexus::primitive;
/// use plexus::primitive::cube::Cube;
///
/// fn map_uv_to_color(uv: &Duplet<R64>) -> Triplet<R64> {
/// #     use num::One;
/// #     Triplet(One::one(), One::one(), One::one())
///     // ...
/// }
///
/// # fn main() {
/// let cube = Cube::new();
/// let polygons =
///     // Zip positions and texture coordinates into each vertex.
///     primitive::zip_vertices((cube.polygons_with_position(), cube.polygons_with_uv_map()))
///         .map_vertices(|(position, uv)| (position, uv, map_uv_to_color(&uv)))
///         .triangulate()
///         .collect::<Vec<_>>();
/// # }
/// ```
pub fn zip_vertices<T, U>(
    tuple: U,
) -> impl Iterator<Item = <<OuterZip<T> as Iterator>::Item as Zip>::Output>
where
    OuterZip<T>: From<U> + Iterator,
    <OuterZip<T> as Iterator>::Item: Zip,
{
    OuterZip::from(tuple).map(|item| item.zip())
}

fn umod<T>(n: T, m: T) -> T
where
    T: Copy + Integer,
{
    ((n % m) + m) % m
}
