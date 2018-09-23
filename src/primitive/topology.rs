use arrayvec::ArrayVec;
use itertools::structs::Zip as ItemZip; // Avoid collision with `Zip`.
use num::Integer;
use std::iter::FromIterator;
use std::marker::PhantomData;

use primitive::decompose::IntoVertices;

pub trait Topological: Sized {
    type Vertex: Clone;
}

pub trait Polygonal: Topological {}

pub trait Arity {
    const ARITY: usize;
}

pub trait Converged: Topological {
    fn converged(value: Self::Vertex) -> Self;
}

pub trait Rotate {
    fn rotate(self, n: isize) -> Self;
}

pub trait Map<U>: Topological
where
    U: Clone,
{
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
        impl<$($g: Clone),*> Zip for ($($t<$g>),*) {
            type Output = $t<($($g),*)>;

            fn zip(self) -> Self::Output {
                let ($($g,)*) = self;
                izip!($($g.into_vertices()),*).collect()
            }
        }
    );
}

pub trait MapVertices<T, U>: Sized
where
    T: Clone,
    U: Clone,
{
    fn map_vertices<F>(self, f: F) -> FlatMap<Self, T, U, F>
    where
        F: FnMut(T) -> U;
}

impl<I, T, U, P, Q> MapVertices<T, U> for I
where
    I: Iterator<Item = P>,
    P: Map<U, Output = Q> + Topological<Vertex = T>,
    Q: Topological<Vertex = U>,
    T: Clone,
    U: Clone,
{
    fn map_vertices<F>(self, f: F) -> FlatMap<Self, T, U, F>
    where
        F: FnMut(T) -> U,
    {
        FlatMap::new(self, f)
    }
}

pub struct FlatMap<I, T, U, F> {
    input: I,
    f: F,
    phantom: PhantomData<(T, U)>,
}

impl<I, T, U, F> FlatMap<I, T, U, F> {
    fn new(input: I, f: F) -> Self {
        FlatMap {
            input,
            f,
            phantom: PhantomData,
        }
    }
}

impl<I, T, U, F, P, Q> Iterator for FlatMap<I, T, U, F>
where
    I: Iterator<Item = P>,
    F: FnMut(T) -> U,
    P: Map<U, Output = Q> + Topological<Vertex = T>,
    Q: Topological<Vertex = U>,
    T: Clone,
    U: Clone,
{
    type Item = Q;

    fn next(&mut self) -> Option<Self::Item> {
        self.input.next().map(|topology| topology.map(&mut self.f))
    }
}

pub struct Edge<T> {
    pub a: T,
    pub b: T,
}

impl<T> Edge<T> {
    pub fn new(a: T, b: T) -> Self {
        Edge { a, b }
    }
}

impl<T> Arity for Edge<T> {
    const ARITY: usize = 2;
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

impl<T, U> Map<U> for Edge<T>
where
    T: Clone,
    U: Clone,
{
    type Output = Edge<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(T) -> U,
    {
        let Edge { a, b } = self;
        Edge::new(f(a), f(b))
    }
}

impl<T> Topological for Edge<T>
where
    T: Clone,
{
    type Vertex = T;
}

impl<T> Rotate for Edge<T>
where
    T: Clone,
{
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

pub struct Triangle<T> {
    pub a: T,
    pub b: T,
    pub c: T,
}

impl<T> Triangle<T> {
    pub fn new(a: T, b: T, c: T) -> Self {
        Triangle { a, b, c }
    }
}

impl<T> Arity for Triangle<T> {
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

impl<T, U> Map<U> for Triangle<T>
where
    T: Clone,
    U: Clone,
{
    type Output = Triangle<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(T) -> U,
    {
        let Triangle { a, b, c } = self;
        Triangle::new(f(a), f(b), f(c))
    }
}

impl<T> Topological for Triangle<T>
where
    T: Clone,
{
    type Vertex = T;
}

impl<T> Polygonal for Triangle<T> where T: Clone {}

impl<T> Rotate for Triangle<T>
where
    T: Clone,
{
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
}

impl<T> Arity for Quad<T> {
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

impl<T, U> Map<U> for Quad<T>
where
    T: Clone,
    U: Clone,
{
    type Output = Quad<U>;

    fn map<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(T) -> U,
    {
        let Quad { a, b, c, d } = self;
        Quad::new(f(a), f(b), f(c), f(d))
    }
}

impl<T> Topological for Quad<T>
where
    T: Clone,
{
    type Vertex = T;
}

impl<T> Polygonal for Quad<T> where T: Clone {}

impl<T> Rotate for Quad<T>
where
    T: Clone,
{
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

pub enum Polygon<T> {
    Triangle(Triangle<T>),
    Quad(Quad<T>),
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
        // Associated constants cannot be used in constant expressions, so the
        // size of the `ArrayVec` uses a literal instead of `Quad::<T>::ARITY`.
        let input = input
            .into_iter()
            .take(Quad::<T>::ARITY)
            .collect::<ArrayVec<[T; 4]>>();
        match input.len() {
            Quad::<T>::ARITY => Polygon::Quad(Quad::from_iter(input)),
            Triangle::<T>::ARITY => Polygon::Triangle(Triangle::from_iter(input)),
            _ => panic!(),
        }
    }
}
zip!(topology => Polygon, geometries => (A, B));
zip!(topology => Polygon, geometries => (A, B, C));
zip!(topology => Polygon, geometries => (A, B, C, D));

impl<T, U> Map<U> for Polygon<T>
where
    T: Clone,
    U: Clone,
{
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

impl<T> Topological for Polygon<T>
where
    T: Clone,
{
    type Vertex = T;
}

impl<T> Polygonal for Polygon<T> where T: Clone {}

impl<T> Rotate for Polygon<T>
where
    T: Clone,
{
    fn rotate(self, n: isize) -> Self {
        match self {
            Polygon::Triangle(triangle) => Polygon::Triangle(triangle.rotate(n)),
            Polygon::Quad(quad) => Polygon::Quad(quad.rotate(n)),
        }
    }
}

fn umod<T>(n: T, m: T) -> T
where
    T: Copy + Integer,
{
    ((n % m) + m) % m
}

/// Zips the vertices and topologies from multiple iterators into a single
/// iterator.
///
/// This is useful for zipping different attributes of a primitive generator.
/// For example, it can be used to combine position, plane, and texture
/// coordinate data of a cube into a single topology stream.
///
/// # Examples
///
/// Create a topological stream of position and texture coordinate data for a
/// cube:
///
/// ```rust
/// # extern crate num;
/// # extern crate plexus;
/// # use plexus::R32;
/// use plexus::prelude::*;
/// use plexus::primitive::cube::Cube;
/// use plexus::primitive;
///
/// # use num::One;
/// # fn map_uv_to_color(texture: &Duplet<R32>) -> Triplet<R32> {
/// #     Triplet(One::one(), One::one(), One::one())
/// # }
/// # fn main() {
/// let cube = Cube::new();
/// let polygons =
///     // Zip positions and texture coordinates into each vertex.
///     primitive::zip_vertices((cube.polygons_with_position(), cube.polygons_with_texture()))
///         .map_vertices(|(position, texture)| (position, texture, map_uv_to_color(&texture)))
///         .triangulate()
///         .collect::<Vec<_>>();
/// # }
/// ```
pub fn zip_vertices<T, U>(
    tuple: U,
) -> impl Iterator<Item = <<ItemZip<T> as Iterator>::Item as Zip>::Output>
where
    ItemZip<T>: From<U> + Iterator,
    <ItemZip<T> as Iterator>::Item: Zip,
{
    ItemZip::from(tuple).map(|item| item.zip())
}
