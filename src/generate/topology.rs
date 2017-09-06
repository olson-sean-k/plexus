use num::{Float, Integer};
use ordered_float::OrderedFloat;
use std::marker::PhantomData;
use std::mem;

use generate::geometry::{FromOrderedFloat, FromUnorderedFloat, Unit, Vector};

// TODO: Traits like `MapVertices` and `Ordered` are concerned with geometry.
//       Should they be moved to the `geometry` module?

pub trait Topological: Sized {
    type Vertex: Clone;
}

pub trait Polygonal: Topological {}

pub trait MapVerticesInto<T, U>: Topological<Vertex = T>
where
    T: Clone,
    U: Clone,
{
    type Output: Topological<Vertex = U>;

    fn map_vertices_into<F>(self, f: F) -> Self::Output
    where
        F: FnMut(T) -> U;
}

pub trait MapVertices<T, U>: Sized
where
    T: Clone,
    U: Clone,
{
    fn map_vertices<F>(self, f: F) -> Map<Self, T, U, F>
    where
        F: FnMut(T) -> U;
}

pub trait Rotate {
    fn rotate(&mut self, n: isize);
}

pub trait Ordered<T>: Iterator + Sized
where
    T: Clone + Vector,
    T::Scalar: Clone + Float + Unit,
{
    fn ordered<U>(self) -> Map<Self, T, U::Output, fn(T) -> U::Output>
    where
        Self::Item: MapVerticesInto<T, U::Output>,
        <Self::Item as MapVerticesInto<T, U::Output>>::Output: Topological<Vertex = U::Output>,
        U: FromUnorderedFloat<T>,
        U::Output: Clone;
}

pub trait Unordered<T, S>: Iterator + Sized
where
    T: Clone + Vector<Scalar = OrderedFloat<S>>,
    S: Clone + Float + Unit,
{
    fn unordered<U>(self) -> Map<Self, T, U::Output, fn(T) -> U::Output>
    where
        Self::Item: MapVerticesInto<T, U::Output>,
        <Self::Item as MapVerticesInto<T, U::Output>>::Output: Topological<Vertex = U::Output>,
        U: FromOrderedFloat<T, S>,
        U::Output: Clone;
}

impl<I, T, U, P, Q> MapVertices<T, U> for I
where
    I: Iterator<Item = P>,
    P: MapVerticesInto<T, U, Output = Q>,
    Q: Topological<Vertex = U>,
    T: Clone,
    U: Clone,
{
    fn map_vertices<F>(self, f: F) -> Map<Self, T, U, F>
    where
        F: FnMut(T) -> U,
    {
        Map::new(self, f)
    }
}

impl<I, T> Ordered<T> for I
where
    I: Iterator,
    T: Clone + Vector,
    T::Scalar: Float + Unit,
{
    fn ordered<U>(self) -> Map<Self, T, U::Output, fn(T) -> U::Output>
    where
        Self::Item: MapVerticesInto<T, U::Output>,
        <Self::Item as MapVerticesInto<T, U::Output>>::Output: Topological<Vertex = U::Output>,
        U: FromUnorderedFloat<T>,
        U::Output: Clone,
    {
        fn f<T, U>(value: T) -> U::Output
        where
            T: Vector,
            T::Scalar: Float + Unit,
            U: FromUnorderedFloat<T>,
        {
            <U as FromUnorderedFloat<T>>::from_unordered_float(value)
        }
        self.map_vertices(f::<T, U>)
    }
}

impl<I, T, S> Unordered<T, S> for I
where
    I: Iterator,
    T: Clone + Vector<Scalar = OrderedFloat<S>>,
    S: Clone + Float + Unit,
{
    fn unordered<U>(self) -> Map<Self, T, U::Output, fn(T) -> U::Output>
    where
        Self::Item: MapVerticesInto<T, U::Output>,
        <Self::Item as MapVerticesInto<T, U::Output>>::Output: Topological<Vertex = U::Output>,
        U: FromOrderedFloat<T, S>,
        U::Output: Clone,
    {
        fn f<T, U, S>(value: T) -> U::Output
        where
            T: Vector<Scalar = OrderedFloat<S>>,
            S: Float + Unit,
            U: FromOrderedFloat<T, S>,
        {
            <U as FromOrderedFloat<T, S>>::from_ordered_float(value)
        }
        self.map_vertices(f::<T, U, S>)
    }
}

pub struct Map<I, T, U, F> {
    input: I,
    f: F,
    phantom: PhantomData<(T, U)>,
}

impl<I, T, U, F> Map<I, T, U, F> {
    fn new(input: I, f: F) -> Self {
        Map {
            input: input,
            f: f,
            phantom: PhantomData,
        }
    }
}

impl<I, T, U, F, P, Q> Iterator for Map<I, T, U, F>
where
    I: Iterator<Item = P>,
    F: FnMut(T) -> U,
    P: MapVerticesInto<T, U, Output = Q>,
    Q: Topological<Vertex = U>,
    T: Clone,
    U: Clone,
{
    type Item = Q;

    fn next(&mut self) -> Option<Self::Item> {
        self.input
            .next()
            .map(|topology| topology.map_vertices_into(&mut self.f))
    }
}

pub struct Line<T> {
    pub a: T,
    pub b: T,
}

impl<T> Line<T> {
    pub fn new(a: T, b: T) -> Self {
        Line { a: a, b: b }
    }

    pub fn converged(value: T) -> Self
    where
        T: Clone,
    {
        Line::new(value.clone(), value)
    }
}

impl<T, U> MapVerticesInto<T, U> for Line<T>
where
    T: Clone,
    U: Clone,
{
    type Output = Line<U>;

    fn map_vertices_into<F>(self, mut f: F) -> Self::Output
    where
        F: FnMut(T) -> U,
    {
        let Line { a, b } = self;
        Line::new(f(a), f(b))
    }
}

impl<T> Topological for Line<T>
where
    T: Clone,
{
    type Vertex = T;
}

impl<T> Rotate for Line<T>
where
    T: Clone,
{
    fn rotate(&mut self, n: isize) {
        if n % 2 != 0 {
            mem::swap(&mut self.a, &mut self.b);
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
        Triangle { a: a, b: b, c: c }
    }

    pub fn converged(value: T) -> Self
    where
        T: Clone,
    {
        Triangle::new(value.clone(), value.clone(), value)
    }
}

impl<T, U> MapVerticesInto<T, U> for Triangle<T>
where
    T: Clone,
    U: Clone,
{
    type Output = Triangle<U>;

    fn map_vertices_into<F>(self, mut f: F) -> Self::Output
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

impl<T> Polygonal for Triangle<T>
where
    T: Clone,
{
}

impl<T> Rotate for Triangle<T>
where
    T: Clone,
{
    fn rotate(&mut self, n: isize) {
        let n = umod(n, 3);
        if n == 1 {
            mem::swap(&mut self.a, &mut self.b);
            mem::swap(&mut self.b, &mut self.c);
        }
        else if n == 2 {
            mem::swap(&mut self.c, &mut self.b);
            mem::swap(&mut self.b, &mut self.a);
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
        Quad {
            a: a,
            b: b,
            c: c,
            d: d,
        }
    }

    pub fn converged(value: T) -> Self
    where
        T: Clone,
    {
        Quad::new(value.clone(), value.clone(), value.clone(), value)
    }
}

impl<T, U> MapVerticesInto<T, U> for Quad<T>
where
    T: Clone,
    U: Clone,
{
    type Output = Quad<U>;

    fn map_vertices_into<F>(self, mut f: F) -> Self::Output
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

impl<T> Polygonal for Quad<T>
where
    T: Clone,
{
}

impl<T> Rotate for Quad<T>
where
    T: Clone,
{
    fn rotate(&mut self, n: isize) {
        let n = umod(n, 4);
        if n == 1 {
            mem::swap(&mut self.a, &mut self.b);
            mem::swap(&mut self.b, &mut self.c);
            mem::swap(&mut self.c, &mut self.d);
        }
        else if n == 2 {
            mem::swap(&mut self.a, &mut self.c);
            mem::swap(&mut self.b, &mut self.d);
        }
        else if n == 3 {
            mem::swap(&mut self.d, &mut self.c);
            mem::swap(&mut self.c, &mut self.b);
            mem::swap(&mut self.b, &mut self.a);
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

impl<T, U> MapVerticesInto<T, U> for Polygon<T>
where
    T: Clone,
    U: Clone,
{
    type Output = Polygon<U>;

    fn map_vertices_into<F>(self, f: F) -> Self::Output
    where
        F: FnMut(T) -> U,
    {
        match self {
            Polygon::Triangle(triangle) => Polygon::Triangle(triangle.map_vertices_into(f)),
            Polygon::Quad(quad) => Polygon::Quad(quad.map_vertices_into(f)),
        }
    }
}

impl<T> Topological for Polygon<T>
where
    T: Clone,
{
    type Vertex = T;
}

impl<T> Polygonal for Polygon<T>
where
    T: Clone,
{
}

impl<T> Rotate for Polygon<T>
where
    T: Clone,
{
    fn rotate(&mut self, n: isize) {
        match *self {
            Polygon::Triangle(ref mut triangle) => triangle.rotate(n),
            Polygon::Quad(ref mut quad) => quad.rotate(n),
        }
    }
}

fn umod<T>(n: T, m: T) -> T
where
    T: Copy + Integer,
{
    ((n % m) + m) % m
}
