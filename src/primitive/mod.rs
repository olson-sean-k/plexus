//! Primitive topological structures.
//!
//! This module provides primitives that can be composed to form more complex
//! structures. This includes simple topological structures like `Triangles`,
//! generators that form shapes from those structures, and iterator expressions
//! that compose and decompose streams.
//!
//! Much functionality in this module is exposed via traits. Many of these
//! traits are included in the `prelude` module, and it is highly recommended
//! to import the `prelude`'s contents as seen in the examples.
//!
//! # Examples
//!
//! Generating position data for a sphere:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use nalgebra::Point3;
//! use plexus::prelude::*;
//! use plexus::primitive::sphere::UvSphere;
//!
//! # fn main() {
//! let sphere = UvSphere::new(16, 16);
//! // Generate the unique set of positional vertices.
//! let positions = sphere
//!     .vertices_with_position::<Point3<f64>>()
//!     .collect::<Vec<_>>();
//! // Generate polygons indexing the unique set of positional vertices.
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
//! # extern crate decorum;
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use decorum::N64;
//! use nalgebra::Point3;
//! use plexus::index::{Flat3, HashIndexer};
//! use plexus::prelude::*;
//! use plexus::primitive::cube::{Bounds, Cube};
//!
//! # fn main() {
//! let (indices, positions) = Cube::new()
//!     .polygons_with_position_from::<Point3<N64>>(Bounds::unit_radius())
//!     .triangulate()
//!     .index_vertices::<Flat3, _>(HashIndexer::default());
//! # }
//! ```

pub mod cube;
pub mod decompose;
pub mod generate;
pub mod sphere;

use arrayvec::ArrayVec;
use itertools::structs::Zip as OuterZip; // Avoid collision with `Zip`.
use num::Integer;
use smallvec::SmallVec;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::{Index, Range};
use theon::{Category, FromObjects, IntoObjects};

pub use theon::ops::{Map, Reduce, ZipMap};
pub use theon::Converged;

use crate::primitive::decompose::IntoVertices;

pub trait Topological: Category<Object = <Self as Topological>::Vertex> + Sized {
    type Vertex;

    fn arity(&self) -> usize;
}

pub trait Polygonal: Topological {}

pub trait ConstantArity {
    const ARITY: usize;
}

pub trait Rotate {
    fn rotate(self, n: isize) -> Self;
}

pub trait Zip {
    type Output: FromObjects + Topological;

    fn zip(self) -> Self::Output;
}
macro_rules! impl_zip {
    (category => $c:ident) => (
        impl_zip!(category => $c, objects => (A, B));
        impl_zip!(category => $c, objects => (A, B, C));
        impl_zip!(category => $c, objects => (A, B, C, D));
        impl_zip!(category => $c, objects => (A, B, C, D, E));
        impl_zip!(category => $c, objects => (A, B, C, D, E, F));
    );
    (category => $c:ident, objects => ($($o:ident),*)) => (
        #[allow(non_snake_case)]
        impl<$($o),*> Zip for ($($c<$o>),*) {
            type Output = $c<($($o),*)>;

            fn zip(self) -> Self::Output {
                let ($($o,)*) = self;
                FromObjects::from_objects(izip!($($o.into_objects()),*)).unwrap()
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

impl<T> Category for Edge<T> {
    type Object = T;
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
        Edge::from_objects(input).unwrap()
    }
}

impl<T> FromObjects for Edge<T> {
    fn from_objects<I>(objects: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Object>,
    {
        let mut objects = objects.into_iter().take(2);
        match (objects.next(), objects.next()) {
            (Some(a), Some(b)) => Some(Edge::new(a, b)),
            _ => None,
        }
    }
}

impl<T> IntoObjects for Edge<T> {
    type Output = ArrayVec<[T; 2]>;

    fn into_objects(self) -> Self::Output {
        ArrayVec::from([self.a, self.b])
    }
}

impl_zip!(category => Edge);

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
        F: FnMut(Self::Object) -> U,
    {
        let Edge { a, b } = self;
        Edge::new(f(a), f(b))
    }
}

impl<T, U> Reduce<T, U> for Edge<T> {
    fn reduce<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, T) -> U,
    {
        for vertex in self.into_vertices() {
            seed = f(seed, vertex);
        }
        seed
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

impl<T> Topological for Edge<T> {
    type Vertex = T;

    fn arity(&self) -> usize {
        Self::ARITY
    }
}

impl<T, U> ZipMap<U> for Edge<T> {
    type Output = Edge<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Object, Self::Object) -> U,
    {
        Edge::new(f(self.a, other.a), f(self.b, other.b))
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

impl<T> Category for Triangle<T> {
    type Object = T;
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
        Triangle::from_objects(input).unwrap()
    }
}

impl<T> FromObjects for Triangle<T> {
    fn from_objects<I>(objects: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Object>,
    {
        let mut objects = objects.into_iter().take(3);
        match (objects.next(), objects.next(), objects.next()) {
            (Some(a), Some(b), Some(c)) => Some(Triangle::new(a, b, c)),
            _ => None,
        }
    }
}

impl<T> IntoObjects for Triangle<T> {
    type Output = ArrayVec<[T; 3]>;

    fn into_objects(self) -> Self::Output {
        ArrayVec::from([self.a, self.b, self.c])
    }
}

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
        F: FnMut(Self::Object) -> U,
    {
        let Triangle { a, b, c } = self;
        Triangle::new(f(a), f(b), f(c))
    }
}

impl<T> Polygonal for Triangle<T> {}

impl<T, U> Reduce<T, U> for Triangle<T> {
    fn reduce<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, T) -> U,
    {
        for vertex in self.into_vertices() {
            seed = f(seed, vertex);
        }
        seed
    }
}

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

impl<T> Topological for Triangle<T> {
    type Vertex = T;

    fn arity(&self) -> usize {
        Self::ARITY
    }
}

impl_zip!(category => Triangle);

impl<T, U> ZipMap<U> for Triangle<T> {
    type Output = Triangle<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Object, Self::Object) -> U,
    {
        Triangle::new(f(self.a, other.a), f(self.b, other.b), f(self.c, other.c))
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

impl<T> Category for Quad<T> {
    type Object = T;
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
        Quad::from_objects(input).unwrap()
    }
}

impl<T> FromObjects for Quad<T> {
    fn from_objects<I>(objects: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Object>,
    {
        let mut objects = objects.into_iter().take(4);
        match (
            objects.next(),
            objects.next(),
            objects.next(),
            objects.next(),
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => Some(Quad::new(a, b, c, d)),
            _ => None,
        }
    }
}

impl<T> IntoObjects for Quad<T> {
    type Output = ArrayVec<[T; 4]>;

    fn into_objects(self) -> Self::Output {
        ArrayVec::from([self.a, self.b, self.c, self.d])
    }
}

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
        F: FnMut(Self::Object) -> U,
    {
        let Quad { a, b, c, d } = self;
        Quad::new(f(a), f(b), f(c), f(d))
    }
}

impl<T> Polygonal for Quad<T> {}

impl<T, U> Reduce<T, U> for Quad<T> {
    fn reduce<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, T) -> U,
    {
        for vertex in self.into_vertices() {
            seed = f(seed, vertex);
        }
        seed
    }
}

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

impl<T> Topological for Quad<T> {
    type Vertex = T;

    fn arity(&self) -> usize {
        Self::ARITY
    }
}

impl_zip!(category => Quad);

impl<T, U> ZipMap<U> for Quad<T> {
    type Output = Quad<U>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Object, Self::Object) -> U,
    {
        Quad::new(
            f(self.a, other.a),
            f(self.b, other.b),
            f(self.c, other.c),
            f(self.d, other.d),
        )
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

// TODO: This may be absusive of the `Category` trait. Note that at runtime,
//       two instances of `Polygon<T>` may not be equivalent categories as
//       their structures may differ (`Triangle` vs.  `Quad`).
//
//       Observe that it is not possible to implement `ZipMap`.
impl<T> Category for Polygon<T> {
    type Object = T;
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
        Polygon::from_objects(input).unwrap()
    }
}

impl<T> FromObjects for Polygon<T> {
    fn from_objects<I>(objects: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Object>,
    {
        let objects = objects
            .into_iter()
            .take(Quad::<T>::ARITY)
            .collect::<ArrayVec<[T; 4]>>();
        match objects.len() {
            Triangle::<T>::ARITY => Triangle::from_objects(objects).map(|triangle| triangle.into()),
            Quad::<T>::ARITY => Quad::from_objects(objects).map(|quad| quad.into()),
            _ => None,
        }
    }
}

impl<T> Index<usize> for Polygon<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match *self {
            Polygon::Triangle(ref triangle) => triangle.index(index),
            Polygon::Quad(ref quad) => quad.index(index),
        }
    }
}

impl<T> IntoObjects for Polygon<T> {
    type Output = SmallVec<[T; 4]>;

    fn into_objects(self) -> Self::Output {
        match self {
            Polygon::Triangle(triangle) => triangle.into_objects().into_iter().collect(),
            Polygon::Quad(quad) => quad.into_objects().into_iter().collect(),
        }
    }
}

impl<T, U> Map<U> for Polygon<T> {
    type Output = Polygon<U>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Object) -> U,
    {
        match self {
            Polygon::Triangle(triangle) => Polygon::Triangle(triangle.map(f)),
            Polygon::Quad(quad) => Polygon::Quad(quad.map(f)),
        }
    }
}

impl<T> Polygonal for Polygon<T> {}

impl<T, U> Reduce<T, U> for Polygon<T> {
    fn reduce<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, T) -> U,
    {
        for vertex in self.into_vertices() {
            seed = f(seed, vertex);
        }
        seed
    }
}

impl<T> Rotate for Polygon<T> {
    fn rotate(self, n: isize) -> Self {
        match self {
            Polygon::Triangle(triangle) => Polygon::Triangle(triangle.rotate(n)),
            Polygon::Quad(quad) => Polygon::Quad(quad.rotate(n)),
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

impl_zip!(category => Polygon);

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
/// # extern crate nalgebra;
/// # extern crate num;
/// # extern crate plexus;
/// # extern crate theon;
/// #
/// use decorum::{N64, R64};
/// use nalgebra::{Point2, Point3, Vector4};
/// # use num::Zero;
/// use plexus::prelude::*;
/// use plexus::primitive;
/// use plexus::primitive::cube::Cube;
/// use theon::space::Vector;
///
/// # fn main() {
/// type E2 = Point2<N64>;
/// type E3 = Point3<N64>;
///
/// fn map_uv_to_rgb(uv: &Vector<E2>) -> Vector4<R64> {
/// #   Zero::zero()
///     // ...
/// }
///
/// let cube = Cube::new();
/// // Zip positions and texture coordinates into each vertex.
/// let polygons = primitive::zip_vertices((
///     cube.polygons_with_position::<E3>(),
///     cube.polygons_with_uv_map::<E2>(),
/// ))
///     .map_vertices(|(position, uv)| (position, uv, map_uv_to_rgb(&uv)))
///     .triangulate()
///     .collect::<Vec<_>>();
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
