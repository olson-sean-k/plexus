//! Primitive topological structures.
//!
//! This module provides composable primitives like $n$-gons that can form more
//! complex structures. This includes simple topological structures like
//! `Triangles`, generators that form more complex shapes like spheres, and
//! iterator expressions that compose and decompose streams of topological and
//! geometric data.
//!
//! Much functionality in this module is exposed via traits, especially
//! generators. Many of these traits are included in the `prelude` module, and
//! it is highly recommended to import the `prelude`'s contents as seen in the
//! examples.
//!
//! # Examples
//!
//! Generating positional data for a sphere:
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
//! // Generate polygon that index the unique set of positional vertices.
//! let indices = sphere
//!     .indices_for_position()
//!     .triangulate()
//!     .vertices()
//!     .collect::<Vec<_>>();
//! # }
//! ```
//! Generating positional data for a cube using an indexer:
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
//! use plexus::primitive::cube::Cube;
//!
//! # fn main() {
//! let (indices, positions) = Cube::new()
//!     .polygons_with_position::<Point3<N64>>()
//!     .triangulate()
//!     .index_vertices::<Flat3, _>(HashIndexer::default());
//! # }
//! ```

pub mod cube;
pub mod decompose;
pub mod generate;
pub mod sphere;

use arrayvec::{Array, ArrayVec};
use itertools::izip;
use itertools::structs::Zip as OuterZip; // Avoid collision with `Zip`.
use num::Integer;
use smallvec::SmallVec;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::slice;
use theon::{Composite, FromItems, IntoItems};

pub use theon::ops::{Map, Reduce, ZipMap};
pub use theon::Converged;

use crate::primitive::decompose::IntoVertices;

pub trait Topological: Composite<Item = <Self as Topological>::Vertex> + Sized {
    type Vertex;

    fn arity(&self) -> usize;
}

pub trait Polygonal: Topological {}

pub trait ConstantArity {
    const ARITY: usize;
}

// TODO: It should be possible to implement this for all `NGon`s, but that
//       implementation would likely be inefficient.
pub trait Rotate {
    fn rotate(self, n: isize) -> Self;
}

pub trait Zip {
    type Output: FromItems + Topological;

    fn zip(self) -> Self::Output;
}
macro_rules! impl_zip {
    (composite => $c:ident, length => $n:expr) => (
        impl_zip!(composite => $c, length => $n, items => (A, B));
        impl_zip!(composite => $c, length => $n, items => (A, B, C));
        impl_zip!(composite => $c, length => $n, items => (A, B, C, D));
        impl_zip!(composite => $c, length => $n, items => (A, B, C, D, E));
        impl_zip!(composite => $c, length => $n, items => (A, B, C, D, E, F));
    );
    (composite => $c:ident, length => $n:expr, items => ($($i:ident),*)) => (
        #[allow(non_snake_case)]
        impl<$($i),*> Zip for ($($c<[$i; $n]>),*) {
            type Output = $c<[($($i),*); $n]>;

            fn zip(self) -> Self::Output {
                let ($($i,)*) = self;
                FromItems::from_items(izip!($($i.into_items()),*)).unwrap()
            }
        }
    );
    (composite => $c:ident) => (
        impl_zip!(composite => $c, items => (A, B));
        impl_zip!(composite => $c, items => (A, B, C));
        impl_zip!(composite => $c, items => (A, B, C, D));
        impl_zip!(composite => $c, items => (A, B, C, D, E));
        impl_zip!(composite => $c, items => (A, B, C, D, E, F));
    );
    (composite => $c:ident, items => ($($i:ident),*)) => (
        #[allow(non_snake_case)]
        impl<$($i),*> Zip for ($($c<$i>),*) {
            type Output = $c<($($i),*)>;

            fn zip(self) -> Self::Output {
                let ($($i,)*) = self;
                FromItems::from_items(izip!($($i.into_items()),*)).unwrap()
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

/// Statically sized $n$-gon.
///
/// `NGon` represents a topological structure as an array. Each array element
/// represents vertex data in order, with neighboring elements being connected
/// by an implicit undirected edge. For example, an `NGon` with three vertices
/// (`NGon<[T; 3]>`) would represent a triangle. Generally these elements are
/// labeled $A$, $B$, $C$, etc.
///
/// `NGon`s with less than three vertices are a degenerate case. An `NGon` with
/// two vertices (`NGon<[T; 2]>`) is considered a _monogon_ despite common
/// definitions specifying a single vertex. Such an `NGon` is not considered a
/// _digon_, as it represents a single undirected edge rather than two distinct
/// (but collapsed) edges. Single-vertex `NGon`s are unsupported. See the
/// `Edge` type definition.
///
/// Monogons and digons are not generally considered polygons, and `NGon` does
/// not implement the `Polygonal` trait in these cases.
#[derive(Clone, Copy, Debug)]
pub struct NGon<A>(pub A)
where
    A: Array;

impl<A> NGon<A>
where
    A: Array,
{
    pub fn into_array(self) -> A {
        self.0
    }

    fn into_array_vec(self) -> ArrayVec<A> {
        ArrayVec::from(self.into_array())
    }
}

/// Gets a slice over the data in an `NGon`.
///
/// Slicing an `NGon` can be used to iterate over references to its data:
///
/// ```rust
/// use plexus::primitive::{Converged, Triangle};
///
/// let triangle = Triangle::converged(0u32);
/// for vertex in triangle.as_ref() {
///     // ...
/// }
/// ```
impl<A> AsRef<[<A as Array>::Item]> for NGon<A>
where
    A: Array,
{
    fn as_ref(&self) -> &[A::Item] {
        unsafe { slice::from_raw_parts(self.0.as_ptr(), A::capacity()) }
    }
}

/// Gets a mutable slice over the data in an `NGon`.
///
/// Slicing an `NGon` can be used to iterate over references to its data:
///
/// ```rust
/// use plexus::primitive::{Converged, Quad};
///
/// let mut quad = Quad::converged(1u32);
/// for mut vertex in quad.as_mut() {
///     *vertex = 0;
/// }
/// ```
impl<A> AsMut<[<A as Array>::Item]> for NGon<A>
where
    A: Array,
{
    fn as_mut(&mut self) -> &mut [A::Item] {
        unsafe { slice::from_raw_parts_mut(self.0.as_mut_ptr(), A::capacity()) }
    }
}

impl<A> Composite for NGon<A>
where
    A: Array,
{
    type Item = A::Item;
}

impl<A> From<A> for NGon<A>
where
    A: Array,
{
    fn from(array: A) -> Self {
        NGon(array)
    }
}

impl<A> FromItems for NGon<A>
where
    A: Array,
{
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        items
            .into_iter()
            .collect::<ArrayVec<A>>()
            .into_inner()
            .ok()
            .map(|array| NGon(array))
    }
}

impl<A> Index<usize> for NGon<A>
where
    A: Array + AsRef<[<A as Array>::Item]>,
{
    type Output = A::Item;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.as_ref().index(index)
    }
}

impl<A> IndexMut<usize> for NGon<A>
where
    A: Array + AsRef<[<A as Array>::Item]> + AsMut<[<A as Array>::Item]>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.as_mut().index_mut(index)
    }
}

impl<A> IntoItems for NGon<A>
where
    A: Array,
{
    type Output = ArrayVec<A>;

    fn into_items(self) -> Self::Output {
        self.into_array_vec()
    }
}

impl<A> IntoIterator for NGon<A>
where
    A: Array,
{
    type Item = <A as Array>::Item;
    type IntoIter = <<Self as IntoItems>::Output as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.into_items().into_iter()
    }
}

impl<A, U> Reduce<U> for NGon<A>
where
    Self: Topological + IntoItems,
    A: Array,
{
    fn reduce<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        for vertex in self.into_vertices() {
            seed = f(seed, vertex);
        }
        seed
    }
}

macro_rules! impl_ngon {
    (length => $n:expr) => (
        impl<T> Converged for NGon<[T; $n]>
        where
            T: Copy,
        {
            fn converged(item: T) -> Self {
                NGon([item; $n])
            }
        }

        impl<T, U> Map<U> for NGon<[T; $n]> {
            type Output = NGon<[U; $n]>;

            fn map<F>(self, f: F) -> Self::Output
            where
                F: FnMut(Self::Item) -> U,
            {
                FromItems::from_items(self.into_iter().map(f)).unwrap()
            }
        }

        impl_zip!(composite => NGon, length => $n);

        impl<T, U> ZipMap<U> for NGon<[T; $n]> {
            type Output = NGon<[U; $n]>;

            fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
            where
                F: FnMut(Self::Item, Self::Item) -> U,
            {
                FromItems::from_items(self.into_iter().zip(other).map(|(a, b)| f(a, b))).unwrap()
            }
        }
    );
    (lengths => $($n:expr),*$(,)?) => (
        $(impl_ngon!(length => $n);)*
    );
}
impl_ngon!(lengths => 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

macro_rules! impl_polygonal_ngon {
    (length => $n:expr) => (
        impl<T> ConstantArity for NGon<[T; $n]> {
            const ARITY: usize = $n;
        }

        impl<T> Polygonal for NGon<[T; $n]> {}

        impl<T> Topological for NGon<[T; $n]> {
            type Vertex = T;

            fn arity(&self) -> usize {
                $n
            }
        }
    );
    (lengths => $($n:expr),*$(,)?) => (
        impl<T> ConstantArity for NGon<[T; 2]> {
            const ARITY: usize = 1;
        }

        impl<T> Topological for NGon<[T; 2]> {
            type Vertex = T;

            fn arity(&self) -> usize {
                1
            }
        }

        $(impl_polygonal_ngon!(length => $n);)*
    );
}
impl_polygonal_ngon!(lengths => 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

pub type Edge<T> = NGon<[T; 2]>;

impl<T> Edge<T> {
    pub fn new(a: T, b: T) -> Self {
        NGon([a, b])
    }
}

impl<T> Rotate for Edge<T> {
    fn rotate(self, n: isize) -> Self {
        if n % 2 != 0 {
            let [a, b] = self.into_array();
            Edge::new(b, a)
        }
        else {
            self
        }
    }
}

pub type Triangle<T> = NGon<[T; 3]>;

impl<T> Triangle<T> {
    pub fn new(a: T, b: T, c: T) -> Self {
        NGon([a, b, c])
    }
}

impl<T> Rotate for Triangle<T> {
    fn rotate(self, n: isize) -> Self {
        let n = umod(n, Self::ARITY as isize);
        if n == 1 {
            let [a, b, c] = self.into_array();
            Triangle::new(b, c, a)
        }
        else if n == 2 {
            let [a, b, c] = self.into_array();
            Triangle::new(c, a, b)
        }
        else {
            self
        }
    }
}

pub type Quad<T> = NGon<[T; 4]>;

impl<T> Quad<T> {
    pub fn new(a: T, b: T, c: T, d: T) -> Self {
        NGon([a, b, c, d])
    }
}

impl<T> Rotate for Quad<T> {
    fn rotate(self, n: isize) -> Self {
        let n = umod(n, Self::ARITY as isize);
        if n == 1 {
            let [a, b, c, d] = self.into_array();
            Quad::new(b, c, d, a)
        }
        else if n == 2 {
            let [a, b, c, d] = self.into_array();
            Quad::new(c, d, a, b)
        }
        else if n == 3 {
            let [a, b, c, d] = self.into_array();
            Quad::new(d, a, b, c)
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
    pub fn triangle(a: T, b: T, c: T) -> Self {
        Polygon::Triangle(Triangle::new(a, b, c))
    }

    pub fn quad(a: T, b: T, c: T, d: T) -> Self {
        Polygon::Quad(Quad::new(a, b, c, d))
    }
}

impl<T> AsRef<[T]> for Polygon<T> {
    fn as_ref(&self) -> &[T] {
        match *self {
            Polygon::Triangle(ref triangle) => triangle.as_ref(),
            Polygon::Quad(ref quad) => quad.as_ref(),
        }
    }
}

impl<T> AsMut<[T]> for Polygon<T> {
    fn as_mut(&mut self) -> &mut [T] {
        match *self {
            Polygon::Triangle(ref mut triangle) => triangle.as_mut(),
            Polygon::Quad(ref mut quad) => quad.as_mut(),
        }
    }
}

impl<T> Composite for Polygon<T> {
    type Item = T;
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

impl<T> FromItems for Polygon<T> {
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        let items = items
            .into_iter()
            .take(Quad::<T>::ARITY)
            .collect::<ArrayVec<[T; 4]>>();
        match items.len() {
            Triangle::<T>::ARITY => Triangle::from_items(items).map(|triangle| triangle.into()),
            Quad::<T>::ARITY => Quad::from_items(items).map(|quad| quad.into()),
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

impl<T> IndexMut<usize> for Polygon<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match *self {
            Polygon::Triangle(ref mut triangle) => triangle.index_mut(index),
            Polygon::Quad(ref mut quad) => quad.index_mut(index),
        }
    }
}

impl<T> IntoItems for Polygon<T> {
    type Output = SmallVec<[T; 4]>;

    fn into_items(self) -> Self::Output {
        match self {
            Polygon::Triangle(triangle) => triangle.into_items().into_iter().collect(),
            Polygon::Quad(quad) => quad.into_items().into_iter().collect(),
        }
    }
}

impl<T> IntoIterator for Polygon<T> {
    type Item = T;
    type IntoIter = <<Self as IntoItems>::Output as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.into_items().into_iter()
    }
}

impl<T, U> Map<U> for Polygon<T> {
    type Output = Polygon<U>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        match self {
            Polygon::Triangle(triangle) => Polygon::Triangle(triangle.map(f)),
            Polygon::Quad(quad) => Polygon::Quad(quad.map(f)),
        }
    }
}

impl<T> Polygonal for Polygon<T> {}

impl<T, U> Reduce<U> for Polygon<T> {
    fn reduce<F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
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

impl_zip!(composite => Polygon);

/// Zips the vertices and topologies from multiple iterators into a single
/// iterator.
///
/// This is useful for zipping different attributes of a primitive generator.
/// For example, it can be used to combine position, plane, and UV-mapping data
/// data of a cube into a single topology stream.
///
/// # Examples
///
/// Zip position and UV-mapping data for a cube and map over it to compute
/// color:
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
/// fn map_uv_to_rgba(uv: &Vector<E2>) -> Vector4<R64> {
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
///     .map_vertices(|(position, uv)| (position, uv, map_uv_to_rgba(&uv)))
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
