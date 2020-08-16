//! Decomposition and tessellation.
//!
//! The [`Decompose`] iterator uses various traits to decompose and tessellate
//! iterators of topological structures.
//!
//! This module provides two kinds of decomposition traits: conversions and
//! [`Iterator`] extensions. Conversion traits are implemented by topological
//! types and decompose a single structure into any number of output structures.
//! Extensions are implemented for iterators of topological structures and
//! perform the corresponding conversion on the input items to produce flattened
//! output. For example, [`IntoTrigons`] converts a [`Polygonal`] type into an
//! iterator of [`Trigon`]s while [`Triangulate`] does the same to the items of
//! an iterator.
//!
//! Many of these traits are re-exported in the [`prelude`] module.
//!
//! # Examples
//!
//! Tessellating a [`Tetragon`] into [`Trigon`]s:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use nalgebra::Point2;
//! use plexus::prelude::*;
//! use plexus::primitive::Tetragon;
//!
//! type E2 = Point2<f64>;
//!
//! let square = Tetragon::from([
//!     E2::new(1.0, 1.0),
//!     E2::new(-1.0, 1.0),
//!     E2::new(-1.0, -1.0),
//!     E2::new(1.0, -1.0),
//! ]);
//! let trigons = square.into_trigons();
//! ```
//!
//! Tessellating an iterator of [`Tetragon`]s from a [generator][`generate`]:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use nalgebra::Point3;
//! use plexus::prelude::*;
//! use plexus::primitive::cube::Cube;
//! use plexus::primitive::generate::Position;
//!
//! type E3 = Point3<f64>;
//!
//! let polygons: Vec<_> = Cube::new()
//!     .polygons::<Position<E3>>()
//!     .subdivide()
//!     .triangulate()
//!     .collect();
//! ```
//!
//! [`Iterator`]: std::iter::Iterator
//! [`prelude`]: crate::prelude
//! [`Decompose`]: crate::primitive::decompose::Decompose
//! [`IntoTrigons`]: crate::primitive::decompose::IntoTrigons
//! [`Triangulate`]: crate::primitive::decompose::Triangulate
//! [`generate`]: crate::primitive::generate
//! [`Polygonal`]: crate::primitive::Polygonal
//! [`Tetragon`]: crate::primitive::Tetragon
//! [`Trigon`]: crate::primitive::Trigon

use arrayvec::ArrayVec;
use std::collections::VecDeque;
use std::iter::IntoIterator;
use theon::adjunct::IntoItems;
use theon::ops::Interpolate;

use crate::primitive::{
    BoundedPolygon, Edge, Polygonal, Tetragon, Topological, Trigon, UnboundedPolygon,
};
use crate::IteratorExt as _;

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
    /// use decorum::R64;
    /// use nalgebra::Point3;
    /// use plexus::index::{Flat4, HashIndexer};
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    ///
    /// let (indices, positions) = Cube::new()
    ///     .polygons::<Position<Point3<R64>>>()
    ///     .subdivide()
    ///     .remap(7) // 8 subdivision operations are applied.
    ///     .index_vertices::<Flat4, _>(HashIndexer::default());
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
            if let Some(ngon) = self.output.pop_front() {
                return Some(ngon);
            }
            if let Some(ngon) = self.input.next() {
                self.output.extend((self.f)(ngon));
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

// TODO: Use a macro to implement this for all `NGon`s.
pub trait IntoEdges: Topological {
    type Output: IntoIterator<Item = Edge<Self::Vertex>>;

    fn into_edges(self) -> Self::Output;
}

pub trait IntoTrigons: Polygonal {
    type Output: IntoIterator<Item = Trigon<Self::Vertex>>;

    fn into_trigons(self) -> Self::Output;
}

pub trait IntoSubdivisions: Polygonal {
    type Output: IntoIterator<Item = Self>;

    fn into_subdivisions(self) -> Self::Output;
}

pub trait IntoTetrahedrons: Polygonal {
    fn into_tetrahedrons(self) -> ArrayVec<[Trigon<Self::Vertex>; 4]>;
}

impl<T> IntoEdges for Edge<T> {
    type Output = Option<Edge<Self::Vertex>>;

    fn into_edges(self) -> Self::Output {
        Some(self)
    }
}

impl<T> IntoEdges for Trigon<T>
where
    T: Clone,
{
    type Output = ArrayVec<[Edge<Self::Vertex>; 3]>;

    fn into_edges(self) -> Self::Output {
        let [a, b, c] = self.into_array();
        ArrayVec::from([
            Edge::new(a.clone(), b.clone()),
            Edge::new(b, c.clone()),
            Edge::new(c, a),
        ])
    }
}

impl<T> IntoEdges for Tetragon<T>
where
    T: Clone,
{
    type Output = ArrayVec<[Edge<Self::Vertex>; 4]>;

    fn into_edges(self) -> Self::Output {
        let [a, b, c, d] = self.into_array();
        ArrayVec::from([
            Edge::new(a.clone(), b.clone()),
            Edge::new(b, c.clone()),
            Edge::new(c, d.clone()),
            Edge::new(d, a),
        ])
    }
}

impl<T> IntoEdges for BoundedPolygon<T>
where
    T: Clone,
{
    type Output = Vec<Edge<Self::Vertex>>;

    fn into_edges(self) -> Self::Output {
        match self {
            BoundedPolygon::N3(trigon) => trigon.into_edges().into_iter().collect(),
            BoundedPolygon::N4(tetragon) => tetragon.into_edges().into_iter().collect(),
        }
    }
}

impl<T> IntoEdges for UnboundedPolygon<T>
where
    T: Clone,
{
    type Output = Vec<Edge<Self::Vertex>>;

    fn into_edges(self) -> Self::Output {
        self.into_iter()
            .perimeter()
            .map(|(a, b)| Edge::new(a, b))
            .collect()
    }
}

impl<T> IntoTrigons for Trigon<T> {
    type Output = ArrayVec<[Trigon<Self::Vertex>; 1]>;

    fn into_trigons(self) -> Self::Output {
        ArrayVec::from([self])
    }
}

impl<T> IntoTrigons for Tetragon<T>
where
    T: Clone,
{
    type Output = ArrayVec<[Trigon<Self::Vertex>; 2]>;

    fn into_trigons(self) -> Self::Output {
        let [a, b, c, d] = self.into_array();
        ArrayVec::from([Trigon::new(a.clone(), b, c.clone()), Trigon::new(c, d, a)])
    }
}

impl<T> IntoTrigons for BoundedPolygon<T>
where
    T: Clone,
{
    type Output = Vec<Trigon<Self::Vertex>>;

    fn into_trigons(self) -> Self::Output {
        match self {
            BoundedPolygon::N3(trigon) => trigon.into_trigons().into_iter().collect(),
            BoundedPolygon::N4(tetragon) => tetragon.into_trigons().into_iter().collect(),
        }
    }
}

impl<T> IntoSubdivisions for Trigon<T>
where
    T: Clone + Interpolate<Output = T>,
{
    type Output = ArrayVec<[Trigon<Self::Vertex>; 2]>;

    fn into_subdivisions(self) -> Self::Output {
        let [a, b, c] = self.into_array();
        let ac = a.clone().midpoint(c.clone());
        ArrayVec::from([Trigon::new(b.clone(), ac.clone(), a), Trigon::new(c, ac, b)])
    }
}

impl<T> IntoSubdivisions for Tetragon<T>
where
    T: Clone + Interpolate<Output = T>,
{
    type Output = ArrayVec<[Tetragon<Self::Vertex>; 4]>;

    fn into_subdivisions(self) -> Self::Output {
        let [a, b, c, d] = self.into_array();
        let ab = a.clone().midpoint(b.clone());
        let bc = b.clone().midpoint(c.clone());
        let cd = c.clone().midpoint(d.clone());
        let da = d.clone().midpoint(a.clone());
        let ac = a.clone().midpoint(c.clone()); // Diagonal.
        ArrayVec::from([
            Tetragon::new(a, ab.clone(), ac.clone(), da.clone()),
            Tetragon::new(ab, b, bc.clone(), ac.clone()),
            Tetragon::new(ac.clone(), bc, c, cd.clone()),
            Tetragon::new(da, ac, cd, d),
        ])
    }
}

impl<T> IntoTetrahedrons for Tetragon<T>
where
    T: Clone + Interpolate<Output = T>,
{
    fn into_tetrahedrons(self) -> ArrayVec<[Trigon<Self::Vertex>; 4]> {
        let [a, b, c, d] = self.into_array();
        let ac = a.clone().midpoint(c.clone()); // Diagonal.
        ArrayVec::from([
            Trigon::new(a.clone(), b.clone(), ac.clone()),
            Trigon::new(b, c.clone(), ac.clone()),
            Trigon::new(c, d.clone(), ac.clone()),
            Trigon::new(d, a, ac),
        ])
    }
}

impl<T> IntoSubdivisions for BoundedPolygon<T>
where
    T: Clone + Interpolate<Output = T>,
{
    type Output = Vec<Self>;

    fn into_subdivisions(self) -> Self::Output {
        match self {
            BoundedPolygon::N3(trigon) => trigon
                .into_subdivisions()
                .into_iter()
                .map(|trigon| trigon.into())
                .collect(),
            BoundedPolygon::N4(tetragon) => tetragon
                .into_subdivisions()
                .into_iter()
                .map(|tetragon| tetragon.into())
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
    P: IntoTrigons,
{
    fn triangulate(self) -> Decompose<Self, P, Trigon<P::Vertex>, P::Output>;
}

impl<I, P> Triangulate<P> for I
where
    I: Iterator<Item = P>,
    P: IntoTrigons,
{
    fn triangulate(self) -> Decompose<Self, P, Trigon<P::Vertex>, P::Output> {
        Decompose::new(self, P::into_trigons)
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
    #[allow(clippy::type_complexity)]
    fn tetrahedrons(self) -> Decompose<Self, Tetragon<T>, Trigon<T>, ArrayVec<[Trigon<T>; 4]>>;
}

impl<I, T> Tetrahedrons<T> for I
where
    I: Iterator<Item = Tetragon<T>>,
    T: Clone + Interpolate<Output = T>,
{
    #[allow(clippy::type_complexity)]
    fn tetrahedrons(self) -> Decompose<Self, Tetragon<T>, Trigon<T>, ArrayVec<[Trigon<T>; 4]>> {
        Decompose::new(self, Tetragon::into_tetrahedrons)
    }
}

fn remap<I, P, R, F>(n: usize, ngons: I, f: F) -> Vec<P>
where
    I: IntoIterator<Item = P>,
    R: IntoIterator<Item = P>,
    F: Fn(P) -> R,
{
    let mut ngons: Vec<_> = ngons.into_iter().collect();
    for _ in 0..n {
        ngons = ngons.into_iter().flat_map(&f).collect();
    }
    ngons
}
