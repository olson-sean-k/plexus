//! **Plexus** is a library for polygonal mesh processing.
//!
//! Please note that versions in the `0.0.*` series are experimental and
//! unstable! Use exact version constraints when specifying a dependency to
//! avoid spurious breakage.

pub mod buffer;
pub mod builder;
pub mod encoding;
pub mod graph;
pub mod index;
mod integration;
pub mod primitive;
mod transact;

use std::fmt::Debug;

use crate::graph::Entry;

pub use theon::{AsPosition, Position};
pub use typenum::{U2, U3, U4};

pub mod prelude {
    //! Re-exports commonly used types and traits.
    //!
    //! Importing the contents of this module is recommended when working with
    //! generators and iterator expressions, as those operations are expressed
    //! mostly through traits.
    //!
    //! # Traits
    //!
    //! This module re-exports numerous traits. Traits from the `primitive`
    //! module for generating and decomposing iterators over topological data
    //! (e.g., `Trigon`, `Tetragon`, etc.) are re-exported so that functions in
    //! iterator expressions can be used without lengthy imports.
    //!
    //! Basic traits for (de)constructing `MeshBuffer`s and `MeshGraph`s are
    //! also re-exported. These traits allow mesh types to be constructed from
    //! raw buffers and buffers to be re-indexed.
    //!
    //! # Types
    //!
    //! The `Selector` enum and its variants are re-exported for convenience.
    //! `Selector` is often used when mutating `MeshGraph`s.

    pub use crate::buffer::{IntoFlatIndex as _, IntoStructuredIndex as _};
    pub use crate::builder::{FacetBuilder as _, MeshBuilder as _, SurfaceBuilder as _};
    pub use crate::graph::Selector;
    pub use crate::index::{CollectWithIndexer as _, IndexVertices as _};
    pub use crate::primitive::decompose::{
        Edges as _, IntoEdges as _, IntoSubdivisions as _, IntoTetrahedrons as _, IntoTrigons as _,
        IntoVertices as _, Subdivide as _, Tetrahedrons as _, Triangulate as _, Vertices as _,
    };
    pub use crate::primitive::generate::Generator as _;
    pub use crate::primitive::{MapVertices as _, Polygonal as _, Topological as _};
    pub use crate::IteratorExt as _;
    pub use crate::{
        FromGeometry as _, FromRawBuffers as _, FromRawBuffersWithArity as _, IntoGeometry as _,
    };

    pub use Selector::ByIndex;
    pub use Selector::ByKey;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Arity {
    Uniform(usize),
    NonUniform(usize, usize),
}

impl Arity {
    pub fn upper_bound(&self) -> usize {
        match *self {
            Arity::Uniform(arity) => arity,
            Arity::NonUniform(_, max) => max,
        }
    }

    pub fn lower_bound(&self) -> usize {
        match *self {
            Arity::Uniform(arity) => arity,
            Arity::NonUniform(min, _) => min,
        }
    }
}

pub trait FromRawBuffers<N, G>: Sized {
    type Error: Debug;

    fn from_raw_buffers<I, J>(indices: I, vertices: J) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = N>,
        J: IntoIterator<Item = G>;
}

pub trait FromRawBuffersWithArity<N, G>: Sized {
    type Error: Debug;

    fn from_raw_buffers_with_arity<I, J>(
        indices: I,
        vertices: J,
        arity: usize,
    ) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = N>,
        J: IntoIterator<Item = G>;
}

pub trait FromGeometry<T> {
    fn from_geometry(other: T) -> Self;
}

impl<T> FromGeometry<T> for T {
    fn from_geometry(other: T) -> Self {
        other
    }
}

impl<T> FromGeometry<()> for T
where
    T: UnitGeometry,
{
    fn from_geometry(_: ()) -> Self {
        T::default()
    }
}

pub trait UnitGeometry: Default {}

pub trait IntoGeometry<T> {
    fn into_geometry(self) -> T;
}

impl<T, U> IntoGeometry<U> for T
where
    U: FromGeometry<T>,
{
    fn into_geometry(self) -> U {
        U::from_geometry(self)
    }
}

/// Extension methods for types implementing `Iterator`.
pub trait IteratorExt: Iterator + Sized {
    /// Provides an iterator over a window of duplets that includes the first
    /// value in the sequence at the beginning and end of the iteration.
    ///
    /// Given a collection of ordered elements $\\{a, b, c\\}$, this iterator
    /// yeilds the ordered items $\\{(a, b), (b, c), (c, a)\\}$.
    fn perimeter(self) -> Perimeter<Self>
    where
        Self::Item: Clone,
    {
        Perimeter::new(self)
    }

    /// Maps an iterator over topological views to the keys of those views.
    ///
    /// It is often useful to examine or collect the keys of views over a
    /// `MeshGraph`. This iterator avoids redundant use of `map` to extract
    /// keys.
    ///
    /// # Examples
    ///
    /// Collecting keys of faces before a topological mutation:
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::R64;
    /// use nalgebra::Point3;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::generate::Position;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// type E3 = Point3<R64>;
    ///
    /// let mut graph = UvSphere::new(6, 6)
    ///     .polygons::<Position<E3>>()
    ///     .collect::<MeshGraph<E3>>();
    ///
    /// let keys = graph
    ///     .faces()
    ///     .filter(|face| face.arity() > 3)
    ///     .keys()
    ///     .collect::<Vec<_>>();
    /// for key in keys {
    ///     graph.face_mut(key).unwrap().poke_with_offset(0.5);
    /// }
    /// ```
    fn keys(self) -> Keys<Self>
    where
        Self::Item: Entry,
    {
        Keys::new(self)
    }
}

impl<I> IteratorExt for I where I: Iterator {}

/// Iterator that produces a window of duplets over its input.
///
/// The duplets produced include the first value in the input sequence at both
/// the beginning and end of the iteration, forming a perimeter. Given a
/// collection of ordered elements $\\{a, b, c\\}$, this iterator yields the
/// ordered items $\\{(a, b), (b, c), (c, a)\\}$.
#[derive(Clone)]
pub struct Perimeter<I>
where
    I: Iterator,
    I::Item: Clone,
{
    input: I,
    first: Option<I::Item>,
    previous: Option<I::Item>,
}

impl<I> Perimeter<I>
where
    I: Iterator,
    I::Item: Clone,
{
    fn new(mut input: I) -> Self {
        let first = input.next();
        let previous = first.clone();
        Perimeter {
            input,
            first,
            previous,
        }
    }
}

impl<I> Iterator for Perimeter<I>
where
    I: Iterator,
    I::Item: Clone,
{
    type Item = (I::Item, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.input.next();
        match (self.previous.clone(), next.or_else(|| self.first.take())) {
            (Some(a), Some(b)) => {
                self.previous = Some(b.clone());
                Some((a, b))
            }
            _ => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.input.size_hint()
    }
}

#[derive(Clone)]
pub struct Keys<I>
where
    I: Iterator,
    I::Item: Entry,
{
    input: I,
}

impl<I> Keys<I>
where
    I: Iterator,
    I::Item: Entry,
{
    fn new(input: I) -> Self {
        Keys { input }
    }
}

impl<I> Iterator for Keys<I>
where
    I: Iterator,
    I::Item: Entry,
{
    type Item = <I::Item as Entry>::Key;

    fn next(&mut self) -> Option<Self::Item> {
        self.input.next().map(|view| view.key())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.input.size_hint()
    }
}
