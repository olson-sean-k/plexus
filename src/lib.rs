//! **Plexus** is a library for polygonal mesh processing.
//!
//! Please note that versions in the `0.0.*` series are experimental and
//! unstable! Use exact version constraints when specifying a dependency to
//! avoid spurious breakage.
#![allow(unknown_lints)] // Allow clippy lints.

extern crate arrayvec;
#[cfg(feature = "geometry-cgmath")]
extern crate cgmath;
extern crate decorum;
#[macro_use]
extern crate derivative;
extern crate either;
#[macro_use]
extern crate failure;
extern crate fnv;
extern crate fool;
#[macro_use]
extern crate itertools;
#[cfg(feature = "geometry-mint")]
extern crate mint;
#[cfg(feature = "geometry-nalgebra")]
extern crate nalgebra;
extern crate num;
extern crate smallvec;
extern crate theon;
extern crate typenum;

use std::fmt::Debug;

pub mod buffer;
pub mod geometry;
pub mod graph;
pub mod index;
pub mod primitive;

// TODO: Documentation comments include static image content from the GitHub
//       repository. This is fragile and difficult to maintain. Use a mechanism
//       provided by rustdoc or doxidize for this instead.

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
    //! (e.g., `Triangle`, `Quad`, etc.) are re-exported so that functions in
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
    pub use crate::graph::Selector;
    pub use crate::index::{CollectWithIndexer as _, OpenIndexVertices as _};
    pub use crate::primitive::decompose::{
        Edges as _, IntoEdges as _, IntoSubdivisions as _, IntoTetrahedrons as _,
        IntoTriangles as _, IntoVertices as _, Subdivide as _, Tetrahedrons as _, Triangulate as _,
        Vertices as _,
    };
    pub use crate::primitive::generate::{
        IndicesForNormal as _, IndicesForPosition as _, PolygonsWithNormal as _,
        PolygonsWithPosition as _, PolygonsWithUvMap as _, VerticesWithNormal as _,
        VerticesWithPosition as _,
    };
    pub use crate::primitive::{Converged as _, MapVertices as _, Zip as _};
    pub use crate::IteratorExt as _;
    pub use crate::{FromRawBuffers as _, FromRawBuffersWithArity as _};

    pub use Selector::ByIndex;
    pub use Selector::ByKey;
}

pub enum Arity {
    Uniform(usize),
    NonUniform(usize, usize),
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

/// Extension methods for types implementing `Iterator`.
pub trait IteratorExt: Iterator + Sized {
    /// Provides an iterator over a window of duplets that includes the first
    /// value in the sequence at the beginning and end of the iteration.
    ///
    /// Given a collection of ordered elements $\\{a, b, c\\}$, this iterator
    /// yeilds the ordered items $\\{(a, b), (b, c), (c, a)\\}$.
    fn perimeter(self) -> Perimeter<Self>
    where
        Self::Item: Clone;
}

impl<I> IteratorExt for I
where
    I: Iterator,
{
    fn perimeter(self) -> Perimeter<I>
    where
        I::Item: Clone,
    {
        Perimeter::new(self)
    }
}

/// Iterator that produces a window of duplets over its input.
///
/// The duplets produced include the first value in the input sequence at both
/// the beginning and end of the iteration, forming a perimeter. Given a
/// collection of ordered elements $\\{a, b, c\\}$, this iterator yields the
/// ordered items $\\{(a, b), (b, c), (c, a)\\}$.
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
