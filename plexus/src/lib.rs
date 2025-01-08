//! **Plexus** is a highly composable library for polygonal mesh processing.
//!
//! Versions of Plexus in the `0.0.*` series are experimental and unstable.
//! Consider depending on the development branch of the repository. See [the
//! website][website] for the latest information and documentation.
//!
//! [website]: https://plexus.rs

#![doc(html_favicon_url = "https://plexus.rs/img/favicon.ico")]
#![doc(html_logo_url = "https://plexus.rs/img/plexus.svg")]
// LINT: This lint is a bit subjective. Using `next` is equivalent, but the Plexus authors find
//       `nth(0)` more clear, especially as part of a non-trivial iterator expression. This may be
//       revisited though.
#![allow(clippy::iter_nth_zero)]

mod entity;
mod integration;
mod transact;

pub mod buffer;
pub mod builder;
pub mod constant;
pub mod encoding;
pub mod geometry;
pub mod graph;
pub mod index;
pub mod primitive;

use arrayvec::ArrayVec;
use itertools::{self, Itertools, MinMaxResult, MultiPeek};
use std::borrow::Borrow;
use std::fmt::Debug;

use crate::entity::view::ClosedView;

pub mod prelude {
    //! Re-exports of commonly used types and traits.
    //!
    //! Importing the contents of this module is recommended, especially when
    //! working with generators and iterator expressions, as those operations
    //! are expressed mostly through traits.
    //!
    //! # Traits
    //!
    //! Traits from the [`primitive`] module for generating and decomposing
    //! iterators of topological data (e.g., [`Trigon`], [`Tetragon`], etc.) are
    //! re-exported so that functions in iterator expressions can be used more
    //! easily.
    //!
    //! Traits for (de)constructing [`MeshBuffer`]s and [`MeshGraph`]s are
    //! re-exported. These traits allow mesh types to be constructed from raw
    //! buffers and buffers to be re-indexed.
    //!
    //! Extension traits are also re-exported.
    //!
    //! # Types
    //!
    //! The [`Selector`] `enum` and its variants are re-exported for
    //! convenience.
    //!
    //! [`MeshBuffer`]: crate::buffer::MeshBuffer
    //! [`MeshGraph`]: crate::graph::MeshGraph
    //! [`Selector`]: crate::graph::Selector
    //! [`Tetragon`]: crate::primitive::Tetragon
    //! [`Trigon`]: crate::primitive::Trigon
    //! [`primitive`]: crate::primitive

    pub use crate::buffer::{
        FromRawBuffers as _, FromRawBuffersWithArity as _, IntoFlatIndex as _,
        IntoStructuredIndex as _,
    };
    pub use crate::builder::{FacetBuilder as _, MeshBuilder as _, SurfaceBuilder as _};
    pub use crate::geometry::{FromGeometry as _, IntoGeometry as _};
    pub use crate::graph::{ClosedView as _, Rebind as _, Selector};
    pub use crate::index::{CollectWithIndexer as _, IndexVertices as _};
    pub use crate::primitive::decompose::{
        Edges as _, IntoEdges as _, IntoSubdivisions as _, IntoTetrahedrons as _, IntoTrigons as _,
        IntoVertices as _, Subdivide as _, Tetrahedrons as _, Triangulate as _, Vertices as _,
    };
    pub use crate::primitive::generate::Generator as _;
    pub use crate::primitive::{
        IntoPolygons as _, MapVertices as _, Polygonal as _, Topological as _,
    };
    pub use crate::DynamicArity as _;
    pub use crate::IteratorExt as _;

    pub use Selector::ByIndex;
    pub use Selector::ByKey;
}

/// Arity of primitives and polygonal meshes.
///
/// The _arity_ of a primitive topological structure (e.g., an edge, trigon,
/// pentagon, etc.) is the number of edges that comprise the structure. For
/// compound structures like polygonal meshes, arity describes the individual
/// polygons that form the structure and may not be representable as a singular
/// value.
///
/// Arity is most generally described as an _open interval_ with a minimum and
/// optional maximum inclusive range. This trait provides a conversion into this
/// general form for all types that represent arity. See the implementations of
/// this trait for more information.
///
/// Types with arity implement the [`StaticArity`] and [`DynamicArity`] traits,
/// which describe their type-level and value-level arity, respectively.
///
/// [`DynamicArity`]: crate::DynamicArity
/// [`StaticArity`]: crate::StaticArity
pub trait Arity: Copy {
    fn into_interval(self) -> (usize, Option<usize>);
}

/// Singular arity.
impl Arity for usize {
    fn into_interval(self) -> (usize, Option<usize>) {
        (self, Some(self))
    }
}

/// Closed interval arity.
///
/// This type represents a _closed interval_ arity with a minimum and maximum
/// inclusive range.
impl Arity for (usize, usize) {
    fn into_interval(self) -> (usize, Option<usize>) {
        let (min, max) = self;
        (min, Some(max))
    }
}

/// Open interval arity.
///
/// This type represents an _open interval_ arity with a minimum and optional
/// maximum inclusive range. When there is no maximum (`None`), the maximum
/// arity is unspecified. This typically means that there is no theoretical
/// maximum.
impl Arity for (usize, Option<usize>) {
    fn into_interval(self) -> (usize, Option<usize>) {
        self
    }
}

/// Type-level arity.
///
/// This trait specifies the arity that a type supports. Values of a
/// `StaticArity` type have an arity that reflects this constant, which may be
/// any type or form implementing the [`Arity`] trait.
///
/// [`Arity`]: crate::Arity
pub trait StaticArity {
    type Static: Arity;

    const ARITY: Self::Static;
}

/// Value-level arity.
///
/// This trait specifies the arity of a value at runtime. This is often
/// distinct from the type-level arity of the [`StaticArity`] trait, which
/// expresses the capabilities of a type.
///
/// [`StaticArity`]: crate::StaticArity
pub trait DynamicArity: StaticArity {
    type Dynamic: Arity;

    fn arity(&self) -> Self::Dynamic;
}

/// Topological types with fixed and singular arity.
///
/// Types are _monomorphic_ if they have a fixed and singular arity as types and
/// values. For example, [`Trigon`] always and only represents a trigon
/// (triangle) with an arity of three. [`Trigon`] values always have an arity of
/// three and types composed of only [`Trigon`]s have a compound arity of three.
///
/// This contrasts _polymorphic_ types like [`BoundedPolygon`], which have an
/// interval arity at the type-level and a singular but varying arity for values
/// (because a [`BoundedPolygon`] value may be either a trigon or tertragon).
///
/// [`BoundedPolygon`]: crate::primitive::BoundedPolygon
/// [`Trigon`]: crate::primitive::Trigon
pub trait Monomorphic: StaticArity<Static = usize> {}

/// Arity of a compound structure.
///
/// `MeshArity` represents the arity of a compound structure, which may be
/// _uniform_ or _non-uniform_. This is typically the value-level arity for
/// mesh data structures like [`MeshGraph`] and [`MeshBuffer`].
///
/// [`MeshBuffer`]: crate::buffer::MeshBuffer
/// [`MeshGraph`]: crate::graph::MeshGraph
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshArity {
    /// A compound structure has _uniform_ arity if all of its components have
    /// the same arity, such as a [`MeshBuffer`] composed entirely of trigons.
    ///
    /// [`MeshBuffer`]: crate::buffer::MeshBuffer
    Uniform(usize),
    /// A compound structure has _non-uniform_ arity if the arity of its
    /// components differ, such as a [`MeshGraph`] composed of trigons and
    /// tetragons.
    ///
    /// Non-uniform arity is represented as an inclusive range known as an
    /// _interval_. This is the minimum and maximum arity of the components, in
    /// that order.
    ///
    /// [`MeshGraph`]: crate::graph::MeshGraph
    NonUniform(usize, usize),
}

impl MeshArity {
    pub fn from_components<T, I>(components: I) -> Self
    where
        T: DynamicArity<Dynamic = usize>,
        I: IntoIterator,
        I::Item: Borrow<T>,
    {
        match components
            .into_iter()
            .map(|component| component.borrow().arity())
            .minmax()
        {
            MinMaxResult::OneElement(exact) => MeshArity::Uniform(exact),
            MinMaxResult::MinMax(min, max) => MeshArity::NonUniform(min, max),
            _ => MeshArity::Uniform(0),
        }
    }
}

impl Arity for MeshArity {
    fn into_interval(self) -> (usize, Option<usize>) {
        match self {
            MeshArity::Uniform(exact) => (exact, Some(exact)),
            MeshArity::NonUniform(min, max) => (min, Some(max)),
        }
    }
}

pub trait TryFromIterator<T>: Sized {
    type Error;

    fn try_from_iter<I>(items: I) -> Result<Self, Self::Error>
    where
        I: Iterator<Item = T>;
}

impl<T, const N: usize> TryFromIterator<T> for [T; N] {
    type Error = ();

    fn try_from_iter<I>(items: I) -> Result<Self, Self::Error>
    where
        I: Iterator<Item = T>,
    {
        items
            .has_exactly(N)
            .and_then(|items| items.collect::<ArrayVec<T, N>>().into_inner().ok())
            .ok_or(())
    }
}

macro_rules! count {
    ($x:tt $($xs:tt)*) => (1usize + count!($($xs)*));
    () => (0usize);
}
macro_rules! substitute {
    ($_t:tt, $with:ty) => {
        $with
    };
}
macro_rules! impl_try_from_iterator {
    (tuples => ($($i:ident),+)) => (
        impl<T> TryFromIterator<T> for ($(substitute!(($i), T),)+) {
            type Error = ();

            // LINT: The `i` metavariable items are conventionally uppercase and represent type
            //       names. Here, these names are substituted, but uppercase is used for
            //       consistency.
            #[expect(non_snake_case)]
            fn try_from_iter<I>(items: I) -> Result<Self, Self::Error>
            where
                I: Iterator<Item = T>,
            {
                use $crate::IteratorExt as _;

                items
                    .has_exactly(count!($($i)*))
                    .map(|mut items| {
                        $(let $i = items.next().unwrap();)*
                        ($($i,)*)
                    })
                    .ok_or(())
            }
        }
    );
}
impl_try_from_iterator!(tuples => (A, B));
impl_try_from_iterator!(tuples => (A, B, C));
impl_try_from_iterator!(tuples => (A, B, C, D));
impl_try_from_iterator!(tuples => (A, B, C, D, E));
impl_try_from_iterator!(tuples => (A, B, C, D, E, F));

/// Extension methods for types implementing [`Iterator`].
///
/// [`Iterator`]: std::iter::Iterator
pub trait IteratorExt: Iterator + Sized {
    /// Provides an iterator over a window of duplets that includes the first
    /// item in the sequence at both the beginning and end of the iteration.
    ///
    /// Given a collection of ordered items $(a,b,c)$, this iterator yeilds the
    /// ordered items $((a,b),(b,c),(c,a))$.
    fn perimeter(self) -> Perimeter<Self>
    where
        Self::Item: Clone,
    {
        Perimeter::new(self)
    }

    /// Maps an iterator over [`graph`] views to the keys of those views.
    ///
    /// It is often useful to examine or collect the keys of views over a
    /// [`MeshGraph`]. This iterator avoids redundant use of
    /// [`map`][`Iterator::map`] to extract keys.
    ///
    /// # Examples
    ///
    /// Collecting keys of faces before a topological mutation in a
    /// [`MeshGraph`]:
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::Total;
    /// use nalgebra::Point3;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::generate::Position;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// type E3 = Point3<Total<f64>>;
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
    ///
    /// [`Iterator::map`]: std::iter::Iterator::map
    /// [`MeshGraph`]: crate::graph::MeshGraph
    /// [`graph`]: crate::graph
    fn keys(self) -> Keys<Self>
    where
        Self::Item: ClosedView,
    {
        Keys::new(self)
    }

    /// Determines if an iterator provides `n` or more items.
    ///
    /// Returns a peekable iterator if the source iterator provides at least `n`
    /// items, otherwise `None`.
    ///
    /// # Examples
    ///
    /// Ensuring that an iterator over vertices has an arity of at least three:
    ///
    /// ```rust
    /// use plexus::IteratorExt;
    ///
    /// fn is_convex(vertices: impl Iterator<Item = [f64; 2]>) -> bool {
    ///     vertices
    ///         .has_at_least(3)
    ///         .and_then(|vertices| {
    ///             for vertex in vertices {
    ///                 // ...
    ///             }
    ///             // ...
    ///             # Some(0usize)
    ///         })
    ///         .is_some()
    /// }
    /// ```
    fn has_at_least(self, n: usize) -> Option<MultiPeek<Self>> {
        peek_n(self, n).map(|mut peekable| {
            peekable.reset_peek();
            peekable
        })
    }

    fn has_exactly(self, n: usize) -> Option<MultiPeek<Self>> {
        peek_n(self, n).and_then(|mut peekable| {
            peekable.peek().is_none().then(|| {
                peekable.reset_peek();
                peekable
            })
        })
    }

    fn try_collect<T>(self) -> Result<T, T::Error>
    where
        T: TryFromIterator<Self::Item>,
    {
        T::try_from_iter(self)
    }
}

impl<I> IteratorExt for I where I: Iterator {}

/// Iterator that produces a window of duplets over its input.
///
/// The duplets produced include the first item in the input at both the
/// beginning and end of the iteration, forming a perimeter. Given a collection
/// of ordered items $(a,b,c)$, this iterator yields the ordered items
/// $((a,b),(b,c),(c,a))$.
///
/// See [`IteratorExt::perimeter`].
///
/// [`IteratorExt::perimeter`]: crate::IteratorExt::perimeter
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
        self.previous
            .clone()
            .zip(next.or_else(|| self.first.take()))
            .map(|(a, b)| {
                self.previous = Some(b.clone());
                (a, b)
            })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.input.size_hint()
    }
}

/// Iterator that maps [`graph`] views to their keys.
///
/// See [`IteratorExt::keys`].
///
/// [`graph`]: crate::graph
/// [`IteratorExt::keys`]: crate::IteratorExt::keys
#[derive(Clone)]
pub struct Keys<I>
where
    I: Iterator,
    I::Item: ClosedView,
{
    input: I,
}

impl<I> Keys<I>
where
    I: Iterator,
    I::Item: ClosedView,
{
    fn new(input: I) -> Self {
        Keys { input }
    }
}

impl<I> Iterator for Keys<I>
where
    I: Iterator,
    I::Item: ClosedView,
{
    type Item = <I::Item as ClosedView>::Key;

    fn next(&mut self) -> Option<Self::Item> {
        self.input.next().map(|view| view.key())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.input.size_hint()
    }
}

fn peek_n<I>(input: I, n: usize) -> Option<MultiPeek<I>>
where
    I: Iterator,
{
    let mut peekable = itertools::multipeek(input);
    for _ in 0..n {
        peekable.peek()?;
    }
    Some(peekable)
}

/// Computes the arity of a polygon with `n` vertices.
///
/// For `n` greater than two, these values are the same and well-formed. For `n`
/// less than three, the polygon is degenerate (a digon, monogon, or zerogon),
/// all of which are assigned an arity of one. Note that some topological types
/// do not allow `n` being one nor zero.
const fn n_arity(n: usize) -> usize {
    if n < 3 {
        1
    }
    else {
        n
    }
}
