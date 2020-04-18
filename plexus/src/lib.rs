//! **Plexus** is a library for polygonal mesh processing.
//!
//! Please note that versions in the `0.0.*` series are experimental and
//! unstable.

// This lint is a bit subjective. Using `next` is equivalent, but the Plexus
// authors find `nth(0)` more clear, especially as part of a non-trivial
// iterator expression. This may be revisited though.
#![allow(clippy::iter_nth_zero)]
#![doc(html_favicon_url = "https://plexus.rs/img/favicon.ico")]
#![doc(html_logo_url = "https://plexus.rs/img/plexus.svg")]

pub mod buffer;
pub mod builder;
pub mod encoding;
pub mod graph;
pub mod index;
pub mod integration;
mod network;
pub mod primitive;
mod transact;

use itertools::{Itertools, MinMaxResult};
use std::borrow::Borrow;
use std::fmt::Debug;

use crate::network::view::ClosedView;

pub use theon::{AsPosition, Position};
pub use typenum::{U2, U3, U4};

pub mod prelude {
    //! Re-exports commonly used types and traits.
    //!
    //! Importing the contents of this module is recommended, especially when
    //! working with generators and iterator expressions, as those operations
    //! are expressed mostly through traits.
    //!
    //! # Traits
    //!
    //! This module re-exports numerous traits. Traits from the [`primitive`]
    //! module for generating and decomposing iterators over topological data
    //! (e.g., [`Trigon`], [`Tetragon`], etc.) are re-exported so that functions
    //! in iterator expressions can be used without lengthy imports.
    //!
    //! Basic traits for (de)constructing [`MeshBuffer`]s and [`MeshGraph`]s are
    //! also re-exported. These traits allow mesh types to be constructed from
    //! raw buffers and buffers to be re-indexed.
    //!
    //! # Types
    //!
    //! The [`Selector`] `enum` and its variants are re-exported for
    //! convenience. [`Selector`] is often used when mutating [`MeshGraph`]s.
    //!
    //! [`MeshBuffer`]: ../buffer/struct.MeshBuffer.html
    //! [`MeshGraph`]: ../graph/struct.MeshGraph.html
    //! [`primitive`]: ../primitive/index.html
    //! [`Selector`]: ../graph/enum.Selector.html
    //! [`Tetragon`]: ../primitive/type.Tetragon.html
    //! [`Trigon`]: ../primitive/type.Trigon.html

    pub use crate::buffer::{
        FromRawBuffers as _, FromRawBuffersWithArity as _, IntoFlatIndex as _,
        IntoStructuredIndex as _,
    };
    pub use crate::builder::{FacetBuilder as _, MeshBuilder as _, SurfaceBuilder as _};
    pub use crate::graph::{ClosedView as _, Edgoid as _, Rebind as _, Ringoid as _, Selector};
    pub use crate::index::{CollectWithIndexer as _, IndexVertices as _};
    pub use crate::primitive::decompose::{
        Edges as _, IntoEdges as _, IntoSubdivisions as _, IntoTetrahedrons as _, IntoTrigons as _,
        IntoVertices as _, Subdivide as _, Tetrahedrons as _, Triangulate as _, Vertices as _,
    };
    pub use crate::primitive::generate::Generator as _;
    pub use crate::primitive::{MapVertices as _, Polygonal as _, Topological as _};
    pub use crate::IteratorExt as _;
    pub use crate::{DynamicArity as _, FromGeometry as _, IntoGeometry as _};

    pub use Selector::ByIndex;
    pub use Selector::ByKey;
}

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
/// [`Arity`]: trait.Arity.html
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
/// [`StaticArity`]: trait.StaticArity.html
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
/// This contrasts _polymorphic_ types like [`Polygon`], which have an interval
/// arity at the type-level and a singular but varying arity for values (because
/// a [`Polygon`] value may be either a trigon or tertragon).
///
/// [`Polygon`]: primitive/enum.Polygon.html
/// [`Trigon`]: primitive/type.Trigon.html
pub trait Monomorphic: StaticArity<Static = usize> {}

/// Arity of a compound structure.
///
/// `MeshArity` represents the arity of a compound structure, which may be
/// _uniform_ or _non-uniform_. This is typically the value-level arity for
/// mesh data structures like [`MeshGraph`] and [`MeshBuffer`].
///
/// [`MeshBuffer`]: buffer/struct.MeshBuffer.html
/// [`MeshGraph`]: graph/struct.MeshGraph.html
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeshArity {
    /// A compound structure has _uniform_ arity if all of its components have
    /// the same arity, such as a `MeshBuffer` composed entirely of trigons.
    Uniform(usize),
    /// A compound structure has _non-uniform_ arity if the arity of its
    /// components differ, such as a `MeshGraph` composed of trigons and
    /// tetragons.
    ///
    /// Non-uniform arity is represented as an inclusive range known as an
    /// _interval_. This is the minimum and maximum arity of the components, in
    /// that order.
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

pub trait FromGeometry<T> {
    fn from_geometry(other: T) -> Self;
}

impl<T> FromGeometry<T> for T {
    fn from_geometry(other: T) -> Self {
        other
    }
}

/// Geometry elision into `()`.
impl<T> FromGeometry<T> for ()
where
    T: UnitGeometry,
{
    fn from_geometry(_: T) -> Self {}
}

/// Geometry elision from `()`.
impl<T> FromGeometry<()> for T
where
    T: UnitGeometry + Default,
{
    fn from_geometry(_: ()) -> Self {
        T::default()
    }
}

/// Geometry elision.
///
/// Geometric types that implement this trait may be elided. In particular,
/// these types may be converted into and from `()` via the [`FromGeometry`] and
/// [`IntoGeometry`] traits.
///
/// For a geometric type `T`, the following table illustrates the elisions in
/// which `T` may participate:
///
/// | Bounds on `T`            | From | Into |
/// |--------------------------|------|------|
/// | `UnitGeometry`           | `T`  | `()` |
/// | `Default + UnitGeometry` | `()` | `T`  |
///
/// These conversions are useful when converting between mesh data structures
/// with incompatible geometry, such as from a [`MeshGraph`] with face geometry
/// to a [`MeshBuffer`] that cannot support such geometry.
///
/// When geometry features are enabled, `UnitGeometry` is implemented for
/// integrated foreign types.
///
/// [`FromGeometry`]: trait.FromGeometry.html
/// [`IntoGeometry`]: trait.IntoGeometry.html
/// [`MeshBuffer`]: buffer/struct.MeshBuffer.html
/// [`MeshGraph`]: graph/struct.MeshGraph.html
pub trait UnitGeometry {}

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
    /// [`MeshGraph`]. This iterator avoids redundant use of `map` to extract
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
    ///
    /// [`MeshGraph`]: graph/struct.MeshGraph.html
    fn keys(self) -> Keys<Self>
    where
        Self::Item: ClosedView,
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
