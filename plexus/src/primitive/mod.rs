//! Primitive topological structures.
//!
//! This module provides composable primitives that describe polygonal
//! structures. This includes simple $n$-gons like triangles, polytope
//! [generators][`generate`], and iterator expressions that compose and
//! decompose iterators of primitives.
//!
//! Types in this module generally describe [cycle graphs][cycle-graph] and are
//! not strictly geometric. For example, [`Polygonal`] types may be
//! geometrically degenerate (e.g., collinear, converged, etc.) or used to
//! approximate polygons from $\Reals^2$ embedded into higher-dimensional
//! spaces. These types are also used for indexing, in which case their
//! representation and data are both entirely topological.
//!
//! Plexus uses the terms _trigon_ and _tetragon_ for its polygon types, which
//! mean _triangle_ and _quadrilateral_, respectively. This is done for
//! consistency with higher arity polygon names (e.g., _decagon_). In some
//! contexts, the term _triangle_ is still used, such as in functions concerning
//! _triangulation_.
//!
//! # Representations
//!
//! Plexus provides various topological types with different capabilities
//! summarized below:
//!
//! | Type               | Morphism    | Arity        | Map | Zip | Tessellate |
//! |--------------------|-------------|--------------|-----|-----|------------|
//! | `NGon`             | Monomorphic | $1,[3,32]$   | Yes | Yes | Yes        |
//! | `BoundedPolygon`   | Polymorphic | $[3,4]$      | Yes | No  | Yes        |
//! | `UnboundedPolygon` | Polymorphic | $[3,\infin)$ | Yes | No  | No         |
//!
//! [`NGon`] is [monomorphic][`Monomorphic`] and supports the broadest set of
//! traits and features. However, its [type-level][`StaticArity`] arity is
//! somewhat limited and its [value-level][`DynamicArity`] arity is fixed. This
//! means, for example, that it is not possible to have an iterator of [`NGon`]s
//! represent both trigons and tetragons, because these polygons must be
//! distinct [`NGon`] types.
//!
//! The polygon types [`BoundedPolygon`] and [`UnboundedPolygon`] are
//! polymorphic and therefore support variable [value-level][`DynamicArity`]
//! [arity][`Arity`]. [`BoundedPolygon`] only expresses a limited set of
//! polygons by enumerating [`NGon`]s, but supports decomposition and other
//! traits. [`UnboundedPolygon`] is most flexible and can represent any
//! arbitrary polygon, but does not support any tessellation features.
//!
//! [`Edge`]s are always represented as `NGon<_, 2>`.
//!
//! # Examples
//!
//! Generating [raw buffers][`buffer`] with positional data of a [cube][`Cube`]
//! using an [`Indexer`]:
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
//! use plexus::primitive::generate::Position;
//!
//! let (indices, positions) = Cube::new()
//!     .polygons::<Position<Point3<N64>>>()
//!     .triangulate()
//!     .index_vertices::<Flat3, _>(HashIndexer::default());
//! ```
//!
//! [cycle-graph]: https://en.wikipedia.org/wiki/cycle_graph
//!
//! [`buffer`]: crate::buffer
//! [`Indexer`]: crate::index::Indexer
//! [`Cube`]: crate::primitive::cube::Cube
//! [`BoundedPolygon`]: crate::primitive::BoundedPolygon
//! [`NGon`]: crate::primitive::NGon
//! [`UnboundedPolygon`]: crate::primitive::UnboundedPolygon
//! [`Arity`]: crate::Arity
//! [`DynamicArity`]: crate::DynamicArity
//! [`StaticArity`]: crate::StaticArity

pub mod cube;
pub mod decompose;
pub mod generate;
pub mod sphere;

use arrayvec::ArrayVec;
use decorum::Real;
use itertools::izip;
use itertools::structs::Zip as OuterZip; // Avoid collision with `Zip`.
use num::{Integer, One, Signed, Unsigned, Zero};
use smallvec::{smallvec, SmallVec};
use std::array;
use std::convert::TryInto;
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use theon::adjunct::{Adjunct, Converged, Extend, Fold, FromItems, IntoItems, Map, ZipMap};
use theon::ops::Cross;
use theon::query::{Intersection, Line, LineLine, LinePlane, Plane, Unit};
use theon::space::{EuclideanSpace, FiniteDimensional, Scalar, Vector, VectorSpace};
use theon::{AsPosition, AsPositionMut, Position};
use typenum::{Cmp, Greater, U1, U2, U3};

use crate::constant::{Constant, ToType, TypeOf};
use crate::geometry::partition::PointPartition;
use crate::primitive::decompose::IntoVertices;
use crate::{DynamicArity, IteratorExt as _, Monomorphic, StaticArity, TryFromIterator};

/// Topological structure.
///
/// Types implementing `Topological` provide some notion of adjacency between
/// vertices of their `Vertex` type. These types typically represent cycle
/// graphs and polygonal structures, but may also include degenerate forms like
/// monogons.
pub trait Topological:
    Adjunct<Item = <Self as Topological>::Vertex>
    + AsMut<[<Self as Topological>::Vertex]>
    + AsRef<[<Self as Topological>::Vertex]>
    + DynamicArity<Dynamic = usize>
    + IntoIterator<Item = <Self as Topological>::Vertex>
    + Sized
{
    type Vertex;

    fn try_from_slice<T>(vertices: T) -> Option<Self>
    where
        Self::Vertex: Copy,
        T: AsRef<[Self::Vertex]>;

    /// Embeds an $n$-gon from $\Reals^2$ into $\Reals^3$.
    ///
    /// The scalar for the additional basis is normalized to the given value.
    ///
    /// # Examples
    ///
    /// Embedding a triangle into the $xy$-plane at $z=1$:
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate theon;
    /// #
    /// use nalgebra::Point2;
    /// use plexus::primitive::{Topological, Trigon};
    /// use theon::space::EuclideanSpace;
    ///
    /// type E2 = Point2<f64>;
    ///
    /// let trigon = Trigon::embed_into_e3_xy(
    ///     Trigon::from([
    ///         E2::from_xy(-1.0, 0.0),
    ///         E2::from_xy(0.0, 1.0),
    ///         E2::from_xy(1.0, 0.0),
    ///     ]),
    ///     1.0,
    /// );
    /// ```
    fn embed_into_e3_xy<P>(ngon: P, z: Scalar<Self::Vertex>) -> Self
    where
        Self::Vertex: EuclideanSpace + FiniteDimensional<N = U3>,
        P: Map<Self::Vertex, Output = Self> + Topological,
        P::Vertex: EuclideanSpace + FiniteDimensional<N = U2> + Extend<Self::Vertex>,
        Vector<P::Vertex>: VectorSpace<Scalar = Scalar<Self::Vertex>>,
    {
        Self::embed_into_e3_xy_with(ngon, z, |position| position)
    }

    fn embed_into_e3_xy_with<P, F>(ngon: P, z: Scalar<Position<Self::Vertex>>, mut f: F) -> Self
    where
        Self::Vertex: AsPosition,
        Position<Self::Vertex>: EuclideanSpace + FiniteDimensional<N = U3>,
        P: Map<Self::Vertex, Output = Self> + Topological,
        P::Vertex: EuclideanSpace + FiniteDimensional<N = U2> + Extend<Position<Self::Vertex>>,
        Vector<P::Vertex>: VectorSpace<Scalar = Scalar<Position<Self::Vertex>>>,
        F: FnMut(Position<Self::Vertex>) -> Self::Vertex,
    {
        ngon.map(move |position| f(position.extend(z)))
    }

    /// Embeds an $n$-gon from $\Reals^2$ into $\Reals^3$.
    ///
    /// The $n$-gon is rotated into the given plane about the origin.
    ///
    /// # Examples
    ///
    /// Embedding a triangle into the $xy$-plane at $z=0$:
    ///
    /// ```rust,no_run
    /// # extern crate nalgebra;
    /// # extern crate theon;
    /// #
    /// use nalgebra::{Point2, Point3};
    /// use plexus::geometry::{Plane, Unit};
    /// use plexus::primitive::{Topological, Trigon};
    /// use theon::space::{Basis, EuclideanSpace};
    ///
    /// type E2 = Point2<f64>;
    /// type E3 = Point3<f64>;
    ///
    /// let trigon = Trigon::embed_into_e3_plane(
    ///     Trigon::from([
    ///         E2::from_xy(-1.0, 0.0),
    ///         E2::from_xy(0.0, 1.0),
    ///         E2::from_xy(1.0, 0.0),
    ///     ]),
    ///     Plane::<E3> {
    ///         origin: EuclideanSpace::origin(),
    ///         normal: Unit::z(),
    ///     },
    /// );
    /// ```
    fn embed_into_e3_plane<P>(ngon: P, plane: Plane<Self::Vertex>) -> Self
    where
        Self::Vertex: EuclideanSpace + FiniteDimensional<N = U3>,
        P: Map<Self::Vertex, Output = Self> + Topological,
        P::Vertex: EuclideanSpace + FiniteDimensional<N = U2> + Extend<Self::Vertex>,
        Vector<P::Vertex>: VectorSpace<Scalar = Scalar<Self::Vertex>>,
    {
        Self::embed_into_e3_plane_with(ngon, plane, |position| position)
    }

    fn embed_into_e3_plane_with<P, F>(ngon: P, _: Plane<Position<Self::Vertex>>, f: F) -> Self
    where
        Self::Vertex: AsPosition,
        Position<Self::Vertex>: EuclideanSpace + FiniteDimensional<N = U3>,
        P: Map<Self::Vertex, Output = Self> + Topological,
        P::Vertex: EuclideanSpace + FiniteDimensional<N = U2> + Extend<Position<Self::Vertex>>,
        Vector<P::Vertex>: VectorSpace<Scalar = Scalar<Position<Self::Vertex>>>,
        F: FnMut(Position<Self::Vertex>) -> Self::Vertex,
    {
        // TODO: Rotate the embedded n-gon into the plane about the origin.
        let _ = Self::embed_into_e3_xy_with(ngon, Zero::zero(), f);
        unimplemented!()
    }

    /// Projects an $n$-gon into a plane.
    ///
    /// The positions in each vertex of the $n$-gon are translated along the
    /// normal of the plane.
    #[must_use]
    fn project_into_plane(mut self, plane: Plane<Position<Self::Vertex>>) -> Self
    where
        Self::Vertex: AsPositionMut,
        Position<Self::Vertex>: EuclideanSpace + FiniteDimensional,
        <Position<Self::Vertex> as FiniteDimensional>::N: Cmp<U2, Output = Greater>,
    {
        for vertex in self.as_mut() {
            let line = Line::<Position<Self::Vertex>> {
                origin: *vertex.as_position(),
                direction: plane.normal,
            };
            // TODO: Assert that this case always occurs; the line lies along
            //       the normal.
            if let Some(LinePlane::TimeOfImpact(distance)) = line.intersection(&plane) {
                let translation = *line.direction.get() * distance;
                vertex.transform(|position| *position + translation);
            }
        }
        self
    }

    // TODO: Once GATs are stabilized, consider using a separate trait for this
    //       to avoid using `Vec` and allocating. This may also require a
    //       different name to avoid collisions with the `decompose` module. See
    //       https://github.com/rust-lang/rust/issues/44265
    fn edges(&self) -> Vec<Edge<&Self::Vertex>> {
        self.as_ref()
            .iter()
            .perimeter()
            .map(|(a, b)| Edge::new(a, b))
            .collect()
    }
}

/// Polygonal structure.
///
/// `Polygonal` types form cycle graphs and extend [`Topological`] types with
/// the additional constraint that all vertices have a degree and valence of
/// two. This requires at least three edges and forbids degenerate structures
/// like monogons.
///
/// These types are topological and do not necessarily represent geometric
/// concepts like polygons in the most strict sense. Polygons are only defined
/// in $\Reals^2$ and cannot have converged or collinear vertices, but
/// `Polygonal` types support this kind of data. However, `Polygonal` types are
/// often used as a geometric approximation of polygons. Moreover, `Polygonal`
/// types often contain non-geometric data, particularly index data.
///
/// [`Topological`]: crate::primitive::Topological
pub trait Polygonal: Topological {
    /// Determines if the polygon is convex.
    ///
    /// This function rejects (returns `false`) degenerate polygons, such as
    /// polygons with collinear or converged vertices.
    fn is_convex(&self) -> bool
    where
        Self::Vertex: AsPosition,
        Position<Self::Vertex>: EuclideanSpace + FiniteDimensional<N = U2>,
    {
        let pi = <Scalar<Position<Self::Vertex>> as Real>::PI;
        let mut sum = <Scalar<Position<Self::Vertex>> as Zero>::zero();
        for (t1, t2) in angles(self).perimeter() {
            if (t1 * t2) <= Zero::zero() {
                return false;
            }
            sum = sum + t1;
        }
        // TODO: Use an approximate comparison and do not explicitly round.
        (sum / (pi + pi)).round().abs() == One::one()
    }
}

pub trait IntoIndexed<N>: Polygonal
where
    N: Copy + Integer + Unsigned,
{
    type Indexed: Polygonal<Vertex = (N, Self::Vertex)>;

    fn into_indexed(self) -> Self::Indexed;
}

impl<N, P, Q> IntoIndexed<N> for P
where
    P: Map<(N, <P as Topological>::Vertex), Output = Q> + Polygonal,
    Q: Polygonal<Vertex = (N, P::Vertex)>,
    N: Copy + Integer + Unsigned,
{
    type Indexed = Q;

    fn into_indexed(self) -> Self::Indexed {
        let mut index = Zero::zero();
        self.map(|vertex| {
            let vertex = (index, vertex);
            index = index + One::one();
            vertex
        })
    }
}

pub trait IntoPolygons: Sized {
    type Output: IntoIterator<Item = Self::Polygon>;
    type Polygon: Polygonal;

    fn into_polygons(self) -> Self::Output;
}

pub trait Rotate {
    #[must_use]
    fn rotate(self, n: isize) -> Self;
}

pub trait Zip {
    type Output: Topological;

    fn zip(self) -> Self::Output;
}

pub trait MapVertices<T, U>: Sized {
    fn map_vertices<F>(self, f: F) -> InteriorMap<Self, U, F>
    where
        F: FnMut(T) -> U;
}

impl<I, T, U> MapVertices<T, U> for I
where
    I: Iterator,
    I::Item: Map<U> + Topological<Vertex = T>,
    <I::Item as Map<U>>::Output: Topological<Vertex = U>,
{
    fn map_vertices<F>(self, f: F) -> InteriorMap<Self, U, F>
    where
        F: FnMut(T) -> U,
    {
        InteriorMap::new(self, f)
    }
}

pub struct InteriorMap<I, T, F> {
    input: I,
    f: F,
    phantom: PhantomData<fn() -> T>,
}

impl<I, T, F> InteriorMap<I, T, F> {
    fn new(input: I, f: F) -> Self {
        InteriorMap {
            input,
            f,
            phantom: PhantomData,
        }
    }
}

impl<I, T, F> Iterator for InteriorMap<I, T, F>
where
    I: Iterator,
    F: FnMut(<I::Item as Topological>::Vertex) -> T,
    I::Item: Map<T> + Topological,
    <I::Item as Map<T>>::Output: Topological<Vertex = T>,
{
    type Item = <I::Item as Map<T>>::Output;

    fn next(&mut self) -> Option<Self::Item> {
        self.input.next().map(|topology| topology.map(&mut self.f))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.input.size_hint()
    }
}

/// Monomorphic $n$-gon.
///
/// `NGon` represents a polygonal structure as an array. Each array element
/// represents vertex data in order with adjacent elements being connected by an
/// implicit undirected edge. For example, an `NGon` with three vertices
/// (`NGon<_, 3>`) would represent a trigon (triangle). Generally these elements
/// are labeled $A$, $B$, $C$, etc. Note that the constant parameter `N`
/// represents the number of the `NGon`'s vertices and **not** the number of its
/// edges (arity).
///
/// `NGon`s with less than three vertices are a degenerate case. An `NGon` with
/// two vertices (`NGon<_, 2>`) is considered a _monogon_ and is used to
/// represent edges. Such an `NGon` is not considered a _digon_, as it
/// represents a single undirected edge rather than two distinct (but collapsed)
/// edges. Note that the polygonal types [`BoundedPolygon`] and
/// [`UnboundedPolygon`] never represent edges. See the [`Edge`] type
/// definition.
///
/// `NGon`s with one or zero vertices are unsupported and lack various trait
/// implementations.
///
/// See the [module][`primitive`] documentation for more information.
///
/// [`BoundedPolygon`]: crate::primitive::BoundedPolygon
/// [`Edge`]: crate::primitive::Edge
/// [`primitive`]: crate::primitive
/// [`UnboundedPolygon`]: crate::primitive::UnboundedPolygon
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NGon<G, const N: usize>(pub [G; N]);

impl<G, const N: usize> NGon<G, N> {
    pub fn into_array(self) -> [G; N] {
        self.0
    }

    pub fn positions(&self) -> NGon<&Position<G>, N>
    where
        G: AsPosition,
    {
        if let Ok(array) = self
            .as_ref()
            .iter()
            .map(|vertex| vertex.as_position())
            .collect::<ArrayVec<_, N>>()
            .into_inner()
        {
            array.into()
        }
        else {
            panic!()
        }
    }
}

impl<'a, G, const N: usize> NGon<&'a G, N> {
    pub fn cloned(self) -> NGon<G, N>
    where
        G: Clone,
    {
        self.map(|vertex| vertex.clone())
    }
}

impl<G, const N: usize> AsRef<[G]> for NGon<G, N> {
    fn as_ref(&self) -> &[G] {
        &self.0
    }
}

impl<G, const N: usize> AsMut<[G]> for NGon<G, N> {
    fn as_mut(&mut self) -> &mut [G] {
        &mut self.0
    }
}

impl<G, const N: usize> Adjunct for NGon<G, N> {
    type Item = G;
}

impl<G, const N: usize> Converged for NGon<G, N>
where
    G: Copy,
{
    fn converged(vertex: G) -> Self {
        NGon([vertex; N])
    }
}

impl<G, const N: usize> DynamicArity for NGon<G, N>
where
    Constant<N>: ToType,
    TypeOf<N>: Cmp<U1, Output = Greater>,
{
    type Dynamic = usize;

    fn arity(&self) -> Self::Dynamic {
        <Self as StaticArity>::ARITY
    }
}

impl<G, const N: usize> Fold for NGon<G, N>
where
    Self: Topological,
{
    fn fold<U, F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        for vertex in self.into_vertices() {
            seed = f(seed, vertex);
        }
        seed
    }
}

impl<G, const N: usize> From<[G; N]> for NGon<G, N> {
    fn from(array: [G; N]) -> Self {
        NGon(array)
    }
}

impl<G, const N: usize> FromItems for NGon<G, N> {
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        items.into_iter().try_collect().ok()
    }
}

impl<G, const N: usize> Index<usize> for NGon<G, N> {
    type Output = G;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.as_ref().index(index)
    }
}

impl<G, const N: usize> IndexMut<usize> for NGon<G, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.as_mut().index_mut(index)
    }
}

impl<G, const N: usize> IntoItems for NGon<G, N> {
    type Output = <Self as IntoIterator>::IntoIter;

    fn into_items(self) -> Self::Output {
        self.into_iter()
    }
}

impl<G, const N: usize> IntoIterator for NGon<G, N> {
    type Item = G;
    type IntoIter = array::IntoIter<G, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_array().into_iter()
    }
}

impl<G, H, const N: usize> Map<H> for NGon<G, N> {
    type Output = NGon<H, N>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> H,
    {
        (self.into_iter().map(f)).try_collect().unwrap()
    }
}

impl<G, const N: usize> Monomorphic for NGon<G, N>
where
    Constant<N>: ToType,
    TypeOf<N>: Cmp<U1, Output = Greater>,
{
}

impl<G, const N: usize> Polygonal for NGon<G, N>
where
    // The compiler cannot deduce that the bounds on `TypeOf<N>` are a strict
    // subset of the similar bounds on the `Topological` implementation, so an
    // explicit bound on `Self` is required.
    Self: Topological,
    Constant<N>: ToType,
    TypeOf<N>: Cmp<U2, Output = Greater>,
{
}

impl<G, const N: usize> StaticArity for NGon<G, N>
where
    Constant<N>: ToType,
    TypeOf<N>: Cmp<U1, Output = Greater>,
{
    type Static = usize;

    const ARITY: Self::Static = crate::n_arity(N);
}

impl<G, const N: usize> Topological for NGon<G, N>
where
    Constant<N>: ToType,
    TypeOf<N>: Cmp<U1, Output = Greater>,
{
    type Vertex = G;

    fn try_from_slice<I>(vertices: I) -> Option<Self>
    where
        Self::Vertex: Copy,
        I: AsRef<[Self::Vertex]>,
    {
        vertices.as_ref().try_into().map(NGon).ok()
    }
}

impl<G, const N: usize> TryFromIterator<G> for NGon<G, N> {
    type Error = ();

    fn try_from_iter<I>(vertices: I) -> Result<Self, Self::Error>
    where
        I: Iterator<Item = G>,
    {
        vertices
            .collect::<ArrayVec<G, N>>()
            .into_inner()
            .map(NGon)
            .map_err(|_| ())
    }
}

impl<G, H, const N: usize> ZipMap<H> for NGon<G, N> {
    type Output = NGon<H, N>;

    fn zip_map<F>(self, other: Self, mut f: F) -> Self::Output
    where
        F: FnMut(Self::Item, Self::Item) -> H,
    {
        (self.into_iter().zip(other).map(|(a, b)| f(a, b)))
            .try_collect()
            .unwrap()
    }
}

macro_rules! impl_zip_ngon {
    (ngons => ($($i:ident),*)) => (
        #[allow(non_snake_case)]
        impl<$($i),*, const N: usize> Zip for ($(NGon<$i, N>),*)
        where
            Constant<N>: ToType,
            TypeOf<N>: Cmp<U1, Output = Greater>,
        {
            type Output = NGon<($($i),*), N>;

            fn zip(self) -> Self::Output {
                let ($($i,)*) = self;
                (izip!($($i.into_iter()),*)).try_collect().unwrap()
            }
        }
    );
}
impl_zip_ngon!(ngons => (A, B));
impl_zip_ngon!(ngons => (A, B, C));
impl_zip_ngon!(ngons => (A, B, C, D));
impl_zip_ngon!(ngons => (A, B, C, D, E));
impl_zip_ngon!(ngons => (A, B, C, D, E, F));

pub type Edge<G> = NGon<G, 2>;

impl<G> Edge<G> {
    pub fn new(a: G, b: G) -> Self {
        NGon([a, b])
    }

    pub fn line(&self) -> Option<Line<Position<G>>>
    where
        G: AsPosition,
    {
        let [origin, endpoint] = self.positions().cloned().into_array();
        Unit::try_from_inner(endpoint - origin).map(|direction| Line { origin, direction })
    }

    pub fn is_bisected(&self, other: &Self) -> bool
    where
        G: AsPosition,
        Position<G>: FiniteDimensional<N = U2>,
    {
        let is_disjoint = |line: Line<Position<G>>, [a, b]: [Position<G>; 2]| {
            line.partition(a)
                .zip(line.partition(b))
                .map(|(pa, pb)| pa != pb)
                .unwrap_or(false)
        };
        self.line()
            .zip(other.line())
            .map(|(left, right)| {
                let left = is_disjoint(left, other.positions().cloned().into_array());
                let right = is_disjoint(right, self.positions().cloned().into_array());
                left && right
            })
            .unwrap_or(false)
    }
}

/// Intersection of edges.
#[derive(Clone, Copy, PartialEq)]
pub enum EdgeEdge<S>
where
    S: EuclideanSpace,
{
    Point(S),
    Edge(Edge<S>),
}

impl<S> EdgeEdge<S>
where
    S: EuclideanSpace,
{
    pub fn into_point(self) -> Option<S> {
        match self {
            EdgeEdge::Point(point) => Some(point),
            _ => None,
        }
    }

    pub fn into_edge(self) -> Option<Edge<S>> {
        match self {
            EdgeEdge::Edge(edge) => Some(edge),
            _ => None,
        }
    }
}

impl<S> Debug for EdgeEdge<S>
where
    S: Debug + EuclideanSpace,
    Vector<S>: Debug,
{
    fn fmt(&self, formatter: &mut Formatter) -> Result<(), fmt::Error> {
        match *self {
            EdgeEdge::Point(point) => write!(formatter, "Point({:?})", point),
            EdgeEdge::Edge(edge) => write!(formatter, "Edge({:?})", edge),
        }
    }
}

impl<T> Intersection<Edge<T>> for Edge<T>
where
    T: AsPosition,
    Position<T>: FiniteDimensional<N = U2>,
{
    type Output = EdgeEdge<Position<T>>;

    // TODO: This first computes a line intersection and then partitions each
    //       edge's endpoints by the other edge's line. That's probably more
    //       expensive than is necessary.
    fn intersection(&self, other: &Edge<T>) -> Option<Self::Output> {
        self.line()
            .zip(other.line())
            .and_then(|(left, right)| match left.intersection(&right) {
                Some(LineLine::Point(point)) => {
                    self.is_bisected(other).then(|| EdgeEdge::Point(point))
                }
                Some(LineLine::Line(_)) => todo!(),
                _ => None,
            })
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

/// Triangle.
pub type Trigon<G> = NGon<G, 3>;

impl<G> Trigon<G> {
    pub fn new(a: G, b: G, c: G) -> Self {
        NGon([a, b, c])
    }

    #[allow(clippy::many_single_char_names)]
    pub fn plane(&self) -> Option<Plane<Position<G>>>
    where
        G: AsPosition,
        Position<G>: EuclideanSpace + FiniteDimensional<N = U3>,
        Vector<Position<G>>: Cross<Output = Vector<Position<G>>>,
    {
        let [a, b, c] = self.positions().cloned().into_array();
        let v = a - b;
        let u = a - c;
        Unit::try_from_inner(v.cross(u))
            .map(move |normal| Plane::<Position<G>> { origin: a, normal })
    }
}

impl<G> Rotate for Trigon<G> {
    fn rotate(self, n: isize) -> Self {
        let n = umod(n, Self::ARITY as isize);
        if n == 1 {
            let [a, b, c] = self.into_array();
            Trigon::new(b, c, a)
        }
        else if n == 2 {
            let [a, b, c] = self.into_array();
            Trigon::new(c, a, b)
        }
        else {
            self
        }
    }
}

/// Quadrilateral.
pub type Tetragon<G> = NGon<G, 4>;

impl<G> Tetragon<G> {
    pub fn new(a: G, b: G, c: G, d: G) -> Self {
        NGon([a, b, c, d])
    }
}

impl<G> Rotate for Tetragon<G> {
    #[allow(clippy::many_single_char_names)]
    fn rotate(self, n: isize) -> Self {
        let n = umod(n, Self::ARITY as isize);
        if n == 1 {
            let [a, b, c, d] = self.into_array();
            Tetragon::new(b, c, d, a)
        }
        else if n == 2 {
            let [a, b, c, d] = self.into_array();
            Tetragon::new(c, d, a, b)
        }
        else if n == 3 {
            let [a, b, c, d] = self.into_array();
            Tetragon::new(d, a, b, c)
        }
        else {
            self
        }
    }
}

/// Bounded polymorphic $n$-gon.
///
/// `BoundedPolygon` represents an $n$-gon with at least three edges by
/// enumerating [`NGon`]s. As such, $n$ is bounded, because the enumeration only
/// supports a limited set of polygons. Only common arities used by generators
/// are provided.
///
/// See the [module][`primitive`] documentation for more information.
///
/// [`primitive`]: crate::primitive
/// [`NGon`]: crate::primitive::NGon
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundedPolygon<G> {
    N3(Trigon<G>),
    N4(Tetragon<G>),
}

impl<G> AsRef<[G]> for BoundedPolygon<G> {
    fn as_ref(&self) -> &[G] {
        match *self {
            BoundedPolygon::N3(ref trigon) => trigon.as_ref(),
            BoundedPolygon::N4(ref tetragon) => tetragon.as_ref(),
        }
    }
}

impl<G> AsMut<[G]> for BoundedPolygon<G> {
    fn as_mut(&mut self) -> &mut [G] {
        match *self {
            BoundedPolygon::N3(ref mut trigon) => trigon.as_mut(),
            BoundedPolygon::N4(ref mut tetragon) => tetragon.as_mut(),
        }
    }
}

impl<G> Adjunct for BoundedPolygon<G> {
    type Item = G;
}

impl<G> DynamicArity for BoundedPolygon<G> {
    type Dynamic = usize;

    fn arity(&self) -> Self::Dynamic {
        match *self {
            BoundedPolygon::N3(..) => Trigon::<G>::ARITY,
            BoundedPolygon::N4(..) => Tetragon::<G>::ARITY,
        }
    }
}

impl<G> Fold for BoundedPolygon<G> {
    fn fold<U, F>(self, mut seed: U, mut f: F) -> U
    where
        F: FnMut(U, Self::Item) -> U,
    {
        for vertex in self.into_vertices() {
            seed = f(seed, vertex);
        }
        seed
    }
}

impl<G> From<[G; 3]> for BoundedPolygon<G> {
    fn from(array: [G; 3]) -> Self {
        BoundedPolygon::N3(array.into())
    }
}

impl<G> From<[G; 4]> for BoundedPolygon<G> {
    fn from(array: [G; 4]) -> Self {
        BoundedPolygon::N4(array.into())
    }
}

impl<G> From<Trigon<G>> for BoundedPolygon<G> {
    fn from(trigon: Trigon<G>) -> Self {
        BoundedPolygon::N3(trigon)
    }
}

impl<G> From<Tetragon<G>> for BoundedPolygon<G> {
    fn from(tetragon: Tetragon<G>) -> Self {
        BoundedPolygon::N4(tetragon)
    }
}

impl<G> Index<usize> for BoundedPolygon<G> {
    type Output = G;

    fn index(&self, index: usize) -> &Self::Output {
        match *self {
            BoundedPolygon::N3(ref trigon) => trigon.index(index),
            BoundedPolygon::N4(ref tetragon) => tetragon.index(index),
        }
    }
}

impl<G> IndexMut<usize> for BoundedPolygon<G> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match *self {
            BoundedPolygon::N3(ref mut trigon) => trigon.index_mut(index),
            BoundedPolygon::N4(ref mut tetragon) => tetragon.index_mut(index),
        }
    }
}

impl<G> IntoItems for BoundedPolygon<G> {
    type Output = SmallVec<[G; 4]>;

    fn into_items(self) -> Self::Output {
        match self {
            BoundedPolygon::N3(trigon) => trigon.into_items().into_iter().collect(),
            BoundedPolygon::N4(tetragon) => tetragon.into_items().into_iter().collect(),
        }
    }
}

impl<G> IntoIterator for BoundedPolygon<G> {
    type Item = G;
    type IntoIter = <<Self as IntoItems>::Output as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.into_items().into_iter()
    }
}

impl<T, U> Map<U> for BoundedPolygon<T> {
    type Output = BoundedPolygon<U>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        match self {
            BoundedPolygon::N3(trigon) => BoundedPolygon::N3(trigon.map(f)),
            BoundedPolygon::N4(tetragon) => BoundedPolygon::N4(tetragon.map(f)),
        }
    }
}

impl<G> Polygonal for BoundedPolygon<G> {}

impl<G> Rotate for BoundedPolygon<G> {
    fn rotate(self, n: isize) -> Self {
        match self {
            BoundedPolygon::N3(trigon) => BoundedPolygon::N3(trigon.rotate(n)),
            BoundedPolygon::N4(tetragon) => BoundedPolygon::N4(tetragon.rotate(n)),
        }
    }
}

impl<G> StaticArity for BoundedPolygon<G> {
    type Static = (usize, usize);

    const ARITY: Self::Static = (3, 4);
}

impl<G> Topological for BoundedPolygon<G> {
    type Vertex = G;

    fn try_from_slice<I>(vertices: I) -> Option<Self>
    where
        Self::Vertex: Copy,
        I: AsRef<[Self::Vertex]>,
    {
        let vertices = vertices.as_ref();
        match vertices.len() {
            3 => Some(BoundedPolygon::N3(NGon(vertices.try_into().unwrap()))),
            4 => Some(BoundedPolygon::N4(NGon(vertices.try_into().unwrap()))),
            _ => None,
        }
    }
}

/// Unbounded polymorphic $n$-gon.
///
/// `UnboundedPolygon` represents an $n$-gon with three or more edges. Unlike
/// [`BoundedPolygon`], there is no limit to the arity of polygons that
/// `UnboundedPolygon` may represent.
///
/// It is not possible to trivially convert an [`UnboundedPolygon`] into other
/// topological types like [`NGon`]. See the [module][`primitive`] documentation
/// for more information.
///
/// [`primitive`]: crate::primitive
/// [`BoundedPolygon`]: crate::primitive::BoundedPolygon
/// [`NGon`]: crate::primitive::NGon
#[derive(Clone, Debug, PartialEq)]
pub struct UnboundedPolygon<G>(SmallVec<[G; 4]>);

impl<G> UnboundedPolygon<G> {
    pub fn trigon(a: G, b: G, c: G) -> Self {
        UnboundedPolygon(smallvec![a, b, c])
    }

    pub fn tetragon(a: G, b: G, c: G, d: G) -> Self {
        UnboundedPolygon(smallvec![a, b, c, d])
    }

    pub fn positions(&self) -> UnboundedPolygon<&Position<G>>
    where
        G: AsPosition,
    {
        UnboundedPolygon(self.0.iter().map(|vertex| vertex.as_position()).collect())
    }
}

impl<'a, G> UnboundedPolygon<&'a G>
where
    G: Clone,
{
    pub fn cloned(self) -> UnboundedPolygon<G> {
        self.map(|vertex| vertex.clone())
    }
}

impl<G> Adjunct for UnboundedPolygon<G> {
    type Item = G;
}

impl<G> AsRef<[G]> for UnboundedPolygon<G> {
    fn as_ref(&self) -> &[G] {
        self.0.as_ref()
    }
}

impl<G> AsMut<[G]> for UnboundedPolygon<G> {
    fn as_mut(&mut self) -> &mut [G] {
        self.0.as_mut()
    }
}

impl<G> DynamicArity for UnboundedPolygon<G> {
    type Dynamic = usize;

    fn arity(&self) -> Self::Dynamic {
        self.0.len()
    }
}

impl<G> From<BoundedPolygon<G>> for UnboundedPolygon<G>
where
    G: Clone,
{
    fn from(polygon: BoundedPolygon<G>) -> Self {
        UnboundedPolygon(SmallVec::from(polygon.as_ref()))
    }
}

impl<G, const N: usize> From<NGon<G, N>> for UnboundedPolygon<G>
where
    Constant<N>: ToType,
    TypeOf<N>: Cmp<U2, Output = Greater>,
    G: Clone,
{
    fn from(ngon: NGon<G, N>) -> Self {
        UnboundedPolygon(SmallVec::from(ngon.as_ref()))
    }
}

impl<G> FromItems for UnboundedPolygon<G> {
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        items.into_iter().try_collect().ok()
    }
}

impl<G> Index<usize> for UnboundedPolygon<G> {
    type Output = G;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<G> IndexMut<usize> for UnboundedPolygon<G> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<G> IntoItems for UnboundedPolygon<G> {
    type Output = SmallVec<[G; 4]>;

    fn into_items(self) -> Self::Output {
        self.0
    }
}

impl<G> IntoIterator for UnboundedPolygon<G> {
    type IntoIter = <SmallVec<[G; 4]> as IntoIterator>::IntoIter;
    type Item = G;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T, U> Map<U> for UnboundedPolygon<T> {
    type Output = UnboundedPolygon<U>;

    fn map<F>(self, f: F) -> Self::Output
    where
        F: FnMut(Self::Item) -> U,
    {
        UnboundedPolygon(self.0.into_iter().map(f).collect())
    }
}

impl<G> Polygonal for UnboundedPolygon<G> {}

impl<G> StaticArity for UnboundedPolygon<G> {
    type Static = (usize, Option<usize>);

    const ARITY: Self::Static = (3, None);
}

impl<G> Topological for UnboundedPolygon<G> {
    type Vertex = G;

    fn try_from_slice<I>(vertices: I) -> Option<Self>
    where
        Self::Vertex: Copy,
        I: AsRef<[Self::Vertex]>,
    {
        let vertices = vertices.as_ref();
        if vertices.len() > 2 {
            Some(UnboundedPolygon(SmallVec::from(vertices)))
        }
        else {
            None
        }
    }
}

impl<G> TryFromIterator<G> for UnboundedPolygon<G> {
    type Error = ();

    fn try_from_iter<I>(vertices: I) -> Result<Self, Self::Error>
    where
        I: Iterator<Item = G>,
    {
        vertices
            .has_at_least(3)
            .map(|items| UnboundedPolygon(items.collect()))
            .ok_or(())
    }
}

/// Zips the vertices of [`Topological`] types from multiple iterators into a
/// single iterator.
///
/// This is useful for zipping different geometric attributes of a
/// [generator][`generate`].  For example, it can be used to combine position,
/// plane, and normal data data of a [`Cube`] into a single topology iterator.
///
/// # Examples
///
/// Zip position, normal, and plane attributes of a [cube][`Cube`]:
///
/// ```rust
/// # extern crate decorum;
/// # extern crate nalgebra;
/// # extern crate plexus;
/// #
/// use decorum::N64;
/// use nalgebra::Point3;
/// use plexus::prelude::*;
/// use plexus::primitive;
/// use plexus::primitive::cube::{Cube, Plane};
/// use plexus::primitive::generate::{Normal, Position};
///
/// type E3 = Point3<N64>;
///
/// let cube = Cube::new();
/// // Zip positions and texture coordinates into each vertex.
/// let polygons = primitive::zip_vertices((
///     cube.polygons::<Position<E3>>(),
///     cube.polygons::<Normal<E3>>(),
///     cube.polygons::<Plane>(),
/// ))
/// .triangulate()
/// .collect::<Vec<_>>();
/// ```
///
/// [`Cube`]: crate::primitive::cube::Cube
/// [`generate`]: crate::primitive::generate
/// [`Topological`]: crate::primitive::Topological
pub fn zip_vertices<T, U>(
    tuple: U,
) -> impl Iterator<Item = <<OuterZip<T> as Iterator>::Item as Zip>::Output>
where
    OuterZip<T>: From<U> + Iterator,
    <OuterZip<T> as Iterator>::Item: Zip,
{
    OuterZip::from(tuple).map(|item| item.zip())
}

/// Gets the relative angles between adjacent edges in a [`Polygonal`] type with
/// positional data in $\Reals^2$.
///
/// Computes the angle between edges at each vertex in order. The angles are
/// wrapped into the interval $(-\pi,\pi]$. The sign of each angle specifies the
/// orientation of adjacent edges. Given a square, exactly four angles of
/// $\plusmn\frac{\pi}{2}$ will be returned, with the sign depending on the
/// winding of the square.
///
/// Adjacent vertices with the same position are ignored, as there is no
/// meaningful edge between such vertices. Because of this, it is possible that
/// the number of angles returned by this function disagrees with the arity of
/// the polygon.
///
/// [`Polygonal`]: crate::primitive::Polygonal
fn angles<P>(polygon: &P) -> impl '_ + Clone + Iterator<Item = Scalar<Position<P::Vertex>>>
where
    P: Polygonal,
    P::Vertex: AsPosition,
    Position<P::Vertex>: EuclideanSpace + FiniteDimensional<N = U2>,
{
    polygon
        .as_ref()
        .iter()
        .map(|vertex| vertex.as_position())
        .perimeter()
        .map(|(a, b)| *b - *a)
        .filter(|vector| !vector.is_zero()) // Reject like positions.
        .map(|vector| vector.into_xy())
        .map(|(x, y)| Real::atan2(x, y)) // Absolute angle.
        .perimeter()
        .map(|(t1, t2)| t2 - t1) // Relative angle (between segments).
        .map(|t| {
            // Wrap the angle into the interval `(-pi, pi]`.
            let pi = <Scalar<Position<P::Vertex>> as Real>::PI;
            if t <= -pi {
                t + (pi + pi)
            }
            else if t > pi {
                t - (pi + pi)
            }
            else {
                t
            }
        })
}

fn umod<T>(n: T, m: T) -> T
where
    T: Copy + Integer,
{
    ((n % m) + m) % m
}

#[cfg(test)]
mod tests {
    use nalgebra::Point2;
    use theon::adjunct::Converged;
    use theon::space::EuclideanSpace;

    use crate::primitive::{NGon, Polygonal, Tetragon, Trigon};

    type E2 = Point2<f64>;

    #[test]
    fn convexity() {
        // Convex triangle.
        let trigon = Trigon::new(
            E2::from_xy(0.0, 1.0),
            E2::from_xy(1.0, 0.0),
            E2::from_xy(-1.0, 0.0),
        );
        assert!(trigon.is_convex());

        // Convex quadrilateral.
        let tetragon = Tetragon::new(
            E2::from_xy(1.0, 1.0),
            E2::from_xy(1.0, -1.0),
            E2::from_xy(-1.0, -1.0),
            E2::from_xy(-1.0, 1.0),
        );
        assert!(tetragon.is_convex());

        // Degenerate (collinear) triangle. Not convex.
        let trigon = Trigon::new(
            E2::from_xy(0.0, 0.0),
            E2::from_xy(0.0, 1.0),
            E2::from_xy(0.0, 2.0),
        );
        assert!(!trigon.is_convex());

        // Degenerate (converged) triangle. Not convex.
        let trigon = Trigon::converged(E2::origin());
        assert!(!trigon.is_convex());

        // Self-intersecting pentagon. Not convex.
        let pentagon = NGon::from([
            E2::from_xy(1.0, 1.0),
            E2::from_xy(1.0, -1.0),
            E2::from_xy(-1.0, -1.0),
            E2::from_xy(-1.0, 1.0),
            E2::from_xy(0.0, -2.0), // Self-intersecting vertex.
        ]);
        assert!(!pentagon.is_convex());
    }
}
