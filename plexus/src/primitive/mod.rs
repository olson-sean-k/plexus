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
//! | `NGon`             | Monomorphic | $[2,12]$     | Yes | Yes | Yes        |
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
//! [`Edge`]s are always represented as `NGon<[_; 2]>`.
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

use arrayvec::{Array, ArrayVec};
use decorum::Real;
use fool::BoolExt as _;
use itertools::izip;
use itertools::structs::Zip as OuterZip; // Avoid collision with `Zip`.
use num::{Integer, One, Signed, Unsigned, Zero};
use smallvec::{smallvec, SmallVec};
use std::convert::TryInto;
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use theon::adjunct::{Adjunct, Converged, Fold, FromItems, IntoItems, Map, Push, ZipMap};
use theon::ops::Cross;
use theon::query::{Intersection, Line, LineLine, LinePlane, Plane, Unit};
use theon::space::{EuclideanSpace, FiniteDimensional, Scalar, Vector, VectorSpace};
use theon::{AsPosition, AsPositionMut, Position};
use typenum::type_operators::Cmp;
use typenum::{Greater, U2, U3};

use crate::geometry::partition::PointPartition;
use crate::primitive::decompose::IntoVertices;
use crate::{DynamicArity, IteratorExt as _, Monomorphic, StaticArity};

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
        P::Vertex: EuclideanSpace + FiniteDimensional<N = U2> + Push<Output = Self::Vertex>,
        Vector<P::Vertex>: VectorSpace<Scalar = Scalar<Self::Vertex>>,
    {
        Self::embed_into_e3_xy_with(ngon, z, |position| position)
    }

    fn embed_into_e3_xy_with<P, F>(ngon: P, z: Scalar<Position<Self::Vertex>>, mut f: F) -> Self
    where
        Self::Vertex: AsPosition,
        Position<Self::Vertex>: EuclideanSpace + FiniteDimensional<N = U3>,
        P: Map<Self::Vertex, Output = Self> + Topological,
        P::Vertex:
            EuclideanSpace + FiniteDimensional<N = U2> + Push<Output = Position<Self::Vertex>>,
        Vector<P::Vertex>: VectorSpace<Scalar = Scalar<Position<Self::Vertex>>>,
        F: FnMut(Position<Self::Vertex>) -> Self::Vertex,
    {
        ngon.map(move |position| f(position.push(z)))
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
    ///         E2::from_xy(1.0, 0.0)
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
        P::Vertex: EuclideanSpace + FiniteDimensional<N = U2> + Push<Output = Self::Vertex>,
        Vector<P::Vertex>: VectorSpace<Scalar = Scalar<Self::Vertex>>,
    {
        Self::embed_into_e3_plane_with(ngon, plane, |position| position)
    }

    fn embed_into_e3_plane_with<P, F>(ngon: P, _: Plane<Position<Self::Vertex>>, f: F) -> Self
    where
        Self::Vertex: AsPosition,
        Position<Self::Vertex>: EuclideanSpace + FiniteDimensional<N = U3>,
        P: Map<Self::Vertex, Output = Self> + Topological,
        P::Vertex:
            EuclideanSpace + FiniteDimensional<N = U2> + Push<Output = Position<Self::Vertex>>,
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
    phantom: PhantomData<T>,
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
/// (`NGon<[T; 3]>`) would represent a trigon (triangle). Generally these
/// elements are labeled $A$, $B$, $C$, etc.
///
/// `NGon`s with less than three vertices are a degenerate case. An `NGon` with
/// two vertices (`NGon<[T; 2]>`) is considered a _monogon_ despite common
/// definitions specifying a single vertex. Such an `NGon` is not considered a
/// _digon_, as it represents a single undirected edge rather than two distinct
/// (but collapsed) edges. Single-vertex `NGon`s are unsupported. See the `Edge`
/// type definition.
///
/// See the [module][`primitive`] documentation for more information.
///
/// [`primitive`]: crate::primitive
#[derive(Clone, Copy, Debug, PartialEq)]
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
}

impl<A> AsRef<[<A as Array>::Item]> for NGon<A>
where
    A: Array,
{
    fn as_ref(&self) -> &[A::Item] {
        self.0.as_slice()
    }
}

impl<A> AsMut<[<A as Array>::Item]> for NGon<A>
where
    A: Array,
{
    fn as_mut(&mut self) -> &mut [A::Item] {
        self.0.as_mut_slice()
    }
}

impl<A> Adjunct for NGon<A>
where
    A: Array,
{
    type Item = A::Item;
}

impl<A> Fold for NGon<A>
where
    Self: Topological + IntoItems,
    A: Array,
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
            .map(NGon)
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
        ArrayVec::from(self.into_array())
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

macro_rules! impl_monomorphic_ngon {
    (length => $n:expr) => (
        impl<T> Monomorphic for NGon<[T; $n]> {}

        impl<T> StaticArity for NGon<[T; $n]> {
            type Static = usize;

            const ARITY: Self::Static = $n;
        }

        impl<T> Polygonal for NGon<[T; $n]> {}
    );
    (lengths => $($n:expr),*$(,)?) => (
        impl<T> Monomorphic for NGon<[T; 2]> {}

        impl<T> StaticArity for NGon<[T; 2]> {
            type Static = usize;

            const ARITY: Self::Static = 1;
        }

        $(impl_monomorphic_ngon!(length => $n);)*
    );
}
impl_monomorphic_ngon!(lengths => 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

macro_rules! impl_zip_ngon {
    (length => $n:expr) => (
        impl_zip_ngon!(length => $n, items => (A, B));
        impl_zip_ngon!(length => $n, items => (A, B, C));
        impl_zip_ngon!(length => $n, items => (A, B, C, D));
        impl_zip_ngon!(length => $n, items => (A, B, C, D, E));
        impl_zip_ngon!(length => $n, items => (A, B, C, D, E, F));
    );
    (length => $n:expr, items => ($($i:ident),*)) => (
        #[allow(non_snake_case)]
        impl<$($i),*> Zip for ($(NGon<[$i; $n]>),*) {
            type Output = NGon<[($($i),*); $n]>;

            fn zip(self) -> Self::Output {
                let ($($i,)*) = self;
                FromItems::from_items(izip!($($i.into_items()),*)).unwrap()
            }
        }
    );
}

// TODO: Some inherent functions are not documented to avoid bloat.
macro_rules! impl_ngon {
    (length => $n:expr) => (
        impl<T> NGon<[T; $n]> {
            #[doc(hidden)]
            pub fn positions(&self) -> NGon<[&Position<T>; $n]>
            where
                T: AsPosition,
            {
                if let Ok(array) = self
                    .as_ref()
                    .iter()
                    .map(|vertex| vertex.as_position())
                    .collect::<ArrayVec<[_; $n]>>()
                    .into_inner()
                {
                    array.into()
                }
                else {
                    panic!()
                }
            }
        }

        impl<'a, T> NGon<[&'a T; $n]> {
            #[doc(hidden)]
            pub fn cloned(self) -> NGon<[T; $n]>
            where
                T: Clone,
            {
                self.map(|vertex| vertex.clone())
            }
        }

        impl<T> Converged for NGon<[T; $n]>
        where
            T: Copy,
        {
            fn converged(item: T) -> Self {
                NGon([item; $n])
            }
        }

        impl<T> DynamicArity for NGon<[T; $n]> {
            type Dynamic = usize;

            fn arity(&self) -> Self::Dynamic {
                <Self as StaticArity>::ARITY
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

        impl<T> Topological for NGon<[T; $n]> {
            type Vertex = T;

            fn try_from_slice<U>(vertices: U) -> Option<Self>
            where
                Self::Vertex: Copy,
                U: AsRef<[Self::Vertex]>,
            {
                vertices.as_ref().try_into().map(NGon).ok()
            }
        }

        impl_zip_ngon!(length => $n);

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

pub type Edge<T> = NGon<[T; 2]>;

impl<T> Edge<T> {
    pub fn new(a: T, b: T) -> Self {
        NGon([a, b])
    }

    pub fn line(&self) -> Option<Line<Position<T>>>
    where
        T: AsPosition,
    {
        let [origin, endpoint] = self.positions().cloned().into_array();
        Unit::try_from_inner(endpoint - origin).map(|direction| Line { origin, direction })
    }

    pub fn is_bisected(&self, other: &Self) -> bool
    where
        T: AsPosition,
        Position<T>: FiniteDimensional<N = U2>,
    {
        let is_disjoint = |line: Line<Position<T>>, [a, b]: [Position<T>; 2]| {
            fool::zip((line.partition(a), line.partition(b)))
                .map(|(pa, pb)| pa != pb)
                .unwrap_or(false)
        };
        fool::zip((self.line(), other.line()))
            .map(|(l1, l2)| {
                let l1 = is_disjoint(l1, other.positions().cloned().into_array());
                let l2 = is_disjoint(l2, self.positions().cloned().into_array());
                l1 && l2
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
    #[allow(unstable_name_collisions)]
    fn intersection(&self, other: &Edge<T>) -> Option<Self::Output> {
        fool::zip((self.line(), other.line())).and_then(|(l1, l2)| match l1.intersection(&l2) {
            Some(LineLine::Point(point)) => {
                self.is_bisected(other).then_some(EdgeEdge::Point(point))
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
pub type Trigon<T> = NGon<[T; 3]>;

impl<T> Trigon<T> {
    pub fn new(a: T, b: T, c: T) -> Self {
        NGon([a, b, c])
    }

    #[allow(clippy::many_single_char_names)]
    pub fn plane(&self) -> Option<Plane<Position<T>>>
    where
        T: AsPosition,
        Position<T>: EuclideanSpace + FiniteDimensional<N = U3>,
        Vector<Position<T>>: Cross<Output = Vector<Position<T>>>,
    {
        let [a, b, c] = self.positions().cloned().into_array();
        let v = a - b;
        let u = a - c;
        Unit::try_from_inner(v.cross(u))
            .map(move |normal| Plane::<Position<T>> { origin: a, normal })
    }
}

impl<T> Rotate for Trigon<T> {
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
pub type Tetragon<T> = NGon<[T; 4]>;

impl<T> Tetragon<T> {
    pub fn new(a: T, b: T, c: T, d: T) -> Self {
        NGon([a, b, c, d])
    }
}

impl<T> Rotate for Tetragon<T> {
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
pub enum BoundedPolygon<T> {
    N3(Trigon<T>),
    N4(Tetragon<T>),
}

impl<T> AsRef<[T]> for BoundedPolygon<T> {
    fn as_ref(&self) -> &[T] {
        match *self {
            BoundedPolygon::N3(ref trigon) => trigon.as_ref(),
            BoundedPolygon::N4(ref tetragon) => tetragon.as_ref(),
        }
    }
}

impl<T> AsMut<[T]> for BoundedPolygon<T> {
    fn as_mut(&mut self) -> &mut [T] {
        match *self {
            BoundedPolygon::N3(ref mut trigon) => trigon.as_mut(),
            BoundedPolygon::N4(ref mut tetragon) => tetragon.as_mut(),
        }
    }
}

impl<T> Adjunct for BoundedPolygon<T> {
    type Item = T;
}

impl<T> DynamicArity for BoundedPolygon<T> {
    type Dynamic = usize;

    fn arity(&self) -> Self::Dynamic {
        match *self {
            BoundedPolygon::N3(..) => Trigon::<T>::ARITY,
            BoundedPolygon::N4(..) => Tetragon::<T>::ARITY,
        }
    }
}

impl<T> Fold for BoundedPolygon<T> {
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

impl<T> From<[T; 3]> for BoundedPolygon<T> {
    fn from(array: [T; 3]) -> Self {
        BoundedPolygon::N3(array.into())
    }
}

impl<T> From<[T; 4]> for BoundedPolygon<T> {
    fn from(array: [T; 4]) -> Self {
        BoundedPolygon::N4(array.into())
    }
}

impl<T> From<Trigon<T>> for BoundedPolygon<T> {
    fn from(trigon: Trigon<T>) -> Self {
        BoundedPolygon::N3(trigon)
    }
}

impl<T> From<Tetragon<T>> for BoundedPolygon<T> {
    fn from(tetragon: Tetragon<T>) -> Self {
        BoundedPolygon::N4(tetragon)
    }
}

impl<T> Index<usize> for BoundedPolygon<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match *self {
            BoundedPolygon::N3(ref trigon) => trigon.index(index),
            BoundedPolygon::N4(ref tetragon) => tetragon.index(index),
        }
    }
}

impl<T> IndexMut<usize> for BoundedPolygon<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match *self {
            BoundedPolygon::N3(ref mut trigon) => trigon.index_mut(index),
            BoundedPolygon::N4(ref mut tetragon) => tetragon.index_mut(index),
        }
    }
}

impl<T> IntoItems for BoundedPolygon<T> {
    type Output = SmallVec<[T; 4]>;

    fn into_items(self) -> Self::Output {
        match self {
            BoundedPolygon::N3(trigon) => trigon.into_items().into_iter().collect(),
            BoundedPolygon::N4(tetragon) => tetragon.into_items().into_iter().collect(),
        }
    }
}

impl<T> IntoIterator for BoundedPolygon<T> {
    type Item = T;
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

impl<T> Polygonal for BoundedPolygon<T> {}

impl<T> Rotate for BoundedPolygon<T> {
    fn rotate(self, n: isize) -> Self {
        match self {
            BoundedPolygon::N3(trigon) => BoundedPolygon::N3(trigon.rotate(n)),
            BoundedPolygon::N4(tetragon) => BoundedPolygon::N4(tetragon.rotate(n)),
        }
    }
}

impl<T> StaticArity for BoundedPolygon<T> {
    type Static = (usize, usize);

    const ARITY: Self::Static = (3, 4);
}

impl<T> Topological for BoundedPolygon<T> {
    type Vertex = T;

    fn try_from_slice<U>(vertices: U) -> Option<Self>
    where
        Self::Vertex: Copy,
        U: AsRef<[Self::Vertex]>,
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
pub struct UnboundedPolygon<T>(SmallVec<[T; 4]>);

impl<T> UnboundedPolygon<T> {
    pub fn trigon(a: T, b: T, c: T) -> Self {
        UnboundedPolygon(smallvec![a, b, c])
    }

    pub fn tetragon(a: T, b: T, c: T, d: T) -> Self {
        UnboundedPolygon(smallvec![a, b, c, d])
    }

    pub fn positions(&self) -> UnboundedPolygon<&Position<T>>
    where
        T: AsPosition,
    {
        UnboundedPolygon(self.0.iter().map(|vertex| vertex.as_position()).collect())
    }
}

impl<'a, T> UnboundedPolygon<&'a T>
where
    T: Clone,
{
    pub fn cloned(self) -> UnboundedPolygon<T> {
        self.map(|vertex| vertex.clone())
    }
}

impl<T> Adjunct for UnboundedPolygon<T> {
    type Item = T;
}

impl<T> AsRef<[T]> for UnboundedPolygon<T> {
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<T> AsMut<[T]> for UnboundedPolygon<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<T> DynamicArity for UnboundedPolygon<T> {
    type Dynamic = usize;

    fn arity(&self) -> Self::Dynamic {
        self.0.len()
    }
}

impl<T> From<BoundedPolygon<T>> for UnboundedPolygon<T>
where
    T: Clone,
{
    fn from(polygon: BoundedPolygon<T>) -> Self {
        UnboundedPolygon(SmallVec::from(polygon.as_ref()))
    }
}

impl<T> FromItems for UnboundedPolygon<T> {
    fn from_items<I>(items: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Item>,
    {
        items
            .into_iter()
            .has_at_least(3)
            .map(|items| UnboundedPolygon(items.collect()))
    }
}

impl<T> Index<usize> for UnboundedPolygon<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<T> IndexMut<usize> for UnboundedPolygon<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<T> IntoItems for UnboundedPolygon<T> {
    type Output = SmallVec<[T; 4]>;

    fn into_items(self) -> Self::Output {
        self.0
    }
}

impl<T> IntoIterator for UnboundedPolygon<T> {
    type IntoIter = <SmallVec<[T; 4]> as IntoIterator>::IntoIter;
    type Item = T;

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

impl<T> Polygonal for UnboundedPolygon<T> {}

impl<T> StaticArity for UnboundedPolygon<T> {
    type Static = (usize, Option<usize>);

    const ARITY: Self::Static = (3, None);
}

impl<T> Topological for UnboundedPolygon<T> {
    type Vertex = T;

    fn try_from_slice<U>(vertices: U) -> Option<Self>
    where
        Self::Vertex: Copy,
        U: AsRef<[Self::Vertex]>,
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

macro_rules! impl_unbounded_polygon {
    (length => $n:expr) => (
        impl<T> From<NGon<[T; $n]>> for UnboundedPolygon<T>
        where
            T: Clone,
        {
            fn from(ngon: NGon<[T; $n]>) -> Self {
                UnboundedPolygon(SmallVec::from(ngon.as_ref()))
            }
        }
    );
    (lengths => $($n:expr),*$(,)?) => (
        $(impl_unbounded_polygon!(length => $n);)*
    );
}
impl_unbounded_polygon!(lengths => 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

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
fn angles<'a, P>(polygon: &'a P) -> impl 'a + Clone + Iterator<Item = Scalar<Position<P::Vertex>>>
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
