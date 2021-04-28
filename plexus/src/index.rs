//! Indexing and aggregation.
//!
//! This module provides types and traits that describe _index buffers_ and
//! _indexers_ that disambiguate vertex data to construct minimal _index_ and
//! _vertex buffers_. Plexus refers to independent vertex and index buffers as
//! _raw buffers_. See the [`buffer`] module and [`MeshBuffer`] type for tools
//! for working with these buffers.
//!
//! # Index Buffers
//!
//! Index buffers describe the topology of a polygonal mesh as ordered groups of
//! indices into a vertex buffer. Each group of indices represents a polygon.
//! The vertex buffer contains data that describes each vertex, such as
//! positions or surface normals. Plexus supports _structured_ and _flat_ index
//! buffers via the [`Grouping`] and [`IndexBuffer`] traits. These traits are
//! implemented for [`Vec`].
//!
//! Flat index buffers contain unstructured indices with an implicit grouping,
//! such as `Vec<usize>`. Arity of these buffers is constant and is described by
//! the [`Flat`] meta-grouping. Rendering pipelines typically expect this
//! format.
//!
//! Structured index buffers contain elements that explicitly group indices,
//! such as `Vec<Trigon<usize>>`. These buffers can be formed from polygonal
//! types in the [`primitive`] module.
//!
//! # Indexers
//!
//! [`Indexer`]s construct index and vertex buffers from iterators of polygonal
//! types in the [`primitive`] module, such as [`NGon`] and
//! [`UnboundedPolygon`]. The [`IndexVertices`] trait provides functions for
//! collecting an iterator of $n$-gons into these buffers.
//!
//! Mesh data structures also implement the [`FromIndexer`] and [`FromIterator`]
//! traits so that iterators of $n$-gons can be collected into these types
//! (using a [`HashIndexer`] by default). A specific [`Indexer`] can be
//! configured using the [`CollectWithIndexer`] trait.
//!
//! # Examples
//!
//! Indexing data for a cube to create raw buffers and a [`MeshBuffer`]:
//!
//! ```rust
//! # extern crate decorum;
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use decorum::R64;
//! use nalgebra::Point3;
//! use plexus::buffer::MeshBuffer;
//! use plexus::index::{Flat3, HashIndexer};
//! use plexus::prelude::*;
//! use plexus::primitive::cube::Cube;
//! use plexus::primitive::generate::Position;
//!
//! type E3 = Point3<R64>;
//!
//! let (indices, positions) = Cube::new()
//!     .polygons::<Position<E3>>()
//!     .triangulate()
//!     .index_vertices::<Flat3, _>(HashIndexer::default());
//! let buffer = MeshBuffer::<Flat3, E3>::from_raw_buffers(indices, positions).unwrap();
//! ```
//!
//! [`FromIterator`]: std::iter::FromIterator
//! [`Vec`]: std::vec::Vec
//! [`MeshBuffer`]: crate::buffer::MeshBuffer
//! [`buffer`]: crate::buffer
//! [`MeshGraph`]: crate::graph::MeshGraph
//! [`CollectWithIndexer`]: crate::index::CollectWithIndexer
//! [`Flat`]: crate::index::Flat
//! [`FromIndexer`]: crate::index::FromIndexer
//! [`HashIndexer`]: crate::index::HashIndexer
//! [`Indexer`]: crate::index::Indexer
//! [`IndexVertices`]: crate::index::IndexVertices
//! [`NGon`]: crate::primitive::NGon
//! [`UnboundedPolygon`]: crate::primitive::UnboundedPolygon
//! [`primitive`]: crate::primitive

use num::{Integer, NumCast, Unsigned};
use std::cmp;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use theon::adjunct::Map;
use typenum::NonZero;

use crate::constant::{Constant, ToType, TypeOf};
use crate::primitive::decompose::IntoVertices;
use crate::primitive::Topological;
use crate::{Monomorphic, StaticArity};

pub(in crate) type BufferOf<R> = Vec<<R as Grouping>::Group>;
pub(in crate) type IndexOf<R> = <BufferOf<R> as IndexBuffer<R>>::Index;

// Note that it isn't possible for `IndexBuffer` types to implement
// `DynamicArity`, because they are typically parameterized by `R` (see
// implementations for `Vec<_>`). Instead, `DynamicArity` is implemented for
// `MeshBuffer`, which can bind a `Grouping` and its implementation of
// `StaticArity` with the underlying index buffer type.
/// Index buffer.
///
/// This trait is implemented by types that can be used as an index buffer. The
/// elements in the buffer are determined by a `Grouping`.
///
/// In particular, this trait is implemented by `Vec`, such as `Vec<usize>` or
/// `Vec<Trigon<usize>>`.
pub trait IndexBuffer<R>
where
    R: Grouping,
{
    /// The type of individual indices in the buffer.
    ///
    /// This type is distinct from the grouping. For example, if an index buffer
    /// contains [`Trigon<usize>`][`Trigon`] elements, then this type is `usize`.
    ///
    /// [`Trigon`]: crate::primitive::Trigon
    type Index: Copy + Integer + Unsigned;
}

impl<T, const A: usize> IndexBuffer<Flat<T, A>> for Vec<T>
where
    Constant<A>: ToType,
    TypeOf<A>: NonZero,
    T: Copy + Integer + Unsigned,
{
    type Index = T;
}

impl<P> IndexBuffer<P> for Vec<P>
where
    P: Topological,
    P::Vertex: Copy + Integer + Unsigned,
{
    type Index = P::Vertex;
}

pub trait Push<R, P>: IndexBuffer<R>
where
    R: Grouping,
    P: Topological<Vertex = Self::Index>,
    P::Vertex: Copy + Integer + Unsigned,
{
    fn push(&mut self, index: P);
}

impl<T, P, const A: usize> Push<Flat<T, A>, P> for Vec<T>
where
    Constant<A>: ToType,
    TypeOf<A>: NonZero,
    T: Copy + Integer + Unsigned,
    P: Monomorphic + IntoVertices + Topological<Vertex = T>,
{
    fn push(&mut self, index: P) {
        for index in index.into_vertices() {
            self.push(index);
        }
    }
}

impl<P, Q> Push<P, Q> for Vec<P>
where
    P: From<Q> + Grouping + Topological,
    P::Vertex: Copy + Integer + Unsigned,
    Q: Topological<Vertex = P::Vertex>,
    Self: IndexBuffer<P, Index = P::Vertex>,
{
    fn push(&mut self, index: Q) {
        self.push(P::from(index));
    }
}

pub trait Grouping: StaticArity {
    type Group;
}

/// Flat index buffer meta-grouping.
///
/// Describes a flat index buffer with a constant arity. Arity is specified
/// using a type constant from the [`typenum`] crate.
///
/// Unlike structured groupings, this meta-grouping is needed to associate an
/// index type with an arity. For example, `Vec<usize>` implements both
/// `IndexBuffer<Flat3<usize>>` (a triangular buffer) and
/// `IndexBuffer<Flat4<usize>>` (a quadrilateral buffer).
///
/// See the [`index`] module documention for more information about index
/// buffers.
///
/// # Examples
///
/// Creating a [`MeshBuffer`] with a flat and triangular index buffer:
///
/// ```rust
/// use plexus::buffer::MeshBuffer;
/// use plexus::index::Flat;
/// use plexus::prelude::*;
///
/// let mut buffer = MeshBuffer::<Flat<usize, 3>, (f64, f64, f64)>::default();
/// ```
///
/// [`typenum`]: https://crates.io/crates/typenum
///
/// [`MeshBuffer`]: crate::buffer::MeshBuffer
/// [`index`]: crate::index
#[derive(Debug)]
pub struct Flat<T, const A: usize>
where
    Constant<A>: ToType,
    TypeOf<A>: NonZero,
    T: Copy + Integer + Unsigned,
{
    phantom: PhantomData<T>,
}

impl<T, const A: usize> Grouping for Flat<T, A>
where
    Constant<A>: ToType,
    TypeOf<A>: NonZero,
    T: Copy + Integer + Unsigned,
{
    /// The elements of flat index buffers are indices. These indices are
    /// implicitly grouped by the arity of the buffer (`A`).
    type Group = T;
}

impl<T, const A: usize> Monomorphic for Flat<T, A>
where
    Constant<A>: ToType,
    TypeOf<A>: NonZero,
    T: Copy + Integer + Unsigned,
{
}

impl<T, const A: usize> StaticArity for Flat<T, A>
where
    Constant<A>: ToType,
    TypeOf<A>: NonZero,
    T: Copy + Integer + Unsigned,
{
    type Static = usize;

    const ARITY: Self::Static = A;
}

/// Alias for a flat and triangular index buffer.
pub type Flat3<T = usize> = Flat<T, 3>;
/// Alias for a flat and quadrilateral index buffer.
pub type Flat4<T = usize> = Flat<T, 4>;

/// Structured index buffer grouping.
///
/// Describes a structured index buffer containing [`Topological`] types with
/// index data in their vertices.
///
/// # Examples
///
/// Creating a [`MeshBuffer`] with a structured index buffer:
///
/// ```rust
/// use plexus::buffer::MeshBuffer;
/// use plexus::prelude::*;
/// use plexus::primitive::BoundedPolygon;
///
/// let mut buffer = MeshBuffer::<BoundedPolygon<usize>, (f64, f64, f64)>::default();
/// ```
///
/// [`MeshBuffer`]: crate::buffer::MeshBuffer
/// [`Topological`]: crate::primitive::Topological
impl<P> Grouping for P
where
    P: Topological,
    P::Vertex: Copy + Integer + Unsigned,
{
    /// [`Topological`] index buffers contain $n$-gons that explicitly group
    /// their indices.
    ///
    /// [`Topological`]: crate::primitive::Topological
    type Group = P;
}

/// Vertex indexer.
///
/// Disambiguates arbitrary vertex data and emits a one-to-one mapping of
/// indices to vertices.
pub trait Indexer<T, K>
where
    T: Topological,
{
    /// Indexes a vertex using a keying function.
    ///
    /// Returns a tuple containing the index and optionally vertex data. Vertex
    /// data is only returned if the data has not yet been indexed, otherwise
    /// `None` is returned.
    fn index<F>(&mut self, vertex: T::Vertex, f: F) -> (usize, Option<T::Vertex>)
    where
        F: Fn(&T::Vertex) -> &K;
}

/// Hashing vertex indexer.
///
/// This indexer hashes key data for vertices to form an index. This is fast,
/// reliable, and requires no configuration. Prefer this indexer when possible.
///
/// The vertex key data must implement [`Hash`]. Vertex data often includes
/// floating-point values (i.e., `f32` or `f64`), which do not implement
/// [`Hash`]. Types from the [`decorum`] crate can be used to allow
/// floating-point data to be hashed.
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
/// use plexus::index::{Flat3, HashIndexer};
/// use plexus::prelude::*;
/// use plexus::primitive::cube::Cube;
/// use plexus::primitive::generate::Position;
///
/// let (indices, positions) = Cube::new()
///     .polygons::<Position<Point3<R64>>>()
///     .triangulate()
///     .index_vertices::<Flat3, _>(HashIndexer::default());
/// ```
///
/// [`decorum`]: https://crates.io/crates/decorum
///
/// [`Hash`]: std::hash::Hash
pub struct HashIndexer<T, K>
where
    T: Topological,
    K: Clone + Eq + Hash,
{
    hash: HashMap<K, usize>,
    n: usize,
    phantom: PhantomData<T>,
}

impl<T, K> HashIndexer<T, K>
where
    T: Topological,
    K: Clone + Eq + Hash,
{
    /// Creates a new `HashIndexer`.
    pub fn new() -> Self {
        HashIndexer {
            hash: HashMap::new(),
            n: 0,
            phantom: PhantomData,
        }
    }
}

impl<T, K> Default for HashIndexer<T, K>
where
    T: Topological,
    K: Clone + Eq + Hash,
{
    fn default() -> Self {
        HashIndexer::new()
    }
}

impl<T, K> Indexer<T, K> for HashIndexer<T, K>
where
    T: Topological,
    K: Clone + Eq + Hash,
{
    fn index<F>(&mut self, input: T::Vertex, f: F) -> (usize, Option<T::Vertex>)
    where
        F: Fn(&T::Vertex) -> &K,
    {
        let mut vertex = None;
        let mut n = self.n;
        let index = self.hash.entry(f(&input).clone()).or_insert_with(|| {
            vertex = Some(input);
            let m = n;
            n += 1;
            m
        });
        self.n = n;
        (*index, vertex)
    }
}

/// LRU caching vertex indexer.
///
/// This indexer uses a _least recently used_ (LRU) cache to form an index. To
/// function correctly, an adequate cache capacity is necessary. If the capacity
/// is insufficient, then redundant vertex data may be emitted. See
/// [`LruIndexer::with_capacity`].
///
/// This indexer is useful if the vertex key data does not implement [`Hash`].
/// If the key data can be hashed, prefer `HashIndexer` instead.
///
/// # Examples
///
/// ```rust
/// # extern crate nalgebra;
/// # extern crate plexus;
/// #
/// use nalgebra::Point3;
/// use plexus::index::{Flat3, LruIndexer};
/// use plexus::prelude::*;
/// use plexus::primitive::generate::Position;
/// use plexus::primitive::sphere::UvSphere;
///
/// let (indices, positions) = UvSphere::new(8, 8)
///     .polygons::<Position<Point3<f64>>>()
///     .triangulate()
///     .index_vertices::<Flat3, _>(LruIndexer::with_capacity(64));
/// ```
///
/// [`Hash`]: std::hash::Hash
/// [`LruIndexer::with_capacity`]: crate::index::LruIndexer::with_capacity
pub struct LruIndexer<T, K>
where
    T: Topological,
    K: Clone + PartialEq,
{
    lru: Vec<(K, usize)>,
    capacity: usize,
    n: usize,
    phantom: PhantomData<T>,
}

impl<T, K> LruIndexer<T, K>
where
    T: Topological,
    K: Clone + PartialEq,
{
    /// Creates a new `LruIndexer` with a default capacity.
    pub fn new() -> Self {
        LruIndexer::with_capacity(16)
    }

    /// Creates a new `LruIndexer` with the specified capacity.
    ///
    /// The capacity of the cache must be sufficient in order to generate a
    /// unique set of index and vertex data.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = cmp::max(1, capacity);
        LruIndexer {
            lru: Vec::with_capacity(capacity),
            capacity,
            n: 0,
            phantom: PhantomData,
        }
    }

    fn find(&self, key: &K) -> Option<(usize, usize)> {
        self.lru
            .iter()
            .enumerate()
            .find(|&(_, entry)| entry.0 == *key)
            .map(|(index, entry)| (index, entry.1))
    }
}

impl<T, K> Default for LruIndexer<T, K>
where
    T: Topological,
    K: Clone + PartialEq,
{
    fn default() -> Self {
        LruIndexer::new()
    }
}

impl<T, K> Indexer<T, K> for LruIndexer<T, K>
where
    T: Topological,
    K: Clone + PartialEq,
{
    fn index<F>(&mut self, input: T::Vertex, f: F) -> (usize, Option<T::Vertex>)
    where
        F: Fn(&T::Vertex) -> &K,
    {
        let mut vertex = None;
        let key = f(&input).clone();
        let index = if let Some(entry) = self.find(&key) {
            let vertex = self.lru.remove(entry.0);
            self.lru.push(vertex);
            entry.1
        }
        else {
            vertex = Some(input);
            let m = self.n;
            self.n += 1;
            if self.lru.len() >= self.capacity {
                self.lru.remove(0);
            }
            self.lru.push((key, m));
            m
        };
        (index, vertex)
    }
}

/// Functions for collecting an iterator of $n$-gons into raw index and vertex
/// buffers.
///
/// Unlike [`IndexVertices`], this trait provides functions that are closed (not
/// parameterized) with respect to [`Grouping`]. Instead, the trait is
/// implemented for a particular [`Grouping`]. These functions cannot be used
/// fluently as part of an iterator expression.
///
/// [`Grouping`]: crate::index::Grouping
/// [`IndexVertices`]: crate::index::IndexVertices
pub trait GroupedIndexVertices<R, P>: Sized
where
    R: Grouping,
    P: Topological,
{
    fn index_vertices_with<N, K, F>(self, indexer: N, f: F) -> (Vec<R::Group>, Vec<P::Vertex>)
    where
        N: Indexer<P, K>,
        F: Fn(&P::Vertex) -> &K;

    fn index_vertices<N>(self, indexer: N) -> (Vec<R::Group>, Vec<P::Vertex>)
    where
        N: Indexer<P, P::Vertex>,
    {
        self.index_vertices_with::<N, P::Vertex, _>(indexer, |vertex| vertex)
    }
}

impl<R, P, I> GroupedIndexVertices<R, P> for I
where
    I: Iterator<Item = P>,
    R: Grouping,
    P: Map<IndexOf<R>> + Topological,
    P::Output: Topological<Vertex = IndexOf<R>>,
    BufferOf<R>: Push<R, P::Output>,
    IndexOf<R>: NumCast,
{
    fn index_vertices_with<N, K, F>(self, mut indexer: N, f: F) -> (Vec<R::Group>, Vec<P::Vertex>)
    where
        N: Indexer<P, K>,
        F: Fn(&P::Vertex) -> &K,
    {
        let mut indices = Vec::new();
        let mut vertices = Vec::new();
        for topology in self {
            Push::push(
                &mut indices,
                topology.map(|vertex| {
                    let (index, vertex) = indexer.index(vertex, &f);
                    if let Some(vertex) = vertex {
                        vertices.push(vertex);
                    }
                    NumCast::from(index).unwrap()
                }),
            );
        }
        (indices, vertices)
    }
}

/// Functions for collecting an iterator of $n$-gons into raw index and vertex
/// buffers.
///
/// Unlike [`GroupedIndexVertices`], this trait provides functions that are
/// parameterized with respect to [`Grouping`].
///
/// See [`HashIndexer`] and [`LruIndexer`].
///
/// # Examples
///
///
/// ```rust
/// # extern crate decorum;
/// # extern crate nalgebra;
/// # extern crate plexus;
/// #
/// use decorum::R64;
/// use nalgebra::Point3;
/// use plexus::index::{Flat3, HashIndexer};
/// use plexus::prelude::*;
/// use plexus::primitive::generate::Position;
/// use plexus::primitive::sphere::UvSphere;
///
/// let sphere = UvSphere::new(32, 32);
/// let (indices, positions) = sphere
///     .polygons::<Position<Point3<R64>>>()
///     .triangulate()
///     .index_vertices::<Flat3, _>(HashIndexer::default());
/// ```
///
/// [`GroupedIndexVertices`]: crate::index::GroupedIndexVertices
/// [`Grouping`]: crate::index::Grouping
/// [`HashIndexer`]: crate::index::HashIndexer
/// [`LruIndexer`]: crate::index::LruIndexer
pub trait IndexVertices<P>
where
    P: Topological,
{
    /// Indexes an iterator of $n$-gons into raw index and vertex buffers using
    /// the given grouping, indexer, and keying function.
    fn index_vertices_with<R, N, K, F>(self, indexer: N, f: F) -> (Vec<R::Group>, Vec<P::Vertex>)
    where
        Self: GroupedIndexVertices<R, P>,
        R: Grouping,
        N: Indexer<P, K>,
        F: Fn(&P::Vertex) -> &K,
    {
        GroupedIndexVertices::<R, P>::index_vertices_with(self, indexer, f)
    }

    /// Indexes an iterator of $n$-gons into raw index and vertex buffers using
    /// the given grouping and indexer.
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
    /// use plexus::index::HashIndexer;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    /// use plexus::primitive::Trigon;
    ///
    /// // `indices` contains `Trigon`s with index data.
    /// let (indices, positions) = Cube::new()
    ///     .polygons::<Position<Point3<R64>>>()
    ///     .subdivide()
    ///     .triangulate()
    ///     .index_vertices::<Trigon<usize>, _>(HashIndexer::default());
    /// ```
    fn index_vertices<R, N>(self, indexer: N) -> (Vec<R::Group>, Vec<P::Vertex>)
    where
        Self: GroupedIndexVertices<R, P>,
        R: Grouping,
        N: Indexer<P, P::Vertex>,
    {
        IndexVertices::<P>::index_vertices_with(self, indexer, |vertex| vertex)
    }
}

impl<P, I> IndexVertices<P> for I
where
    I: Iterator<Item = P>,
    P: Topological,
{
}

pub trait FromIndexer<P, Q>: Sized
where
    P: Topological,
    Q: Topological<Vertex = P::Vertex>,
{
    type Error: Debug;

    fn from_indexer<I, N>(input: I, indexer: N) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = P>,
        N: Indexer<Q, P::Vertex>;
}

/// Functions for collecting an iterator of $n$-gons into a mesh data structure.
///
/// These functions can be used to collect data from an iterator into mesh data
/// structures like [`MeshBuffer`] or [`MeshGraph`].
///
/// See [`HashIndexer`] and [`LruIndexer`].
///
/// [`MeshBuffer`]: crate::buffer::MeshBuffer
/// [`MeshGraph`]: crate::graph::MeshGraph
/// [`HashIndexer`]: crate::index::HashIndexer
/// [`LruIndexer`]: crate::index::LruIndexer
pub trait CollectWithIndexer<P, Q>
where
    P: Topological,
    Q: Topological<Vertex = P::Vertex>,
{
    /// Collects an iterator of $n$-gons into a mesh data structure using the
    /// given indexer.
    ///
    /// Unlike `collect`, this function allows the indexer to be specified.
    ///
    /// # Errors
    ///
    /// Returns an error defined by the implementer if the target type cannot be
    /// constructed from the indexed vertex data.
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
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    /// use plexus::index::HashIndexer;
    ///
    /// let graph: MeshGraph<Point3<f64>> = Cube::new()
    ///     .polygons::<Position<Point3<R64>>>()
    ///     .collect_with_indexer(HashIndexer::default())
    ///     .unwrap();
    fn collect_with_indexer<T, N>(self, indexer: N) -> Result<T, T::Error>
    where
        T: FromIndexer<P, Q>,
        N: Indexer<Q, P::Vertex>;
}

impl<P, Q, I> CollectWithIndexer<P, Q> for I
where
    I: Iterator<Item = P>,
    P: Topological,
    Q: Topological<Vertex = P::Vertex>,
{
    fn collect_with_indexer<T, N>(self, indexer: N) -> Result<T, T::Error>
    where
        T: FromIndexer<P, Q>,
        N: Indexer<Q, P::Vertex>,
    {
        T::from_indexer(self, indexer)
    }
}
