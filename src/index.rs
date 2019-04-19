//! Indexing and aggregation.
//!
//! This module provides _indexers_, which disambiguate geometry and produce
//! the minimal set of topological and geometric data. This can be collected
//! into aggregate data structures like `MeshGraph` and `MeshBuffer`.
//!
//! Plexus abstracts over _structured_ (_polygonal_) and _unstructured_
//! (_flat_) index buffers. `Flat` and `Structured`.
//!
//! Flat index buffers directly store individual indices. Because there is no
//! structure, arity must by constant, but arbitrary N-gons are trivially
//! supported. Flat index buffers tend to be more useful for rendering,
//! especially triangular buffers.
//!
//! Structured index buffers contain sub-structures that in turn contain
//! indices. Structured index buffers typically contain `Triangle`s, `Quad`s,
//! or `Polygon`s, all of which preserve the topology of a mesh even if its
//! arity is non-constant.
//!
//! See the `buffer` module and `MeshBuffer` for applications of indexing and
//! index buffers.
//!
//! # Examples
//!
//! ```rust
//! use plexus::buffer::MeshBuffer3;
//! use plexus::index::{Flat3, HashIndexer};
//! use plexus::prelude::*;
//! use plexus::primitive::cube::Cube;
//!
//! let (indices, positions) = Cube::new()
//!     .polygons_with_position()
//!     .triangulate()
//!     .index_vertices::<Flat3, _>(HashIndexer::default());
//! let buffer = MeshBuffer3::<u32, _>::from_raw_buffers(indices, positions).unwrap();
//! ```

use num::{Integer, NumCast, Unsigned};
use std::cmp;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use typenum::NonZero;

use crate::primitive::decompose::IntoVertices;
use crate::primitive::{Arity, Map, Polygon, Quad, Topological, Triangle};

pub use typenum::{U3, U4};

pub trait Grouping {
    type Item;

    /// Arity of the index buffer (and by extension the mesh).
    const ARITY: Option<NonZeroUsize>;
}

/// Index buffer.
pub trait IndexBuffer<R>
where
    R: Grouping,
{
    type Index: Copy + Integer + NumCast + Unsigned;
}

impl<A, N> IndexBuffer<Flat<A, N>> for Vec<N>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
    Flat<A, N>: Grouping<Item = N>,
{
    type Index = N;
}

impl<P> IndexBuffer<Structured<P>> for Vec<P>
where
    P: Topological,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
    Structured<P>: Grouping<Item = P>,
{
    type Index = P::Vertex;
}

pub trait Push<R, P>: IndexBuffer<R>
where
    R: Grouping,
    P: Topological<Vertex = Self::Index>,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
{
    fn push(&mut self, index: P);
}

impl<A, N, P> Push<Flat<A, N>, P> for Vec<N>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
    P: Arity + IntoVertices + Topological<Vertex = N>,
    Flat<A, N>: Grouping<Item = N>,
{
    fn push(&mut self, index: P) {
        for index in index.into_vertices() {
            self.push(index);
        }
    }
}

impl<P> Push<Structured<P>, P> for Vec<P>
where
    P: Topological,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
    Structured<P>: Grouping<Item = P>,
{
    fn push(&mut self, index: P) {
        self.push(index);
    }
}

/// Flat index buffer grouping.
///
/// A flat (unstructured) index buffer with a constant arity. Arity is
/// specified using a type constant from the
/// [typenum](https://crates.io/crates/typenum) crate. `U3` and `U4` are
/// re-exported in the `index` module.
///
/// # Examples
///
/// Creating a flat and triangular `MeshBuffer`:
///
/// ```rust
/// use plexus::buffer::MeshBuffer;
/// use plexus::index::{Flat, U3};
/// use plexus::prelude::*;
///
/// let mut buffer = MeshBuffer::<Flat<U3, usize>, Triplet<f64>>::default();
/// ```
#[derive(Debug)]
pub struct Flat<A = U3, N = usize>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
{
    phantom: PhantomData<(A, N)>,
}

impl<A, N> Grouping for Flat<A, N>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
{
    /// Flat index buffers directly contain indices into vertex data. These
    /// indices are implicitly grouped by the arity of the buffer.
    type Item = N;

    /// Flat index buffers have a constant arity.
    const ARITY: Option<NonZeroUsize> = unsafe { Some(NonZeroUsize::new_unchecked(A::USIZE)) };
}

/// Alias for a flat and triangular index buffer.
pub type Flat3<N = usize> = Flat<U3, N>;
/// Alias for a flat and quadrilateral index buffer.
pub type Flat4<N = usize> = Flat<U4, N>;

/// Structured index buffer grouping.
///
/// A structured index buffer of triangles, quads, etc. Useful if a buffer
/// representing a mesh comprised of both triangles and quads is needed (no
/// need for triangulation).
///
/// # Examples
///
/// Creating a structured `MeshBuffer`:
///
/// ```rust
/// use plexus::buffer::MeshBuffer;
/// use plexus::index::Structured;
/// use plexus::prelude::*;
/// use plexus::primitive::Polygon;
///
/// let mut buffer = MeshBuffer::<Structured<Polygon<usize>>, Triplet<f64>>::default();
/// ```
#[derive(Debug)]
pub struct Structured<P = Polygon<usize>>
where
    P: Topological,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
{
    phantom: PhantomData<P>,
}

/// `Structured` index buffer that contains dynamic `Polygon`s.
impl<N> Grouping for Structured<Polygon<N>>
where
    N: Copy + Integer + NumCast + Unsigned,
{
    /// `Polygon` index buffers contain topological structures that explicitly
    /// group their indices into vertex data.
    type Item = Polygon<N>;

    /// `Polygon` index buffers may or may not have a constant arity. Arity is
    /// encoded independently by each item in the buffer.
    const ARITY: Option<NonZeroUsize> = None;
}

/// `Structured` index buffer that contains `Topological` structures with
/// constant arity.
impl<P> Grouping for Structured<P>
where
    P: Arity + Topological,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
{
    /// `Topological` index buffers contain topological structures that
    /// explicitly group their indices into vertex data.
    type Item = P;

    /// `Topological` index buffers have a constant arity.
    const ARITY: Option<NonZeroUsize> = Some(<P as Arity>::ARITY);
}

/// Alias for a structured and triangular index buffer.
pub type Structured3<N = usize> = Structured<Triangle<N>>;
/// Alias for a structured and quadrilateral index buffer.
pub type Structured4<N = usize> = Structured<Quad<N>>;
/// Alias for a structured and polygonal (variable arity) index buffer.
pub type StructuredN<N = usize> = Structured<Polygon<N>>;

/// Vertex indexer.
///
/// Disambiguates arbitrary vertex data and emits a one-to-one mapping of
/// indices to vertices. This is useful for generating basic rendering buffers
/// for graphics pipelines.
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
/// The vertex key data must be hashable (implement `Hash`). Most vertex data
/// includes floating-point values (i.e., `f32` or `f64`), which do not
/// implement `Hash`. To avoid problems with hashing, primitive generators emit
/// wrapper types (see `R32` and `R64`) that provide hashable floating-point
/// values, so this indexer can typically be used without any additional work.
///
/// See the [decorum](https://crates.io/crates/decorum) crate for more about
/// hashable floating-point values.
///
/// # Examples
///
/// ```rust
/// use plexus::index::{Flat3, HashIndexer};
/// use plexus::prelude::*;
/// use plexus::primitive::cube::Cube;
///
/// let (indices, positions) = Cube::new()
///     .polygons_with_position()
///     .triangulate()
///     .index_vertices::<Flat3, _>(HashIndexer::default());
/// ```
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
/// This indexer uses an LRU (least-recently-used) cache to form an index. To
/// function correctly, an adequate cache capacity is necessary. If the
/// capacity is insufficient, then redundant vertex data may be emitted. See
/// `with_capacity`.
///
/// This indexer is useful if the vertex key data cannot be hashed (does not
/// implement `Hash`). If the key data can be hashed, prefer `HashIndexer`
/// instead.
///
/// # Examples
///
/// ```rust
/// use plexus::index::{Flat3, LruIndexer};
/// use plexus::prelude::*;
/// use plexus::primitive::sphere::UvSphere;
///
/// let (indices, positions) = UvSphere::new(8, 8)
///     .polygons_with_position()
///     .triangulate()
///     .index_vertices::<Flat3, _>(LruIndexer::with_capacity(64));
/// ```
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
    /// unique set of vertex data and indices.
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

// TODO: The name `(indices, vertices)` that is commonly used for indexing
//       output is a bit ambiguous. The indices are contained in topological
//       structures which have vertices.

/// Functions for collecting a topology stream into raw index and vertex
/// buffers.
///
/// See `HashIndexer` and `LruIndexer`.
///
/// # Examples
///
/// Note that using an indexer is not always the most effecient method to
/// create buffers or meshes from a topology stream. Primitives provide
/// per-attribute indeces that may be less expensive to generate than using an
/// indexer. For iterator expressions operating on a single attribute
/// (position, normal, etc.), this can be more efficient.
///
/// ```rust
/// use plexus::index::{Flat3, HashIndexer};
/// use plexus::prelude::*;
/// use plexus::primitive::sphere::UvSphere;
///
/// // Detailed UV-sphere.
/// let sphere = UvSphere::new(64, 32);
///
/// // Using a positional index is more efficient.
/// let (indices, positions) = (
///     sphere
///         .indices_for_position()
///         .triangulate()
///         .collect::<Vec<_>>(),
///     sphere.vertices_with_position().collect::<Vec<_>>(),
/// );
///
/// // Using an indexer is less efficient.
/// let (indices, positions) = sphere
///     .polygons_with_position()
///     .triangulate()
///     .index_vertices::<Flat3, _>(HashIndexer::default());
/// ```
pub trait ClosedIndexVertices<R, P>: Sized
where
    R: Grouping,
    P: Topological,
{
    /// Indexes a topology stream into a structured index buffer and vertex
    /// buffer using the given indexer and keying function.
    fn index_vertices_with<N, K, F>(self, indexer: N, f: F) -> (Vec<R::Item>, Vec<P::Vertex>)
    where
        N: Indexer<P, K>,
        F: Fn(&P::Vertex) -> &K;

    /// Indexes a topology stream into a structured index buffer and vertex
    /// buffer using the given indexer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::index::{HashIndexer, Structured3};
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    ///
    /// // `indices` contains `Triangle`s with index data.
    /// let (indices, positions) = Cube::new()
    ///     .polygons_with_position()
    ///     .subdivide()
    ///     .triangulate()
    ///     .index_vertices::<Structured3, _>(HashIndexer::default());
    /// ```
    fn index_vertices<N>(self, indexer: N) -> (Vec<R::Item>, Vec<P::Vertex>)
    where
        N: Indexer<P, P::Vertex>,
    {
        self.index_vertices_with::<N, P::Vertex, _>(indexer, |vertex| vertex)
    }
}

impl<R, P, I> ClosedIndexVertices<R, P> for I
where
    I: Iterator<Item = P>,
    R: Grouping,
    P: Map<<Vec<R::Item> as IndexBuffer<R>>::Index> + Topological,
    P::Output: Topological,
    Vec<R::Item>: IndexBuffer<R> + Push<R, P::Output>,
{
    fn index_vertices_with<N, K, F>(self, mut indexer: N, f: F) -> (Vec<R::Item>, Vec<P::Vertex>)
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

pub trait OpenIndexVertices<P>
where
    P: Topological,
{
    fn index_vertices_with<R, N, K, F>(self, indexer: N, f: F) -> (Vec<R::Item>, Vec<P::Vertex>)
    where
        Self: ClosedIndexVertices<R, P>,
        R: Grouping,
        N: Indexer<P, K>,
        F: Fn(&P::Vertex) -> &K,
    {
        ClosedIndexVertices::<R, P>::index_vertices_with(self, indexer, f)
    }

    fn index_vertices<R, N>(self, indexer: N) -> (Vec<R::Item>, Vec<P::Vertex>)
    where
        Self: ClosedIndexVertices<R, P>,
        R: Grouping,
        N: Indexer<P, P::Vertex>,
    {
        OpenIndexVertices::<P>::index_vertices_with(self, indexer, |vertex| vertex)
    }
}

impl<P, I> OpenIndexVertices<P> for I
where
    I: Iterator<Item = P>,
    P: Topological,
{
}

pub use OpenIndexVertices as IndexVertices;

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

/// Functions for collecting a topology stream into a mesh, buffer, etc.
///
/// See `HashIndexer` and `LruIndexer`.
pub trait CollectWithIndexer<P, Q>
where
    P: Topological,
    Q: Topological<Vertex = P::Vertex>,
{
    /// Collects a topology stream into a mesh, buffer, etc. using an indexer.
    ///
    /// This allows the default indexer (used by `collect`) to be overridden or
    /// otherwise made explicit in calling code.
    ///
    /// # Errors
    ///
    /// Returns an error defined by the implementer if the target type cannot be
    /// constructed from the indexed vertices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// use nalgebra::Point3;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::index::HashIndexer;
    ///
    /// # fn main() {
    /// let graph = Cube::new()
    ///     .polygons_with_position()
    ///     .collect_with_indexer::<MeshGraph<Point3<f32>>, _>(HashIndexer::default())
    ///     .unwrap();
    /// # }
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
