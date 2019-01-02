//! Indexing and aggregation.
//!
//! This module provides _indexers_, which disambiguate geometry and produce
//! the minimal set of topological and geometric data. This can be collected
//! into aggregate data structures like `MeshGraph` and `MeshBuffer`.
//!
//! # Examples
//!
//! ```rust
//! use plexus::buffer::MeshBuffer3;
//! use plexus::prelude::*;
//! use plexus::primitive::cube::Cube;
//! use plexus::primitive::index::HashIndexer;
//!
//! let (indices, positions) = Cube::new()
//!     .polygons_with_position()
//!     .triangulate()
//!     .flat_index_vertices(HashIndexer::default());
//! let buffer = MeshBuffer3::<u32, _>::from_raw_buffers(indices, positions).unwrap();
//! ```

use std::cmp;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;

use crate::primitive::decompose::IntoVertices;
use crate::primitive::topology::{Arity, Map, Topological};

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
/// use plexus::prelude::*;
/// use plexus::primitive::cube::Cube;
/// use plexus::primitive::index::HashIndexer;
///
/// let (indices, positions) = Cube::new()
///     .polygons_with_position()
///     .triangulate()
///     .index_vertices(HashIndexer::default());
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
/// use plexus::prelude::*;
/// use plexus::primitive::index::LruIndexer;
/// use plexus::primitive::sphere::UvSphere;
///
/// let (indices, positions) = UvSphere::new(8, 8)
///     .polygons_with_position()
///     .triangulate()
///     .index_vertices(LruIndexer::with_capacity(64));
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

/// Functions for collecting a topology stream into raw index and vertex
/// buffers.
///
/// Produces structured index buffers with arbitrary arity. The buffers may
/// contain `Triangle`s, `Quad`s, `Polygon`s, etc. For flat buffers with
/// constant arity, see `FlatIndexVertices`.
///
/// See `HashIndexer` and `LruIndexer`.
///
/// # Examples
///
/// Note that using an indexer is not always the most effecient method to
/// create buffers or meshes from a topology stream. Primitives provide
/// per-attribute indeces that may be less expensive to generate than using an
/// indexer. For iterator expressions operating on a single attribute
/// (position, normal, etc.), this can be more effecient.
///
/// ```rust
/// use plexus::prelude::*;
/// use plexus::primitive::index::HashIndexer;
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
///     .index_vertices(HashIndexer::default());
/// ```
pub trait IndexVertices<P>: Sized
where
    P: Map<usize> + Topological,
{
    /// Indexes a topology stream into a structured index buffer and vertex
    /// buffer using the given indexer and keying function.
    fn index_vertices_with<N, K, F>(
        self,
        indexer: N,
        f: F,
    ) -> (Vec<<P as Map<usize>>::Output>, Vec<P::Vertex>)
    where
        N: Indexer<P, K>,
        F: Fn(&P::Vertex) -> &K;

    /// Indexes a topology stream into a structured index buffer and vertex
    /// buffer using the given indexer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::index::HashIndexer;
    ///
    /// // `indices` contains `Triangle`s with index data.
    /// let (indices, positions) = Cube::new()
    ///     .polygons_with_position()
    ///     .subdivide()
    ///     .triangulate()
    ///     .index_vertices(HashIndexer::default());
    /// ```
    fn index_vertices<N>(self, indexer: N) -> (Vec<<P as Map<usize>>::Output>, Vec<P::Vertex>)
    where
        N: Indexer<P, P::Vertex>,
    {
        self.index_vertices_with::<N, P::Vertex, _>(indexer, |vertex| vertex)
    }
}

// TODO: The name `(indices, vertices)` that is commonly used for indexing
//       output is a bit ambiguous. The indices are contained in topological
//       structures which have vertices.
impl<P, I> IndexVertices<P> for I
where
    I: Iterator<Item = P>,
    P: Map<usize> + Topological,
{
    fn index_vertices_with<N, K, F>(
        self,
        mut indexer: N,
        f: F,
    ) -> (Vec<<P as Map<usize>>::Output>, Vec<P::Vertex>)
    where
        N: Indexer<P, K>,
        F: Fn(&P::Vertex) -> &K,
    {
        let mut indices = Vec::new();
        let mut vertices = Vec::new();
        for topology in self {
            indices.push(topology.map(|vertex| {
                let (index, vertex) = indexer.index(vertex, &f);
                if let Some(vertex) = vertex {
                    vertices.push(vertex);
                }
                index
            }));
        }
        (indices, vertices)
    }
}

/// Functions for collecting a topology stream into raw index and vertex
/// buffers.
///
/// Produces flat index buffers, where the polygon arity is constant. This
/// typically requires some kind of tessellation, such as triangulation, to
/// ensure that all polygons have the same arity. For structured buffers with
/// variable arity, see `IndexVertices`.
///
/// See `HashIndexer` and `LruIndexer`.
///
/// # Examples
///
/// Note that using an indexer is not always the most effecient method to
/// create buffers or meshes from a topology stream. Primitives provide
/// per-attribute indeces that may be less expensive to generate than using an
/// indexer. For iterator expressions operating on a single attribute
/// (position, normal, etc.), this can be more effecient.
///
/// ```rust
/// use plexus::prelude::*;
/// use plexus::primitive::index::HashIndexer;
/// use plexus::primitive::sphere::UvSphere;
///
/// let sphere = UvSphere::new(64, 32);
///
/// // Using a positional index is more efficient.
/// let (indices, positions) = (
///     sphere
///         .indices_for_position()
///         .triangulate()
///         .vertices()
///         .collect::<Vec<_>>(),
///     sphere.vertices_with_position().collect::<Vec<_>>(),
/// );
///
/// // Using an indexer is less efficient.
/// let (indices, positions) = sphere
///     .polygons_with_position()
///     .triangulate()
///     .flat_index_vertices(HashIndexer::default());
/// ```
pub trait FlatIndexVertices<P>: Sized
where
    P: Arity + IntoVertices + Topological,
{
    /// Indexes a topology stream into a flat index buffer and vertex buffer
    /// using the given indexer and keying function.
    fn flat_index_vertices_with<N, K, F>(self, indexer: N, f: F) -> (Vec<usize>, Vec<P::Vertex>)
    where
        N: Indexer<P, K>,
        F: Fn(&P::Vertex) -> &K;

    /// Indexes a topology stream into a flat index buffer and vertex buffer
    /// using the given indexer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// use nalgebra::Point3;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::index::HashIndexer;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// # fn main() {
    /// let (indices, positions) = UvSphere::new(16, 16)
    ///     .polygons_with_position()
    ///     .triangulate()
    ///     .flat_index_vertices(HashIndexer::default());
    /// // `indices` is a flat buffer with arity 3.
    /// let mut graph = MeshGraph::<Point3<f64>>::from_raw_buffers_with_arity(indices, positions, 3);
    /// # }
    /// ```
    fn flat_index_vertices<N>(self, indexer: N) -> (Vec<usize>, Vec<P::Vertex>)
    where
        N: Indexer<P, P::Vertex>,
    {
        self.flat_index_vertices_with::<N, P::Vertex, _>(indexer, |vertex| vertex)
    }
}

impl<P, I> FlatIndexVertices<P> for I
where
    I: Iterator<Item = P>,
    P: Arity + IntoVertices + Topological,
{
    fn flat_index_vertices_with<N, K, F>(self, mut indexer: N, f: F) -> (Vec<usize>, Vec<P::Vertex>)
    where
        N: Indexer<P, K>,
        F: Fn(&P::Vertex) -> &K,
    {
        // Do not use `index_vertices`, because flattening index topologies
        // would require allocated an additional `Vec`.
        let mut indices = Vec::new();
        let mut vertices = Vec::new();
        for topology in self {
            for vertex in topology.into_vertices() {
                let (index, vertex) = indexer.index(vertex, &f);
                if let Some(vertex) = vertex {
                    vertices.push(vertex);
                }
                indices.push(index);
            }
        }
        (indices, vertices)
    }
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
    /// use plexus::primitive::index::HashIndexer;
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
