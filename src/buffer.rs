//! Linear representation of meshes.
//!
//! This module provides a `MeshBuffer` that can be read by graphics pipelines
//! to render meshes. `MeshBuffer` combines an index buffer and vertex buffer,
//! which are exposed as slices.
//!
//! # Vertex Buffers
//!
//! Vertex buffers describe the geometry of a `MeshBuffer`.
//!
//! `MeshBuffer`s use _composite_ vertex buffers (as opposed to _component_
//! buffers). Each index in the buffer refers to a datum in the vertex buffer
//! that completely describes that vertex. For example, if each vertex is
//! described by a position and color attribute, then each element in the
//! vertex buffer contains both attributes within a single structure.
//!
//! # Index Buffers
//!
//! Index buffers describe the topology of a `MeshBuffer`. Both _structured_
//! (_polygonal_) and _unstructured_ (_flat_) index buffers are supported. See
//! `Flat` and `Structured`.
//!
//! Structured index buffers store indices as `Triangle`s, `Quad`s, or
//! `Polygon`s, all of which preserve the topology of a mesh even if its arity
//! is non-constant. These types only support triangles and quads; higher arity
//! N-gons are not supported.
//!
//! Flat index buffers store individual indices. Because there is no structure,
//! arity must by constant, but arbitrary N-gons are supported. Flat buffers
//! tend to be more useful for rendering, especially triangular flat buffers.
//!
//! The `MeshBuffer3` and `MeshBufferN` type aliases avoid verbose type
//! parameters and provide the most common index buffer configurations.
//!
//! # Examples
//!
//! Generating a flat `MeshBuffer` from a primitive:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point3;
//! use plexus::buffer::MeshBuffer3;
//! use plexus::prelude::*;
//! use plexus::primitive::sphere::UvSphere;
//!
//! # fn main() {
//! let buffer = UvSphere::new(16, 16)
//!     .polygons_with_position()
//!     .triangulate()
//!     .collect::<MeshBuffer3<u32, Point3<f32>>>();
//! let indices = buffer.as_index_slice();
//! let positions = buffer.as_vertex_slice();
//! # }
//! ```
//!
//! Converting a `MeshGraph` to a flat `MeshBuffer`:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point3;
//! use plexus::buffer::U4;
//! use plexus::graph::MeshGraph;
//! use plexus::prelude::*;
//! use plexus::primitive::cube::Cube;
//!
//! # fn main() {
//! let graph = Cube::new()
//!     .polygons_with_position()
//!     .collect::<MeshGraph<Point3<f32>>>();
//! let buffer = graph
//!     .to_mesh_buffer_by_vertex::<U4, u32, Point3<f32>>()
//!     .unwrap();
//! # }
//! ```

use itertools::Itertools;
use num::{Integer, NumCast, ToPrimitive, Unsigned};
use std::hash::Hash;
use std::iter::FromIterator;
use std::marker::PhantomData;
use typenum::{self, NonZero};

use crate::geometry::convert::IntoGeometry;
use crate::primitive::decompose::IntoVertices;
use crate::primitive::index::{
    FlatIndexVertices, FromIndexer, HashIndexer, IndexVertices, Indexer,
};
use crate::primitive::{Arity, Map, Polygon, Polygonal, Quad, Topological, Triangle};
use crate::FromRawBuffers;

pub use typenum::{U3, U4};

#[derive(Debug, Fail)]
pub enum BufferError {
    #[fail(display = "index into vertex data out of bounds")]
    IndexOutOfBounds,
    #[fail(display = "conflicting arity")]
    ArityConflict,
}

/// Index buffer.
///
/// Describes the contents of an index buffer. This includes the arity of the
/// buffer if constant.
pub trait IndexBuffer {
    type Item;

    // TODO: Use `Option<NonZeroU8>`.
    /// Arity of the index buffer (and by extension the mesh).
    const ARITY: Option<usize>;
}

/// Flat index buffer.
///
/// A flat (unstructured) index buffer with a constant and static arity. Arity
/// is specified using a type constant from the
/// [typenum](https://crates.io/crates/typenum) crate. `U3` and `U4` are
/// re-exported in the `buffer` module.
///
/// # Examples
///
/// Creating a flat and triangular `MeshBuffer`:
///
/// ```rust
/// use plexus::buffer::{Flat, MeshBuffer, U3};
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

impl<A, N> IndexBuffer for Flat<A, N>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
{
    /// Flat index buffers directly contain indices into vertex data. These
    /// indices are implicitly grouped by the arity of the buffer.
    type Item = N;

    /// Flat index buffers have a constant arity.
    const ARITY: Option<usize> = Some(A::USIZE);
}

/// Alias for a flat and triangular index buffer.
pub type Flat3<N = usize> = Flat<U3, N>;
/// Alias for a flat and quadrilateral index buffer.
pub type Flat4<N = usize> = Flat<U4, N>;

/// Alias for a flat and triangular `MeshBuffer`. Prefer this alias.
///
/// For most applications, this alias can be used to avoid more complex and
/// verbose type parameters. Flat and triangular index buffers are most common
/// and should generally be preferred.
pub type MeshBuffer3<N, G> = MeshBuffer<Flat3<N>, G>;
/// Alias for a flat and quadrilateral `MeshBuffer`.
pub type MeshBuffer4<N, G> = MeshBuffer<Flat4<N>, G>;

/// Structured index buffer.
///
/// A structured index buffer of triangles and/or quads. Useful if a buffer
/// representing a mesh comprised of both triangles and quads is needed (no
/// need for triangulation).
///
/// # Examples
///
/// Creating a structured `MeshBuffer`:
///
/// ```rust
/// use plexus::buffer::{MeshBuffer, Structured};
/// use plexus::prelude::*;
/// use plexus::primitive::Polygon;
///
/// let mut buffer = MeshBuffer::<Structured<Polygon<usize>>, Triplet<f64>>::default();
/// ```
#[derive(Debug)]
pub struct Structured<P = Polygon<usize>>
where
    P: Polygonal,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
{
    phantom: PhantomData<P>,
}

/// `Structured` index buffer that contains dynamic `Polygon`s.
impl<N> IndexBuffer for Structured<Polygon<N>>
where
    N: Copy + Integer + NumCast + Unsigned,
{
    /// `Polygon` index buffers contain topological structures that explicitly
    /// group their indices into vertex data.
    type Item = Polygon<N>;

    /// `Polygon` index buffers may or may not have a constant arity. Arity is
    /// encoded independently by each item in the buffer.
    const ARITY: Option<usize> = None;
}

/// `Structured` index buffer that contains `Polygonal` structures with
/// constant arity.
impl<P> IndexBuffer for Structured<P>
where
    P: Arity + Polygonal,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
{
    /// `Polygonal` index buffers contain topological structures that
    /// explicitly group their indices into vertex data.
    type Item = P;

    /// `Polygonal` index buffers have a constant arity.
    const ARITY: Option<usize> = Some(<P as Arity>::ARITY);
}

/// Alias for a structured and triangular index buffer.
pub type Structured3<N = usize> = Structured<Triangle<N>>;
/// Alias for a structured and quadrilateral index buffer.
pub type Structured4<N = usize> = Structured<Quad<N>>;
/// Alias for a structured and polygonal (variable arity) index buffer.
pub type StructuredN<N = usize> = Structured<Polygon<N>>;

/// Alias for a structured and polygonal `MeshBuffer`.
pub type MeshBufferN<N, G> = MeshBuffer<StructuredN<N>, G>;

pub trait IntoFlatIndex<A, G>
where
    A: NonZero + typenum::Unsigned,
{
    type Item: Copy + Integer + NumCast + Unsigned;

    fn into_flat_index(self) -> MeshBuffer<Flat<A, Self::Item>, G>;
}

pub trait IntoStructuredIndex<G>
where
    Structured<Self::Item>: IndexBuffer,
    <Self::Item as Topological>::Vertex: Copy + Integer + NumCast + Unsigned,
{
    type Item: Polygonal;

    fn into_structured_index(self) -> MeshBuffer<Structured<Self::Item>, G>;
}

/// Linear representation of a mesh.
///
/// A `MeshBuffer` is a linear representation of a mesh that can be consumed by
/// a graphics pipeline. A `MeshBuffer` is composed of two separate buffers:
/// an index buffer and a vertex buffer. The index buffer contains ordered
/// indices into the data in the vertex buffer and describes the topology of
/// the mesh. The vertex buffer contains arbitrary geometric data.
#[derive(Debug)]
pub struct MeshBuffer<I, G>
where
    I: IndexBuffer,
{
    indices: Vec<I::Item>,
    vertices: Vec<G>,
}

impl<I, G> MeshBuffer<I, G>
where
    I: IndexBuffer,
{
    /// Creates an empty `MeshBuffer`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::buffer::{Flat3, MeshBuffer};
    /// use plexus::geometry::Triplet;
    ///
    /// let buffer = MeshBuffer::<Flat3<u32>, Triplet<f64>>::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    pub fn into_raw_buffers(self) -> (Vec<I::Item>, Vec<G>) {
        let MeshBuffer { indices, vertices } = self;
        (indices, vertices)
    }

    /// Maps over the vertex data in a `MeshBuffer`.
    ///
    /// # Examples
    ///
    /// Translating the position data in a buffer:
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    ///
    /// use nalgebra::{Point3, Vector3};
    /// use plexus::buffer::MeshBuffer3;
    /// use plexus::prelude::*;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// # fn main() {
    /// let buffer = UvSphere::new(16, 8)
    ///     .polygons_with_position()
    ///     .triangulate()
    ///     .collect::<MeshBuffer3<usize, Point3<f64>>>();
    /// // Translate the positions.
    /// let translation = Vector3::new(1.0.into(), 0.0.into(), 0.0.into());
    /// let buffer = buffer.map_vertices_into(|position| position + translation);
    /// # }
    /// ```
    pub fn map_vertices_into<H, F>(self, f: F) -> MeshBuffer<I, H>
    where
        F: FnMut(G) -> H,
    {
        let (indices, vertices) = self.into_raw_buffers();
        MeshBuffer {
            indices,
            vertices: vertices.into_iter().map(f).collect::<Vec<_>>(),
        }
    }

    pub fn arity(&self) -> Option<usize> {
        I::ARITY
    }

    /// Gets a slice of the index data.
    pub fn as_index_slice(&self) -> &[I::Item] {
        self.indices.as_slice()
    }

    /// Gets a slice of the vertex data.
    pub fn as_vertex_slice(&self) -> &[G] {
        self.vertices.as_slice()
    }
}

impl<I, G> Default for MeshBuffer<I, G>
where
    I: IndexBuffer,
{
    fn default() -> Self {
        MeshBuffer {
            indices: Default::default(),
            vertices: Default::default(),
        }
    }
}

impl<A, N, G> MeshBuffer<Flat<A, N>, G>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
{
    /// Appends the contents of a `MeshBuffer` into another `MeshBuffer`. The
    /// source buffer is drained.
    pub fn append<U, H>(&mut self, buffer: &mut MeshBuffer<U, H>)
    where
        U: IndexBuffer,
        U::Item: Into<<Flat<A, N> as IndexBuffer>::Item>,
        H: IntoGeometry<G>,
    {
        let offset = N::from(self.vertices.len()).unwrap();
        self.vertices.extend(
            buffer
                .vertices
                .drain(..)
                .map(|vertex| vertex.into_geometry()),
        );
        self.indices
            .extend(buffer.indices.drain(..).map(|index| index.into() + offset))
    }
}

impl<P, G> MeshBuffer<Structured<P>, G>
where
    P: Polygonal,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
    Structured<P>: IndexBuffer,
{
    /// Converts a structured `MeshBuffer` into an iterator of polygons
    /// containing vertex data.
    ///
    /// # Examples
    ///
    /// Mapping over the polygons described by a buffer:
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    ///
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBufferN;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::sphere::UvSphere;
    /// use plexus::R64;
    ///
    /// # fn main() {
    /// let buffer = UvSphere::new(8, 8)
    ///     .polygons_with_position()
    ///     .collect::<MeshBufferN<usize, Point3<R64>>>();
    /// let graph = buffer
    ///     .into_polygons()
    ///     .map_vertices(|position| position * 2.0.into())
    ///     .triangulate()
    ///     .collect::<MeshGraph<Point3<R64>>>();
    /// # }
    /// ```
    pub fn into_polygons(
        self,
    ) -> impl Clone + Iterator<Item = <<Structured<P> as IndexBuffer>::Item as Map<G>>::Output>
    where
        G: Clone,
        <Structured<P> as IndexBuffer>::Item: Map<G>,
        <<Structured<P> as IndexBuffer>::Item as Map<G>>::Output: Clone,
        <<Structured<P> as IndexBuffer>::Item as Topological>::Vertex: ToPrimitive,
    {
        let (indices, vertices) = self.into_raw_buffers();
        indices
            .into_iter()
            .map(|polygon| {
                polygon.map(|index| vertices[<usize as NumCast>::from(index).unwrap()].clone())
            })
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Appends the contents of a `MeshBuffer` into another `MeshBuffer`. The
    /// source buffer is drained.
    pub fn append<U, H>(&mut self, buffer: &mut MeshBuffer<U, H>)
    where
        U: IndexBuffer,
        U::Item: Into<<Structured<P> as IndexBuffer>::Item>,
        H: IntoGeometry<G>,
        <Structured<P> as IndexBuffer>::Item: Copy
            + Map<Output = <Structured<P> as IndexBuffer>::Item>
            + Topological<Vertex = P::Vertex>,
    {
        let offset = <P::Vertex as NumCast>::from(self.vertices.len()).unwrap();
        self.vertices.extend(
            buffer
                .vertices
                .drain(..)
                .map(|vertex| vertex.into_geometry()),
        );
        self.indices.extend(
            buffer
                .indices
                .drain(..)
                .map(|topology| topology.into().map(|index| index + offset)),
        )
    }
}

impl<A, N, P, G> FromIndexer<P, P> for MeshBuffer<Flat<A, N>, G>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
    P: Arity + IntoVertices + Polygonal,
    P::Vertex: IntoGeometry<G>,
{
    type Error = BufferError;

    fn from_indexer<I, M>(input: I, indexer: M) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = P>,
        M: Indexer<P, P::Vertex>,
    {
        let (indices, vertices) = input.into_iter().flat_index_vertices(indexer);
        MeshBuffer::<Flat<_, _>, _>::from_raw_buffers(
            indices.into_iter().map(|index| N::from(index).unwrap()),
            vertices.into_iter().map(|vertex| vertex.into_geometry()),
        )
    }
}

impl<P, Q, G> FromIndexer<P, P> for MeshBuffer<Structured<Q>, G>
where
    P: Map<usize> + Polygonal,
    P::Output: Map<Q::Vertex>,
    P::Vertex: IntoGeometry<G>,
    Q: Polygonal,
    Q::Vertex: Copy + Integer + NumCast + Unsigned,
    Structured<Q>: IndexBuffer,
    <Structured<Q> as IndexBuffer>::Item: Copy
        + From<<P::Output as Map<Q::Vertex>>::Output>
        + IntoVertices
        + Topological<Vertex = Q::Vertex>,
{
    type Error = BufferError;

    fn from_indexer<I, M>(input: I, indexer: M) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = P>,
        M: Indexer<P, P::Vertex>,
    {
        let (indices, vertices) = input.into_iter().index_vertices(indexer);
        MeshBuffer::<Structured<_>, _>::from_raw_buffers(
            indices
                .into_iter()
                .map(|topology| topology.map(|index| <Q::Vertex as NumCast>::from(index).unwrap())),
            vertices.into_iter().map(|vertex| vertex.into_geometry()),
        )
    }
}

impl<A, N, P, G> FromIterator<P> for MeshBuffer<Flat<A, N>, G>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
    P: Arity + IntoVertices + Polygonal,
    P::Vertex: Copy + Eq + Hash + IntoGeometry<G>,
{
    fn from_iter<I>(input: I) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        Self::from_indexer(input, HashIndexer::default()).unwrap_or_else(|_| Self::default())
    }
}

impl<P, Q, G> FromIterator<P> for MeshBuffer<Structured<Q>, G>
where
    P: Map<usize> + Polygonal,
    P::Output: Map<Q::Vertex>,
    P::Vertex: Copy + Eq + Hash + IntoGeometry<G>,
    Q: Polygonal,
    Q::Vertex: Copy + Integer + NumCast + Unsigned,
    Structured<Q>: IndexBuffer,
    <Structured<Q> as IndexBuffer>::Item: Copy
        + From<<P::Output as Map<Q::Vertex>>::Output>
        + IntoVertices
        + Topological<Vertex = Q::Vertex>,
{
    fn from_iter<I>(input: I) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        Self::from_indexer(input, HashIndexer::default()).unwrap_or_else(|_| Self::default())
    }
}

impl<A, N, M, G> FromRawBuffers<M, G> for MeshBuffer<Flat<A, N>, G>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
    M: Copy + Integer + NumCast + Unsigned,
    <Flat<A, N> as IndexBuffer>::Item: ToPrimitive,
{
    type Error = BufferError;

    /// Creates a flat `MeshBuffer` from raw index and vertex buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if the index data is out of bounds within the vertex
    /// buffer or if the number of indices disagrees with the arity of the
    /// index buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::buffer::MeshBuffer3;
    /// use plexus::prelude::*;
    /// use plexus::primitive;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::index::HashIndexer;
    ///
    /// # fn main() {
    /// let cube = Cube::new();
    /// let (indices, vertices) = primitive::zip_vertices((
    ///     cube.polygons_with_position(),
    ///     cube.polygons_with_normal(),
    ///     cube.polygons_with_uv_map(),
    /// ))
    /// .flat_index_vertices(HashIndexer::default());
    /// let buffer = MeshBuffer3::<usize, _>::from_raw_buffers(indices, vertices).unwrap();
    /// # }
    /// ```
    fn from_raw_buffers<I, J>(indices: I, vertices: J) -> Result<Self, BufferError>
    where
        I: IntoIterator<Item = M>,
        J: IntoIterator<Item = G>,
    {
        let indices = indices
            .into_iter()
            .map(|index| <<Flat<A, N> as IndexBuffer>::Item as NumCast>::from(index).unwrap())
            .collect::<Vec<_>>();
        if indices.len() % Flat::<A, N>::ARITY.unwrap() != 0 {
            Err(BufferError::ArityConflict)
        }
        else {
            let vertices = vertices.into_iter().collect::<Vec<_>>();
            let len = N::from(vertices.len()).unwrap();
            if indices.iter().any(|index| *index >= len) {
                Err(BufferError::IndexOutOfBounds)
            }
            else {
                Ok(MeshBuffer { indices, vertices })
            }
        }
    }
}

impl<P, Q, G> FromRawBuffers<Q, G> for MeshBuffer<Structured<P>, G>
where
    P: Polygonal,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
    Q: Into<<Structured<P> as IndexBuffer>::Item>,
    Structured<P>: IndexBuffer,
    <Structured<P> as IndexBuffer>::Item: Copy + IntoVertices + Topological<Vertex = P::Vertex>,
{
    type Error = BufferError;

    /// Creates a structured `MeshBuffer` from raw index and vertex buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if the index data is out of bounds within the vertex
    /// buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBufferN;
    /// use plexus::prelude::*;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// # fn main() {
    /// let sphere = UvSphere::new(8, 8);
    /// let buffer = MeshBufferN::<usize, _>::from_raw_buffers(
    ///     sphere.indices_for_position(),
    ///     sphere
    ///         .vertices_with_position()
    ///         .map(|position| -> Point3<f32> { position.into() }),
    /// )
    /// .unwrap();
    /// # }
    /// ```
    fn from_raw_buffers<I, J>(indices: I, vertices: J) -> Result<Self, BufferError>
    where
        I: IntoIterator<Item = Q>,
        J: IntoIterator<Item = G>,
    {
        let indices = indices
            .into_iter()
            .map(|topology| topology.into())
            .collect::<Vec<_>>();
        let vertices = vertices
            .into_iter()
            .map(|geometry| geometry)
            .collect::<Vec<_>>();
        let len = <P::Vertex as NumCast>::from(vertices.len()).unwrap();
        if indices.iter().any(|polygon| {
            polygon
                .into_vertices()
                .into_iter()
                .any(|index| index >= len)
        }) {
            Err(BufferError::IndexOutOfBounds)
        }
        else {
            Ok(MeshBuffer { indices, vertices })
        }
    }
}

impl<A, N, G> IntoFlatIndex<A, G> for MeshBuffer<Flat<A, N>, G>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
{
    type Item = N;

    fn into_flat_index(self) -> MeshBuffer<Flat<A, Self::Item>, G> {
        self
    }
}

impl<N, G> IntoFlatIndex<U3, G> for MeshBuffer<Structured3<N>, G>
where
    N: Copy + Integer + NumCast + Unsigned,
{
    type Item = N;

    /// Converts a structured index buffer into a flat index buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// use nalgebra::Point3;
    /// use plexus::buffer::{MeshBuffer, Structured3};
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    ///
    /// # fn main() {
    /// let cube = Cube::new();
    /// let buffer = MeshBuffer::<Structured3, _>::from_raw_buffers(
    ///     cube.indices_for_position().triangulate(),
    ///     cube.vertices_with_position(),
    /// )
    /// .unwrap();
    /// let buffer = buffer.into_flat_index();
    /// for index in buffer.as_index_slice() {
    ///     // ...
    /// }
    /// # }
    /// ```
    fn into_flat_index(self) -> MeshBuffer<Flat<U3, Self::Item>, G> {
        let MeshBuffer { indices, vertices } = self;
        MeshBuffer {
            indices: indices
                .into_iter()
                .flat_map(|triangle| triangle.into_vertices())
                .collect(),
            vertices,
        }
    }
}

impl<N, G> IntoFlatIndex<U4, G> for MeshBuffer<Structured4<N>, G>
where
    N: Copy + Integer + NumCast + Unsigned,
{
    type Item = N;

    /// Converts a structured index buffer into a flat index buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// use nalgebra::Point3;
    /// use plexus::buffer::{MeshBuffer, Structured4};
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    ///
    /// # fn main() {
    /// let cube = Cube::new();
    /// let buffer = MeshBuffer::<Structured4, _>::from_raw_buffers(
    ///     cube.indices_for_position(),
    ///     cube.vertices_with_position(),
    /// )
    /// .unwrap();
    /// let buffer = buffer.into_flat_index();
    /// for index in buffer.as_index_slice() {
    ///     // ...
    /// }
    /// # }
    /// ```
    fn into_flat_index(self) -> MeshBuffer<Flat<U4, Self::Item>, G> {
        let MeshBuffer { indices, vertices } = self;
        MeshBuffer {
            indices: indices
                .into_iter()
                .flat_map(|quad| quad.into_vertices())
                .collect(),
            vertices,
        }
    }
}

impl<P, G> IntoStructuredIndex<G> for MeshBuffer<Structured<P>, G>
where
    P: Polygonal,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
    Structured<P>: IndexBuffer,
{
    type Item = P;

    fn into_structured_index(self) -> MeshBuffer<Structured<Self::Item>, G> {
        self
    }
}

impl<N, G> IntoStructuredIndex<G> for MeshBuffer<Flat3<N>, G>
where
    N: Copy + Integer + NumCast + Unsigned,
{
    type Item = Triangle<N>;

    /// Converts a flat index buffer into a structured index buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer3;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    ///
    /// # fn main() {
    /// let cube = Cube::new();
    /// let buffer = MeshBuffer3::<usize, _>::from_raw_buffers(
    ///     cube.indices_for_position().triangulate().vertices(),
    ///     cube.vertices_with_position(),
    /// )
    /// .unwrap();
    /// let buffer = buffer.into_structured_index();
    /// for triangle in buffer.as_index_slice() {
    ///     // ...
    /// }
    /// # }
    /// ```
    fn into_structured_index(self) -> MeshBuffer<Structured<Self::Item>, G> {
        let MeshBuffer { indices, vertices } = self;
        MeshBuffer {
            indices: indices
                .into_iter()
                .chunks(Flat3::<N>::ARITY.unwrap())
                .into_iter()
                .map(|triangle| <Structured<Self::Item> as IndexBuffer>::Item::from_iter(triangle))
                .collect(),
            vertices,
        }
    }
}

impl<N, G> IntoStructuredIndex<G> for MeshBuffer<Flat4<N>, G>
where
    N: Copy + Integer + NumCast + Unsigned,
{
    type Item = Quad<N>;

    /// Converts a flat index buffer into a structured index buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer4;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    ///
    /// # fn main() {
    /// let cube = Cube::new();
    /// let buffer = MeshBuffer4::<usize, _>::from_raw_buffers(
    ///     cube.indices_for_position().vertices(),
    ///     cube.vertices_with_position(),
    /// )
    /// .unwrap();
    /// let buffer = buffer.into_structured_index();
    /// for quad in buffer.as_index_slice() {
    ///     // ...
    /// }
    /// # }
    /// ```
    fn into_structured_index(self) -> MeshBuffer<Structured<Self::Item>, G> {
        let MeshBuffer { indices, vertices } = self;
        MeshBuffer {
            indices: indices
                .into_iter()
                .chunks(Flat4::<N>::ARITY.unwrap())
                .into_iter()
                .map(|quad| <Structured<Self::Item> as IndexBuffer>::Item::from_iter(quad))
                .collect(),
            vertices,
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use crate::buffer::*;
    use crate::graph::*;
    use crate::primitive::cube::Cube;
    use crate::primitive::decompose::*;
    use crate::primitive::generate::*;
    use crate::primitive::sphere::UvSphere;

    #[test]
    fn collect_topology_into_flat_buffer() {
        let buffer = UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .triangulate()
            .collect::<MeshBuffer3<u32, Point3<f32>>>();

        assert_eq!(18, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
    }

    #[test]
    fn collect_topology_into_structured_buffer() {
        let buffer = UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<MeshBufferN<u32, Point3<f32>>>();

        assert_eq!(6, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
    }

    #[test]
    fn append_structured_buffers() {
        let mut buffer = UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<MeshBuffer<StructuredN<u32>, Point3<f32>>>();
        buffer.append(
            &mut Cube::new()
                .polygons_with_position() // 6 quads, 24 vertices.
                .collect::<MeshBuffer<Structured4<u32>, Point3<f32>>>(),
        );

        assert_eq!(12, buffer.as_index_slice().len());
        assert_eq!(13, buffer.as_vertex_slice().len());
    }

    #[test]
    fn convert_mesh_to_buffer_by_vertex() {
        let graph = UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f32>>>();
        let buffer = graph
            .to_mesh_buffer_by_vertex::<U3, u32, Point3<f32>>()
            .unwrap();

        assert_eq!(18, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
    }

    #[test]
    fn convert_mesh_to_buffer_by_face() {
        let graph = UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f32>>>();
        let buffer = graph
            .to_mesh_buffer_by_face::<U3, u32, Point3<f32>>()
            .unwrap();

        assert_eq!(18, buffer.as_index_slice().len());
        assert_eq!(18, buffer.as_vertex_slice().len());
    }
}
