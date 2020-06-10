//! Linear representation of meshes.
//!
//! This module provides a `MeshBuffer` that represents a mesh as a linear
//! collection of vertex geometry and an ordered collection of indices into that
//! vertex geometry. These two buffers are called the _vertex buffer_ and _index
//! buffer_, respectively. `MeshBuffer` combines these buffers and exposes them
//! as slices. This layout is well-suited for graphics pipelines.
//!
//! # Vertex Buffers
//!
//! Vertex buffers describe the geometry of a `MeshBuffer`. Only vertex geometry
//! is supported; there is no way to associate geometry with an edge nor face,
//! for example.
//!
//! `MeshBuffer`s use _composite_ vertex buffers. Each element of the vertex
//! buffer completely describes the geometry of that vertex. For example, if
//! each vertex is described by a position and color attribute, then each
//! element in the vertex buffer contains both attributes within a single
//! structure. _Component_ buffers, which store attributes in separate buffers,
//! are not supported.
//!
//! # Index Buffers
//!
//! Index buffers describe the topology of a `MeshBuffer`. Both _structured_ and
//! _flat_ index buffers are supported. See the `index` module for more
//! information about index buffer formats.
//!
//! The `MeshBuffer3` and `MeshBufferN` type definitions avoid verbose type
//! parameters and provide the most common index buffer configurations.
//!
//! # Examples
//!
//! Generating a flat `MeshBuffer` from a $uv$-sphere:
//!
//! ```rust
//! # extern crate decorum;
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use decorum::N32;
//! use nalgebra::Point3;
//! use plexus::buffer::MeshBuffer3;
//! use plexus::prelude::*;
//! use plexus::primitive::generate::Position;
//! use plexus::primitive::sphere::UvSphere;
//!
//! let buffer = UvSphere::new(16, 16)
//!     .polygons::<Position<Point3<N32>>>()
//!     .triangulate()
//!     .collect::<MeshBuffer3<u32, Point3<f32>>>();
//! let indices = buffer.as_index_slice();
//! let positions = buffer.as_vertex_slice();
//! ```
//!
//! Converting a `MeshGraph` to a flat `MeshBuffer`:
//!
//! ```rust
//! # extern crate decorum;
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use decorum::N64;
//! use nalgebra::Point3;
//! use plexus::buffer::MeshBuffer4;
//! use plexus::graph::MeshGraph;
//! use plexus::prelude::*;
//! use plexus::primitive::cube::Cube;
//! use plexus::primitive::generate::Position;
//!
//! let graph = Cube::new()
//!     .polygons::<Position<Point3<N64>>>()
//!     .collect::<MeshGraph<Point3<N64>>>();
//! let buffer = graph
//!     .to_mesh_by_vertex::<MeshBuffer4<usize, Point3<N64>>>()
//!     .unwrap();
//! ```

mod builder;

use itertools::Itertools;
use num::{Integer, NumCast, ToPrimitive, Unsigned};
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::FromIterator;
use theon::adjunct::{FromItems, Map};
use thiserror::Error;
use typenum::{self, NonZero, Unsigned as _, U3, U4};

use crate::buffer::builder::BufferBuilder;
use crate::builder::{Buildable, MeshBuilder};
use crate::encoding::{FaceDecoder, FromEncoding, VertexDecoder};
use crate::index::{
    Flat, Flat3, Flat4, FromIndexer, Grouping, HashIndexer, IndexBuffer, IndexVertices, Indexer,
    Push,
};
use crate::primitive::decompose::IntoVertices;
use crate::primitive::{BoundedPolygon, Polygonal, Tetragon, Topological, Trigon};
use crate::IntoGeometry;
use crate::{DynamicArity, MeshArity, Monomorphic, StaticArity};

#[derive(Debug, Error, PartialEq)]
pub enum BufferError {
    #[error("index into vertex data out of bounds")]
    IndexOutOfBounds,
    #[error("index buffer conflicts with arity")]
    IndexUnaligned,
    #[error("conflicting arity; expected {expected}, but got {actual}")]
    ArityConflict { expected: usize, actual: usize },
}

/// Alias for a flat and triangular `MeshBuffer`.
///
/// For most applications, this alias can be used to avoid more complex and
/// verbose type parameters. Flat and triangular index buffers are most common
/// and should generally be preferred.
pub type MeshBuffer3<N, G> = MeshBuffer<Flat3<N>, G>;
/// Alias for a flat and quadrilateral `MeshBuffer`.
pub type MeshBuffer4<N, G> = MeshBuffer<Flat4<N>, G>;

/// Alias for a structured and polygonal `MeshBuffer`.
pub type MeshBufferN<N, G> = MeshBuffer<BoundedPolygon<N>, G>;

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

pub trait IntoFlatIndex<A, G>
where
    A: NonZero + typenum::Unsigned,
{
    type Item: Copy + Integer + NumCast + Unsigned;

    fn into_flat_index(self) -> MeshBuffer<Flat<A, Self::Item>, G>;
}

pub trait IntoStructuredIndex<G>
where
    <Self::Item as Topological>::Vertex: Copy + Integer + NumCast + Unsigned,
{
    type Item: Polygonal;

    fn into_structured_index(self) -> MeshBuffer<Self::Item, G>;
}

/// Linear representation of a mesh.
///
/// A `MeshBuffer` is a linear representation of a mesh that is composed of two
/// separate buffers: an _index buffer_ and a _vertex buffer_. The index buffer
/// contains ordered indices into the data in the vertex buffer and describes
/// the topology of the mesh. The vertex buffer contains arbitrary geometric
/// data.
///
/// See the module documention for more information.
#[derive(Debug)]
pub struct MeshBuffer<R, G>
where
    R: Grouping,
{
    indices: Vec<R::Item>,
    vertices: Vec<G>,
}

impl<R, G> MeshBuffer<R, G>
where
    R: Grouping,
    Vec<R::Item>: IndexBuffer<R>,
{
    pub(in crate::buffer) fn from_raw_buffers_unchecked(
        indices: Vec<R::Item>,
        vertices: Vec<G>,
    ) -> Self {
        MeshBuffer { indices, vertices }
    }

    /// Creates an empty `MeshBuffer`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::buffer::MeshBuffer;
    /// use plexus::index::Flat3;
    ///
    /// let buffer = MeshBuffer::<Flat3<u64>, (f64, f64, f64)>::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }
}

impl<R, G> MeshBuffer<R, G>
where
    R: Grouping,
{
    pub fn into_raw_buffers(self) -> (Vec<R::Item>, Vec<G>) {
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
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::N64;
    /// use nalgebra::{Point3, Vector3};
    /// use plexus::buffer::MeshBuffer3;
    /// use plexus::prelude::*;
    /// use plexus::primitive::generate::Position;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// let buffer = UvSphere::new(16, 8)
    ///     .polygons::<Position<Point3<N64>>>()
    ///     .triangulate()
    ///     .collect::<MeshBuffer3<usize, Point3<f64>>>();
    /// // Translate the positions.
    /// let translation = Vector3::<f64>::x() * 2.0;
    /// let buffer = buffer.map_vertices(|position| position + translation);
    /// ```
    pub fn map_vertices<H, F>(self, f: F) -> MeshBuffer<R, H>
    where
        F: FnMut(G) -> H,
    {
        let (indices, vertices) = self.into_raw_buffers();
        MeshBuffer {
            indices,
            vertices: vertices.into_iter().map(f).collect::<Vec<_>>(),
        }
    }

    /// Gets a slice of the index data.
    pub fn as_index_slice(&self) -> &[R::Item] {
        self.indices.as_slice()
    }

    /// Gets a slice of the vertex data.
    pub fn as_vertex_slice(&self) -> &[G] {
        self.vertices.as_slice()
    }
}

/// Exposes a `MeshBuilder` that can be used to construct a `MeshBuffer`
/// incrementally from _surfaces_ and _facets_.
///
/// Note that the facet geometry for `MeshBuffer` is always the unit type `()`.
///
/// See the documentation for the `builder` module for more.
///
/// # Examples
///
/// Creating a buffer from a triangle:
///
/// ```rust
/// # extern crate nalgebra;
/// # extern crate plexus;
/// #
/// use nalgebra::Point2;
/// use plexus::buffer::MeshBuffer3;
/// use plexus::builder::Buildable;
/// use plexus::prelude::*;
///
/// let mut builder = MeshBuffer3::<usize, Point2<f64>>::builder();
/// let buffer = builder
///     .surface_with(|builder| {
///         let a = builder.insert_vertex((0.0, 0.0))?;
///         let b = builder.insert_vertex((1.0, 0.0))?;
///         let c = builder.insert_vertex((0.0, 1.0))?;
///         builder.facets_with(|builder| builder.insert_facet(&[a, b, c], ()))
///     })
///     .and_then(|_| builder.build())
///     .unwrap();
/// ```
impl<R, G> Buildable for MeshBuffer<R, G>
where
    R: Grouping,
    Vec<R::Item>: IndexBuffer<R>,
    BufferBuilder<R, G>: MeshBuilder<Error = BufferError, Output = Self, Vertex = G, Facet = ()>,
{
    type Builder = BufferBuilder<R, G>;
    type Error = BufferError;

    type Vertex = G;
    type Facet = ();

    fn builder() -> Self::Builder {
        BufferBuilder::default()
    }
}

impl<R, G> Default for MeshBuffer<R, G>
where
    R: Grouping,
    Vec<R::Item>: IndexBuffer<R>,
{
    fn default() -> Self {
        MeshBuffer {
            indices: Default::default(),
            vertices: Default::default(),
        }
    }
}

impl<A, N, G> DynamicArity for MeshBuffer<Flat<A, N>, G>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
{
    type Dynamic = <Flat<A, N> as StaticArity>::Static;

    fn arity(&self) -> Self::Dynamic {
        Flat::<A, N>::ARITY
    }
}

impl<P, G> DynamicArity for MeshBuffer<P, G>
where
    P: Grouping + Monomorphic + Polygonal,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
{
    type Dynamic = <P as StaticArity>::Static;

    fn arity(&self) -> Self::Dynamic {
        P::ARITY
    }
}

impl<N, G> DynamicArity for MeshBuffer<BoundedPolygon<N>, G>
where
    N: Copy + Integer + NumCast + Unsigned,
{
    type Dynamic = MeshArity;

    fn arity(&self) -> Self::Dynamic {
        MeshArity::from_components::<BoundedPolygon<N>, _>(self.indices.iter())
    }
}

impl<R, G> Monomorphic for MeshBuffer<R, G> where R: Grouping + Monomorphic {}

impl<R, G> StaticArity for MeshBuffer<R, G>
where
    R: Grouping,
{
    type Static = <R as StaticArity>::Static;

    const ARITY: Self::Static = R::ARITY;
}

impl<A, N, G> MeshBuffer<Flat<A, N>, G>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
{
    /// Appends the contents of a `MeshBuffer` into another `MeshBuffer`. The
    /// source buffer is drained.
    pub fn append<R, H>(&mut self, buffer: &mut MeshBuffer<R, H>)
    where
        R: Grouping,
        R::Item: Into<<Flat<A, N> as Grouping>::Item>,
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

impl<P, G> MeshBuffer<P, G>
where
    P: Grouping + Polygonal,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
{
    /// Converts a structured `MeshBuffer` into an iterator of polygons
    /// containing vertex data.
    ///
    /// # Examples
    ///
    /// Mapping over the polygons described by a buffer:
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::R64;
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBufferN;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::generate::Position;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// let buffer = UvSphere::new(8, 8)
    ///     .polygons::<Position<Point3<R64>>>()
    ///     .collect::<MeshBufferN<usize, Point3<R64>>>();
    /// let graph = buffer
    ///     .into_polygons()
    ///     .map_vertices(|position| position * 2.0.into())
    ///     .triangulate()
    ///     .collect::<MeshGraph<Point3<R64>>>();
    /// ```
    pub fn into_polygons(self) -> impl Iterator<Item = <<P as Grouping>::Item as Map<G>>::Output>
    where
        G: Clone,
        <P as Grouping>::Item: Map<G> + Topological,
        <<P as Grouping>::Item as Topological>::Vertex: ToPrimitive,
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
    pub fn append<R, H>(&mut self, buffer: &mut MeshBuffer<R, H>)
    where
        R: Grouping,
        R::Item: Into<<P as Grouping>::Item>,
        H: IntoGeometry<G>,
        <P as Grouping>::Item:
            Copy + Map<P::Vertex, Output = <P as Grouping>::Item> + Topological<Vertex = P::Vertex>,
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

impl<E, G> FromEncoding<E> for MeshBuffer<BoundedPolygon<usize>, G>
where
    E: FaceDecoder<Face = (), Index = BoundedPolygon<usize>> + VertexDecoder,
    E::Vertex: IntoGeometry<G>,
{
    type Error = BufferError;

    fn from_encoding(
        vertices: <E as VertexDecoder>::Output,
        faces: <E as FaceDecoder>::Output,
    ) -> Result<Self, Self::Error> {
        let indices = faces.into_iter().map(|(index, _)| index);
        let vertices = vertices.into_iter().map(|vertex| vertex.into_geometry());
        MeshBuffer::from_raw_buffers(indices, vertices)
    }
}

impl<R, P, G> FromIndexer<P, P> for MeshBuffer<R, G>
where
    R: Grouping,
    P: Map<<Vec<R::Item> as IndexBuffer<R>>::Index> + Topological,
    P::Output: Topological<Vertex = <Vec<R::Item> as IndexBuffer<R>>::Index>,
    P::Vertex: IntoGeometry<G>,
    Vec<R::Item>: Push<R, P::Output>,
    Self: FromRawBuffers<R::Item, G>,
{
    type Error = <Self as FromRawBuffers<R::Item, G>>::Error;

    fn from_indexer<I, M>(input: I, indexer: M) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = P>,
        M: Indexer<P, P::Vertex>,
    {
        let (indices, vertices) = input.into_iter().index_vertices(indexer);
        MeshBuffer::<R, _>::from_raw_buffers(
            indices,
            vertices.into_iter().map(|vertex| vertex.into_geometry()),
        )
    }
}

impl<R, P, G> FromIterator<P> for MeshBuffer<R, G>
where
    R: Grouping,
    P: Topological,
    P::Vertex: Copy + Eq + Hash + IntoGeometry<G>,
    Vec<R::Item>: IndexBuffer<R>,
    Self: FromIndexer<P, P>,
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
    <Flat<A, N> as Grouping>::Item: ToPrimitive,
{
    type Error = BufferError;

    /// Creates a flat `MeshBuffer` from raw index and vertex buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if the index data is out of bounds within the vertex
    /// buffer or if the number of indices disagrees with the arity of the index
    /// buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::N64;
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer3;
    /// use plexus::index::{Flat3, HashIndexer};
    /// use plexus::prelude::*;
    /// use plexus::primitive;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::{Normal, Position};
    ///
    /// type E3 = Point3<N64>;
    ///
    /// let cube = Cube::new();
    /// let (indices, vertices) = primitive::zip_vertices((
    ///     cube.polygons::<Position<E3>>(),
    ///     cube.polygons::<Normal<E3>>()
    ///         .map_vertices(|normal| normal.into_inner()),
    /// ))
    /// .triangulate()
    /// .index_vertices::<Flat3, _>(HashIndexer::default());
    /// let buffer = MeshBuffer3::<usize, _>::from_raw_buffers(indices, vertices).unwrap();
    /// ```
    fn from_raw_buffers<I, J>(indices: I, vertices: J) -> Result<Self, BufferError>
    where
        I: IntoIterator<Item = M>,
        J: IntoIterator<Item = G>,
    {
        let indices = indices
            .into_iter()
            .map(|index| <<Flat<A, N> as Grouping>::Item as NumCast>::from(index).unwrap())
            .collect::<Vec<_>>();
        if indices.len() % A::USIZE != 0 {
            Err(BufferError::IndexUnaligned)
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

impl<P, Q, G> FromRawBuffers<Q, G> for MeshBuffer<P, G>
where
    P: Grouping + Polygonal,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
    Q: Into<<P as Grouping>::Item>,
    <P as Grouping>::Item: Copy + IntoVertices + Topological<Vertex = P::Vertex>,
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
    /// #
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBufferN;
    /// use plexus::prelude::*;
    /// use plexus::primitive::generate::Position;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// let sphere = UvSphere::new(8, 8);
    /// let buffer = MeshBufferN::<usize, _>::from_raw_buffers(
    ///     sphere.indexing_polygons::<Position>(),
    ///     sphere.vertices::<Position<Point3<f64>>>(),
    /// )
    /// .unwrap();
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
        let is_out_of_bounds = {
            let len = <P::Vertex as NumCast>::from(vertices.len()).unwrap();
            indices.iter().any(|polygon| {
                polygon
                    .into_vertices()
                    .into_iter()
                    .any(|index| index >= len)
            })
        };
        if is_out_of_bounds {
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

impl<N, G> IntoFlatIndex<U3, G> for MeshBuffer<Trigon<N>, G>
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
    /// #
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    /// use plexus::primitive::Trigon;
    ///
    /// let cube = Cube::new();
    /// let buffer = MeshBuffer::<Trigon<usize>, _>::from_raw_buffers(
    ///     cube.indexing_polygons::<Position>().triangulate(),
    ///     cube.vertices::<Position<Point3<f32>>>(),
    /// )
    /// .unwrap();
    /// let buffer = buffer.into_flat_index();
    /// for index in buffer.as_index_slice() {
    ///     // ...
    /// }
    /// ```
    fn into_flat_index(self) -> MeshBuffer<Flat<U3, Self::Item>, G> {
        let MeshBuffer { indices, vertices } = self;
        MeshBuffer {
            indices: indices
                .into_iter()
                .flat_map(|trigon| trigon.into_vertices())
                .collect(),
            vertices,
        }
    }
}

impl<N, G> IntoFlatIndex<U4, G> for MeshBuffer<Tetragon<N>, G>
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
    /// #
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    /// use plexus::primitive::Tetragon;
    ///
    /// let cube = Cube::new();
    /// let buffer = MeshBuffer::<Tetragon<usize>, _>::from_raw_buffers(
    ///     cube.indexing_polygons::<Position>(),
    ///     cube.vertices::<Position<Point3<f64>>>(),
    /// )
    /// .unwrap();
    /// let buffer = buffer.into_flat_index();
    /// for index in buffer.as_index_slice() {
    ///     // ...
    /// }
    /// ```
    fn into_flat_index(self) -> MeshBuffer<Flat<U4, Self::Item>, G> {
        let MeshBuffer { indices, vertices } = self;
        MeshBuffer {
            indices: indices
                .into_iter()
                .flat_map(|tetragon| tetragon.into_vertices())
                .collect(),
            vertices,
        }
    }
}

impl<P, G> IntoStructuredIndex<G> for MeshBuffer<P, G>
where
    P: Grouping + Polygonal,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
{
    type Item = P;

    fn into_structured_index(self) -> MeshBuffer<Self::Item, G> {
        self
    }
}

impl<N, G> IntoStructuredIndex<G> for MeshBuffer<Flat3<N>, G>
where
    N: Copy + Integer + NumCast + Unsigned,
    Trigon<N>: Grouping<Item = Trigon<N>>,
{
    type Item = Trigon<N>;

    /// Converts a flat index buffer into a structured index buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer3;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    ///
    /// let cube = Cube::new();
    /// let buffer = MeshBuffer3::<usize, _>::from_raw_buffers(
    ///     cube.indexing_polygons::<Position>()
    ///         .triangulate()
    ///         .vertices(),
    ///     cube.vertices::<Position<Point3<f32>>>(),
    /// )
    /// .unwrap();
    /// let buffer = buffer.into_structured_index();
    /// for trigon in buffer.as_index_slice() {
    ///     // ...
    /// }
    /// ```
    fn into_structured_index(self) -> MeshBuffer<Self::Item, G> {
        let MeshBuffer { indices, vertices } = self;
        let indices = indices
            .into_iter()
            .chunks(U3::USIZE)
            .into_iter()
            .map(<Self::Item as Grouping>::Item::from_items)
            .collect::<Option<Vec<_>>>()
            .expect("inconsistent index buffer");
        MeshBuffer { indices, vertices }
    }
}

impl<N, G> IntoStructuredIndex<G> for MeshBuffer<Flat4<N>, G>
where
    N: Copy + Integer + NumCast + Unsigned,
    Tetragon<N>: Grouping<Item = Tetragon<N>>,
{
    type Item = Tetragon<N>;

    /// Converts a flat index buffer into a structured index buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer4;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    ///
    /// let cube = Cube::new();
    /// let buffer = MeshBuffer4::<usize, _>::from_raw_buffers(
    ///     cube.indexing_polygons::<Position>().vertices(),
    ///     cube.vertices::<Position<Point3<f64>>>(),
    /// )
    /// .unwrap();
    /// let buffer = buffer.into_structured_index();
    /// for tetragon in buffer.as_index_slice() {
    ///     // ...
    /// }
    /// ```
    fn into_structured_index(self) -> MeshBuffer<Self::Item, G> {
        let MeshBuffer { indices, vertices } = self;
        let indices = indices
            .into_iter()
            .chunks(U4::USIZE)
            .into_iter()
            .map(<Self::Item as Grouping>::Item::from_items)
            .collect::<Option<Vec<_>>>()
            .expect("inconsistent index buffer");
        MeshBuffer { indices, vertices }
    }
}

#[cfg(test)]
mod tests {
    use decorum::N64;
    use nalgebra::Point3;

    use crate::buffer::{MeshBuffer, MeshBuffer3, MeshBufferN};
    use crate::graph::MeshGraph;
    use crate::prelude::*;
    use crate::primitive::cube::Cube;
    use crate::primitive::generate::Position;
    use crate::primitive::sphere::UvSphere;
    use crate::primitive::{BoundedPolygon, Tetragon};

    type E3 = Point3<N64>;

    #[test]
    fn collect_topology_into_flat_buffer() {
        let buffer = UvSphere::new(3, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .triangulate()
            .collect::<MeshBuffer3<u32, Point3<f64>>>();

        assert_eq!(18, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
    }

    #[test]
    fn collect_topology_into_structured_buffer() {
        let buffer = UvSphere::new(3, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect::<MeshBufferN<u32, Point3<f64>>>();

        assert_eq!(6, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
    }

    #[test]
    fn append_structured_buffers() {
        let mut buffer = UvSphere::new(3, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect::<MeshBuffer<BoundedPolygon<u32>, Point3<f64>>>();
        buffer.append(
            &mut Cube::new()
                .polygons::<Position<E3>>() // 6 quadrilaterals, 24 vertices.
                .collect::<MeshBuffer<Tetragon<u32>, Point3<f64>>>(),
        );

        assert_eq!(12, buffer.as_index_slice().len());
        assert_eq!(13, buffer.as_vertex_slice().len());
    }

    #[test]
    fn convert_mesh_to_buffer_by_vertex() {
        let graph = UvSphere::new(3, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f64>>>();
        let buffer = graph
            .to_mesh_by_vertex::<MeshBuffer3<u32, Point3<f64>>>()
            .unwrap();

        assert_eq!(18, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
    }

    #[test]
    fn convert_mesh_to_buffer_by_face() {
        let graph = UvSphere::new(3, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f64>>>();
        let buffer = graph
            .to_mesh_by_face::<MeshBuffer3<u32, Point3<f64>>>()
            .unwrap();

        assert_eq!(18, buffer.as_index_slice().len());
        assert_eq!(18, buffer.as_vertex_slice().len());
    }
}
