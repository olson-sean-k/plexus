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
//! let buffer: MeshBuffer3<u32, Point3<f32>> = UvSphere::new(16, 16)
//!     .polygons::<Position<Point3<N32>>>()
//!     .triangulate()
//!     .collect();
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
//! use decorum::R64;
//! use nalgebra::Point3;
//! use plexus::buffer::MeshBuffer4;
//! use plexus::graph::MeshGraph;
//! use plexus::prelude::*;
//! use plexus::primitive::cube::Cube;
//! use plexus::primitive::generate::Position;
//!
//! type E3 = Point3<R64>;
//!
//! let graph: MeshGraph<E3> = Cube::new().polygons::<Position<E3>>().collect();
//! let buffer: MeshBuffer4<usize, E3> = graph.to_mesh_by_vertex().unwrap();
//! ```

// `MeshBuffer`s must convert and sum indices into their vertex data. Some of
// these conversions may fail, but others are never expected to fail, because of
// invariants enforced by `MeshBuffer`.
//
// Index types require `Unsigned` and `Vec` capacity is limited by word size
// (the width of `usize`). An overflow cannot occur in some contexts, because a
// consistent `MeshBuffer` cannot index into a `Vec` with an index larger than
// its maximum addressable capacity (the maximum value that `usize` can
// represent).

// TODO: More consistently `expect` or `ok_or_else` index conversions and sums.

mod builder;

use itertools::Itertools;
use num::{Integer, NumCast, Unsigned};
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::FromIterator;
use std::vec;
use theon::adjunct::{FromItems, Map};
use thiserror::Error;
use typenum::{self, NonZero, Unsigned as _, U3, U4};

use crate::buffer::builder::BufferBuilder;
use crate::builder::{Buildable, MeshBuilder};
use crate::encoding::{FaceDecoder, FromEncoding, VertexDecoder};
use crate::geometry::{FromGeometry, IntoGeometry};
use crate::index::{
    BufferOf, Flat, Flat3, Flat4, FromIndexer, Grouping, HashIndexer, IndexBuffer, IndexOf,
    IndexVertices, Indexer, Push,
};
use crate::primitive::decompose::IntoVertices;
use crate::primitive::{
    BoundedPolygon, IntoIndexed, IntoPolygons, Polygonal, Tetragon, Topological, Trigon,
    UnboundedPolygon,
};
use crate::{Arity, DynamicArity, MeshArity, Monomorphic, StaticArity};

#[derive(Debug, Error, PartialEq)]
pub enum BufferError {
    #[error("index into vertex data out of bounds")]
    IndexOutOfBounds,
    #[error("index overflow")]
    IndexOverflow,
    #[error("index buffer conflicts with arity")]
    IndexUnaligned,
    #[error("conflicting arity; expected {expected}, but got {actual}")]
    ArityConflict { expected: usize, actual: usize },
}

/// Alias for a triangular `MeshBuffer`.
///
/// The index buffer for this type contains `Trigon`s. This should be preferred
/// over flat index buffers, which are more prone to error. For applications
/// where a flat index buffer is necessary, consider `IntoFlatIndex` or the
/// `Flat3` meta-grouping.
pub type MeshBuffer3<N, G> = MeshBuffer<Trigon<N>, G>;

/// Alias for a quadrilateral `MeshBuffer`.
///
/// The index buffer for this type contains `Tetragon`s. This should be
/// preferred over flat index buffers, which are more prone to error.
pub type MeshBuffer4<N, G> = MeshBuffer<Tetragon<N>, G>;

/// Alias for a `MeshBuffer` that supports arbitrary polygons.
pub type MeshBufferN<N, G> = MeshBuffer<UnboundedPolygon<N>, G>;

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
    type Item: Copy + Integer + Unsigned;

    fn into_flat_index(self) -> MeshBuffer<Flat<A, Self::Item>, G>;
}

pub trait IntoStructuredIndex<G>
where
    <Self::Item as Topological>::Vertex: Copy + Integer + Unsigned,
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
    indices: Vec<R::Group>,
    vertices: Vec<G>,
}

impl<R, G> MeshBuffer<R, G>
where
    R: Grouping,
    Vec<R::Group>: IndexBuffer<R>,
{
    pub(in crate::buffer) fn from_raw_buffers_unchecked(
        indices: Vec<R::Group>,
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
    pub fn into_raw_buffers(self) -> (Vec<R::Group>, Vec<G>) {
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
    /// let buffer: MeshBuffer3<usize, Point3<f64>> = UvSphere::new(16, 8)
    ///     .polygons::<Position<Point3<N64>>>()
    ///     .triangulate()
    ///     .collect();
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
    pub fn as_index_slice(&self) -> &[R::Group] {
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
    Vec<R::Group>: IndexBuffer<R>,
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
    Vec<R::Group>: IndexBuffer<R>,
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
    N: Copy + Integer + Unsigned,
{
    type Dynamic = <Flat<A, N> as StaticArity>::Static;

    fn arity(&self) -> Self::Dynamic {
        Flat::<A, N>::ARITY
    }
}

impl<P, G> DynamicArity for MeshBuffer<P, G>
where
    P: Grouping + Monomorphic + Polygonal,
    P::Vertex: Copy + Integer + Unsigned,
{
    type Dynamic = <P as StaticArity>::Static;

    fn arity(&self) -> Self::Dynamic {
        P::ARITY
    }
}

impl<N, G> DynamicArity for MeshBuffer<BoundedPolygon<N>, G>
where
    N: Copy + Integer + Unsigned,
{
    type Dynamic = MeshArity;

    fn arity(&self) -> Self::Dynamic {
        MeshArity::from_components::<BoundedPolygon<N>, _>(self.indices.iter())
    }
}

impl<N, G> DynamicArity for MeshBuffer<UnboundedPolygon<N>, G>
where
    N: Copy + Integer + Unsigned,
{
    type Dynamic = MeshArity;

    fn arity(&self) -> Self::Dynamic {
        MeshArity::from_components::<UnboundedPolygon<N>, _>(self.indices.iter())
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
    ///
    /// # Errors
    ///
    /// Returns an error if an index overflows.
    pub fn append<R, H>(&mut self, buffer: &mut MeshBuffer<R, H>) -> Result<(), BufferError>
    where
        G: FromGeometry<H>,
        R: Grouping,
        R::Group: Into<<Flat<A, N> as Grouping>::Group>,
    {
        let offset = N::from(self.vertices.len()).ok_or_else(|| BufferError::IndexOverflow)?;
        self.vertices.extend(
            buffer
                .vertices
                .drain(..)
                .map(|vertex| vertex.into_geometry()),
        );
        self.indices
            .extend(buffer.indices.drain(..).map(|index| index.into() + offset));
        Ok(())
    }
}

impl<P, G> MeshBuffer<P, G>
where
    P: Grouping + Polygonal,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
{
    /// Appends the contents of a `MeshBuffer` into another `MeshBuffer`. The
    /// source buffer is drained.
    ///
    /// # Errors
    ///
    /// Returns an error if an index overflows.
    pub fn append<R, H>(&mut self, buffer: &mut MeshBuffer<R, H>) -> Result<(), BufferError>
    where
        G: FromGeometry<H>,
        R: Grouping,
        R::Group: Into<<P as Grouping>::Group>,
        <P as Grouping>::Group:
            Map<P::Vertex, Output = <P as Grouping>::Group> + Topological<Vertex = P::Vertex>,
    {
        let offset = <P::Vertex as NumCast>::from(self.vertices.len())
            .ok_or_else(|| BufferError::IndexOverflow)?;
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
        );
        Ok(())
    }
}

impl<P, Q, T, R, N, G> From<P> for MeshBuffer<R, G>
where
    P: IntoIndexed<N, Indexed = Q> + Polygonal,
    Q: Clone + Map<G, Output = T> + Map<N, Output = R> + Polygonal<Vertex = (N, P::Vertex)>,
    T: Polygonal<Vertex = G>,
    R: Grouping<Group = R> + Polygonal<Vertex = N>,
    N: Copy + Integer + NumCast + Unsigned,
    G: FromGeometry<P::Vertex>,
{
    fn from(polygon: P) -> Self {
        let indexed = polygon.into_indexed();
        MeshBuffer::from_raw_buffers(
            // It is tempting to use a range over the polygon's arity to
            // construct `R`, but that weakens the relationship between the
            // input polygon `P` and the index buffer grouping `R`. These types
            // must have compatible type-level arity, so this implementation
            // relies on the `Map` implementation from `P` to `R`.
            Some(indexed.clone().map(|(index, _)| index)),
            Map::<G>::map(indexed, |(_, vertex)| vertex.into_geometry()),
        )
        .expect("inconsistent index buffer")
    }
}

impl<E, P, G> FromEncoding<E> for MeshBuffer<P, G>
where
    E: FaceDecoder<Face = ()> + VertexDecoder,
    E::Index: AsRef<[P::Vertex]>,
    P: Polygonal<Vertex = usize>,
    G: FromGeometry<E::Vertex>,
    Self: FromRawBuffers<P, G, Error = BufferError>,
{
    type Error = <Self as FromRawBuffers<P, G>>::Error;

    fn from_encoding(
        vertices: <E as VertexDecoder>::Output,
        faces: <E as FaceDecoder>::Output,
    ) -> Result<Self, Self::Error> {
        let indices: Vec<_> = faces
            .into_iter()
            .map(|(index, _)| {
                P::try_from_slice(index.as_ref()).ok_or_else(|| BufferError::ArityConflict {
                    expected: P::ARITY.into_interval().0,
                    actual: index.as_ref().len(),
                })
            })
            .collect::<Result<_, _>>()?;
        let vertices = vertices.into_iter().map(|vertex| vertex.into_geometry());
        MeshBuffer::from_raw_buffers(indices, vertices)
    }
}

impl<R, P, G> FromIndexer<P, P> for MeshBuffer<R, G>
where
    R: Grouping,
    G: FromGeometry<P::Vertex>,
    P: Map<IndexOf<R>> + Topological,
    P::Output: Topological<Vertex = IndexOf<R>>,
    BufferOf<R>: Push<R, P::Output>,
    IndexOf<R>: NumCast,
    Self: FromRawBuffers<R::Group, G>,
{
    type Error = <Self as FromRawBuffers<R::Group, G>>::Error;

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
    G: FromGeometry<P::Vertex>,
    P: Topological,
    P::Vertex: Copy + Eq + Hash,
    BufferOf<R>: IndexBuffer<R>,
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
    /// use plexus::buffer::MeshBuffer;
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
    /// let buffer = MeshBuffer::<Flat3, _>::from_raw_buffers(indices, vertices).unwrap();
    /// ```
    fn from_raw_buffers<I, J>(indices: I, vertices: J) -> Result<Self, BufferError>
    where
        I: IntoIterator<Item = M>,
        J: IntoIterator<Item = G>,
    {
        let indices = indices
            .into_iter()
            .map(|index| <N as NumCast>::from(index).ok_or_else(|| BufferError::IndexOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        if indices.len() % A::USIZE != 0 {
            Err(BufferError::IndexUnaligned)
        }
        else {
            let vertices: Vec<_> = vertices.into_iter().collect();
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
    P: From<Q> + Grouping<Group = P> + Polygonal,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
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
        let indices: Vec<_> = indices.into_iter().map(P::from).collect();
        let vertices: Vec<_> = vertices.into_iter().map(|geometry| geometry).collect();
        let is_out_of_bounds = {
            let len = <P::Vertex as NumCast>::from(vertices.len())
                .ok_or_else(|| BufferError::IndexOverflow)?;
            indices
                .iter()
                .any(|polygon| polygon.as_ref().iter().any(|index| *index >= len))
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
    N: Copy + Integer + Unsigned,
{
    type Item = N;

    fn into_flat_index(self) -> MeshBuffer<Flat<A, Self::Item>, G> {
        self
    }
}

impl<N, G> IntoFlatIndex<U3, G> for MeshBuffer<Trigon<N>, G>
where
    N: Copy + Integer + Unsigned,
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
    N: Copy + Integer + Unsigned,
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

impl<N, G> IntoPolygons for MeshBuffer<Flat3<N>, G>
where
    N: Copy + Integer + NumCast + Unsigned,
    G: Clone,
    Trigon<N>: Grouping<Group = Trigon<N>>,
{
    type Output = vec::IntoIter<Self::Polygon>;
    type Polygon = Trigon<G>;

    /// Converts a triangular flat `MeshBuffer` into an iterator of `Trigon`s
    /// containing vertex data.
    fn into_polygons(self) -> Self::Output {
        let (indices, vertices) = self.into_raw_buffers();
        indices
            .into_iter()
            .chunks(U3::USIZE)
            .into_iter()
            .map(|chunk| {
                // These conversions should never fail.
                Trigon::from_items(chunk.map(|index| {
                    let index = <usize as NumCast>::from(index).expect("index overflow");
                    vertices[index].clone()
                }))
                .expect("inconsistent index buffer")
            })
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<N, G> IntoPolygons for MeshBuffer<Flat4<N>, G>
where
    N: Copy + Integer + NumCast + Unsigned,
    G: Clone,
    Trigon<N>: Grouping<Group = Trigon<N>>,
{
    type Output = vec::IntoIter<Self::Polygon>;
    type Polygon = Tetragon<G>;

    /// Converts a quadrilateral flat `MeshBuffer` into an iterator of
    /// `Tetragon`s containing vertex data.
    ///
    /// # Examples
    ///
    /// Mapping over the polygons described by a flat buffer:
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::R64;
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer;
    /// use plexus::graph::MeshGraph;
    /// use plexus::index::Flat4;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    ///
    /// type E3 = Point3<R64>;
    ///
    /// let buffer: MeshBuffer<Flat4, E3> = Cube::new().polygons::<Position<E3>>().collect();
    /// let graph: MeshGraph<E3> = buffer
    ///     .into_polygons()
    ///     .map_vertices(|position| position * 2.0.into())
    ///     .collect();
    /// ```
    fn into_polygons(self) -> Self::Output {
        let (indices, vertices) = self.into_raw_buffers();
        indices
            .into_iter()
            .chunks(U4::USIZE)
            .into_iter()
            .map(|chunk| {
                // These conversions should never fail.
                Tetragon::from_items(chunk.map(|index| {
                    let index = <usize as NumCast>::from(index).expect("index overflow");
                    vertices[index].clone()
                }))
                .expect("inconsistent index buffer")
            })
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<P, G> IntoPolygons for MeshBuffer<P, G>
where
    P: Grouping + Polygonal,
    P::Group: Map<G> + Polygonal,
    <P::Group as Map<G>>::Output: Polygonal<Vertex = G>,
    <P::Group as Topological>::Vertex: NumCast,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
    G: Clone,
{
    type Output = vec::IntoIter<Self::Polygon>;
    type Polygon = <P::Group as Map<G>>::Output;

    /// Converts a structured `MeshBuffer` into an iterator of polygons
    /// containing vertex data.
    ///
    /// # Examples
    ///
    /// Mapping over the polygons described by a structured buffer:
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::R64;
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::generate::Position;
    /// use plexus::primitive::sphere::UvSphere;
    /// use plexus::primitive::BoundedPolygon;
    ///
    /// type E3 = Point3<R64>;
    ///
    /// let buffer: MeshBuffer<BoundedPolygon<usize>, E3> =
    ///     UvSphere::new(8, 8).polygons::<Position<E3>>().collect();
    /// let graph: MeshGraph<E3> = buffer
    ///     .into_polygons()
    ///     .map_vertices(|position| position * 2.0.into())
    ///     .triangulate()
    ///     .collect();
    /// ```
    fn into_polygons(self) -> Self::Output {
        let (indices, vertices) = self.into_raw_buffers();
        indices
            .into_iter()
            .map(|polygon| {
                polygon.map(|index| {
                    // This conversion should never fail.
                    let index = <usize as NumCast>::from(index).expect("index overflow");
                    vertices[index].clone()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<P, G> IntoStructuredIndex<G> for MeshBuffer<P, G>
where
    P: Grouping + Polygonal,
    P::Vertex: Copy + Integer + Unsigned,
{
    type Item = P;

    fn into_structured_index(self) -> MeshBuffer<Self::Item, G> {
        self
    }
}

impl<N, G> IntoStructuredIndex<G> for MeshBuffer<Flat3<N>, G>
where
    N: Copy + Integer + Unsigned,
    Trigon<N>: Grouping<Group = Trigon<N>>,
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
    /// use plexus::buffer::MeshBuffer;
    /// use plexus::index::Flat3;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    ///
    /// let cube = Cube::new();
    /// let buffer = MeshBuffer::<Flat3, _>::from_raw_buffers(
    ///     cube.indexing_polygons::<Position>()
    ///         .triangulate()
    ///         .vertices(),
    ///     cube.vertices::<Position<Point3<f64>>>(),
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
            .map(<Self::Item as Grouping>::Group::from_items)
            .collect::<Option<Vec<_>>>()
            .expect("inconsistent index buffer");
        MeshBuffer { indices, vertices }
    }
}

impl<N, G> IntoStructuredIndex<G> for MeshBuffer<Flat4<N>, G>
where
    N: Copy + Integer + Unsigned,
    Tetragon<N>: Grouping<Group = Tetragon<N>>,
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
    ///     cube.indexing_polygons::<Position>(),
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
            .map(<Self::Item as Grouping>::Group::from_items)
            .collect::<Option<Vec<_>>>()
            .expect("inconsistent index buffer");
        MeshBuffer { indices, vertices }
    }
}

#[cfg(test)]
mod tests {
    use decorum::N64;
    use nalgebra::Point3;

    use crate::buffer::{MeshBuffer, MeshBuffer4, MeshBufferN};
    use crate::graph::MeshGraph;
    use crate::index::Flat3;
    use crate::prelude::*;
    use crate::primitive::cube::Cube;
    use crate::primitive::generate::Position;
    use crate::primitive::sphere::UvSphere;
    use crate::primitive::{BoundedPolygon, UnboundedPolygon};

    type E3 = Point3<N64>;

    #[test]
    fn collect_into_flat_buffer() {
        let buffer: MeshBuffer<Flat3<usize>, E3> = UvSphere::new(3, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .triangulate()
            .collect();

        assert_eq!(18, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
    }

    #[test]
    fn collect_into_bounded_buffer() {
        let buffer: MeshBuffer<BoundedPolygon<usize>, E3> = UvSphere::new(3, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect();

        assert_eq!(6, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
    }

    #[test]
    fn collect_into_unbounded_buffer() {
        let buffer: MeshBuffer<UnboundedPolygon<usize>, E3> =
            Cube::new().polygons::<Position<E3>>().collect();

        assert_eq!(6, buffer.as_index_slice().len());
        assert_eq!(8, buffer.as_vertex_slice().len());
        for polygon in buffer.as_index_slice() {
            assert_eq!(4, polygon.arity());
        }
    }

    #[test]
    fn append_structured_buffers() {
        let mut buffer: MeshBufferN<usize, E3> = UvSphere::new(3, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect();
        buffer
            .append(
                &mut Cube::new()
                    .polygons::<Position<E3>>() // 6 quadrilaterals, 24 vertices.
                    .collect::<MeshBuffer4<usize, E3>>(),
            )
            .unwrap();

        assert_eq!(12, buffer.as_index_slice().len());
        assert_eq!(13, buffer.as_vertex_slice().len());
    }

    #[test]
    fn convert_mesh_to_buffer_by_vertex() {
        let graph: MeshGraph<E3> = UvSphere::new(3, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect();
        let buffer: MeshBufferN<usize, E3> = graph.to_mesh_by_vertex().unwrap();

        assert_eq!(6, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
    }

    #[test]
    fn convert_mesh_to_buffer_by_face() {
        let graph: MeshGraph<E3> = UvSphere::new(3, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect();
        let buffer: MeshBufferN<usize, E3> = graph.to_mesh_by_face().unwrap();

        assert_eq!(6, buffer.as_index_slice().len());
        assert_eq!(18, buffer.as_vertex_slice().len());
    }
}
