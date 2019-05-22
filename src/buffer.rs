//! Linear representation of meshes.
//!
//! This module provides a `MeshBuffer` that represents a mesh as a linear
//! collection of vertex geometry and a linear collection of indeces into that
//! vertex geometry (topology). These two buffers are called the _vertex
//! buffer_ and _index buffer_, respectively. This layout is well-suited for
//! graphics pipelines. `MeshBuffer` combines an index buffer and vertex
//! buffer, which are exposed as slices.
//!
//! # Vertex Buffers
//!
//! Vertex buffers describe the geometry of a `MeshBuffer`. Only vertex
//! geometry is supported; there is no way to associate geometry with an edge
//! or face, for example.
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
//! Index buffers describe the topology of a `MeshBuffer`. Both _structured_
//! (_polygonal_) and _unstructured_ (_flat_) index buffers are supported. See
//! the `index` module.
//!
//! Flat index buffers directly store individual indices and tend to be more
//! useful for rendering, especially triangular buffers (see the `MeshBuffer3`
//! type definition).
//!
//! Structured index buffers contain sub-structures that in turn contain
//! indices. This is more flexible, but may complicate the consumption of a
//! `MeshBuffer` (by a graphics pipeline, for example).
//!
//! The `MeshBuffer3` and `MeshBufferN` type definitions avoid verbose type
//! parameters and provide the most common index buffer configurations.
//!
//! # Examples
//!
//! Generating a flat `MeshBuffer` from a primitive:
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
//! use plexus::primitive::sphere::UvSphere;
//!
//! # fn main() {
//! let buffer = UvSphere::new(16, 16)
//!     .polygons_with_position::<Point3<N32>>()
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
//! # extern crate decorum;
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use decorum::N64;
//! use nalgebra::Point3;
//! use plexus::graph::MeshGraph;
//! use plexus::prelude::*;
//! use plexus::primitive::cube::Cube;
//! use plexus::U4;
//!
//! # fn main() {
//! let graph = Cube::new()
//!     .polygons_with_position::<Point3<N64>>()
//!     .collect::<MeshGraph<Point3<N64>>>();
//! let buffer = graph
//!     .to_mesh_buffer_by_vertex::<U4, usize, Point3<N64>>()
//!     .unwrap();
//! # }
//! ```

use itertools::Itertools;
use num::{Integer, NumCast, ToPrimitive, Unsigned};
use std::hash::Hash;
use std::iter::FromIterator;
use theon::ops::Map;
use typenum::{self, NonZero, Unsigned as _, U3, U4};

use crate::index::{
    ClosedIndexVertices, Flat, Flat3, Flat4, FromIndexer, Grouping, HashIndexer, IndexBuffer,
    Indexer, Push, Structured, Structured3, Structured4, StructuredN,
};
use crate::primitive::decompose::IntoVertices;
use crate::primitive::{Polygonal, Quad, Topological, Triangle};
use crate::{Arity, FromRawBuffers, IntoGeometry};

#[derive(Debug, Fail)]
pub enum BufferError {
    #[fail(display = "index into vertex data out of bounds")]
    IndexOutOfBounds,
    #[fail(display = "conflicting arity")]
    ArityConflict,
}

/// Alias for a flat and triangular `MeshBuffer`. Prefer this alias.
///
/// For most applications, this alias can be used to avoid more complex and
/// verbose type parameters. Flat and triangular index buffers are most common
/// and should generally be preferred.
pub type MeshBuffer3<N, G> = MeshBuffer<Flat3<N>, G>;
/// Alias for a flat and quadrilateral `MeshBuffer`.
pub type MeshBuffer4<N, G> = MeshBuffer<Flat4<N>, G>;

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
    /// Creates an empty `MeshBuffer`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use plexus::buffer::MeshBuffer;
    /// use plexus::index::Flat3;
    ///
    /// let buffer = MeshBuffer::<Flat3<u32>, (f64, f64, f64)>::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    pub fn arity(&self) -> Arity {
        self.indices.arity()
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
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// # fn main() {
    /// let buffer = UvSphere::new(16, 8)
    ///     .polygons_with_position::<Point3<N64>>()
    ///     .triangulate()
    ///     .collect::<MeshBuffer3<usize, Point3<f64>>>();
    /// // Translate the positions.
    /// let translation = Vector3::<f64>::x() * 2.0;
    /// let buffer = buffer.map_vertices_into(|position| position + translation);
    /// # }
    /// ```
    pub fn map_vertices_into<H, F>(self, f: F) -> MeshBuffer<R, H>
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

impl<P, G> MeshBuffer<Structured<P>, G>
where
    P: Polygonal,
    P::Vertex: Copy + Integer + NumCast + Unsigned,
    Structured<P>: Grouping,
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
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// # fn main() {
    /// let buffer = UvSphere::new(8, 8)
    ///     .polygons_with_position::<Point3<R64>>()
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
    ) -> impl Clone + Iterator<Item = <<Structured<P> as Grouping>::Item as Map<G>>::Output>
    where
        G: Clone,
        <Structured<P> as Grouping>::Item: Map<G> + Topological,
        <<Structured<P> as Grouping>::Item as Map<G>>::Output: Clone,
        <<Structured<P> as Grouping>::Item as Topological>::Vertex: ToPrimitive,
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
        R::Item: Into<<Structured<P> as Grouping>::Item>,
        H: IntoGeometry<G>,
        <Structured<P> as Grouping>::Item:
            Copy
                + Map<P::Vertex, Output = <Structured<P> as Grouping>::Item>
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
    /// buffer or if the number of indices disagrees with the arity of the
    /// index buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::N64;
    /// use nalgebra::{Point2, Point3};
    /// use plexus::buffer::MeshBuffer3;
    /// use plexus::index::{Flat3, HashIndexer};
    /// use plexus::prelude::*;
    /// use plexus::primitive;
    /// use plexus::primitive::cube::Cube;
    ///
    /// # fn main() {
    /// type E2 = Point2<N64>;
    /// type E3 = Point3<N64>;
    ///
    /// let cube = Cube::new();
    /// let (indices, vertices) = primitive::zip_vertices((
    ///     cube.polygons_with_position::<E3>(),
    ///     cube.polygons_with_normal::<E3>(),
    ///     cube.polygons_with_uv_map::<E2>(),
    /// ))
    /// .triangulate()
    /// .index_vertices::<Flat3, _>(HashIndexer::default());
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
            .map(|index| <<Flat<A, N> as Grouping>::Item as NumCast>::from(index).unwrap())
            .collect::<Vec<_>>();
        if indices.len() % A::USIZE != 0 {
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
    Q: Into<<Structured<P> as Grouping>::Item>,
    Structured<P>: Grouping,
    <Structured<P> as Grouping>::Item: Copy + IntoVertices + Topological<Vertex = P::Vertex>,
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
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// # fn main() {
    /// let sphere = UvSphere::new(8, 8);
    /// let buffer = MeshBufferN::<usize, _>::from_raw_buffers(
    ///     sphere.indices_for_position(),
    ///     sphere.vertices_with_position::<Point3<f64>>(),
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
    /// #
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer;
    /// use plexus::index::Structured3;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    ///
    /// # fn main() {
    /// let cube = Cube::new();
    /// let buffer = MeshBuffer::<Structured3, _>::from_raw_buffers(
    ///     cube.indices_for_position().triangulate(),
    ///     cube.vertices_with_position::<Point3<f32>>(),
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
    /// #
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer;
    /// use plexus::index::Structured4;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    ///
    /// # fn main() {
    /// let cube = Cube::new();
    /// let buffer = MeshBuffer::<Structured4, _>::from_raw_buffers(
    ///     cube.indices_for_position(),
    ///     cube.vertices_with_position::<Point3<f64>>(),
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
    /// #
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer3;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    ///
    /// # fn main() {
    /// let cube = Cube::new();
    /// let buffer = MeshBuffer3::<usize, _>::from_raw_buffers(
    ///     cube.indices_for_position().triangulate().vertices(),
    ///     cube.vertices_with_position::<Point3<f32>>(),
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
                .chunks(U3::USIZE)
                .into_iter()
                .map(|triangle| <Structured<Self::Item> as Grouping>::Item::from_iter(triangle))
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
    /// #
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer4;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    ///
    /// # fn main() {
    /// let cube = Cube::new();
    /// let buffer = MeshBuffer4::<usize, _>::from_raw_buffers(
    ///     cube.indices_for_position().vertices(),
    ///     cube.vertices_with_position::<Point3<f64>>(),
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
                .chunks(U4::USIZE)
                .into_iter()
                .map(|quad| <Structured<Self::Item> as Grouping>::Item::from_iter(quad))
                .collect(),
            vertices,
        }
    }
}

#[cfg(test)]
mod tests {
    use decorum::N64;
    use nalgebra::Point3;
    use typenum::U3;

    use crate::buffer::{MeshBuffer, MeshBuffer3, MeshBufferN};
    use crate::graph::MeshGraph;
    use crate::index::{Structured4, StructuredN};
    use crate::prelude::*;
    use crate::primitive::cube::Cube;
    use crate::primitive::sphere::UvSphere;

    type E3 = Point3<N64>;

    #[test]
    fn collect_topology_into_flat_buffer() {
        let buffer = UvSphere::new(3, 2)
            .polygons_with_position::<E3>() // 6 triangles, 18 vertices.
            .triangulate()
            .collect::<MeshBuffer3<u32, Point3<f64>>>();

        assert_eq!(18, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
    }

    #[test]
    fn collect_topology_into_structured_buffer() {
        let buffer = UvSphere::new(3, 2)
            .polygons_with_position::<E3>() // 6 triangles, 18 vertices.
            .collect::<MeshBufferN<u32, Point3<f64>>>();

        assert_eq!(6, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
    }

    #[test]
    fn append_structured_buffers() {
        let mut buffer = UvSphere::new(3, 2)
            .polygons_with_position::<E3>() // 6 triangles, 18 vertices.
            .collect::<MeshBuffer<StructuredN<u32>, Point3<f64>>>();
        buffer.append(
            &mut Cube::new()
                .polygons_with_position::<E3>() // 6 quads, 24 vertices.
                .collect::<MeshBuffer<Structured4<u32>, Point3<f64>>>(),
        );

        assert_eq!(12, buffer.as_index_slice().len());
        assert_eq!(13, buffer.as_vertex_slice().len());
    }

    #[test]
    fn convert_mesh_to_buffer_by_vertex() {
        let graph = UvSphere::new(3, 2)
            .polygons_with_position::<E3>() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f64>>>();
        let buffer = graph
            .to_mesh_buffer_by_vertex::<U3, u32, Point3<f64>>()
            .unwrap();

        assert_eq!(18, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
    }

    #[test]
    fn convert_mesh_to_buffer_by_face() {
        let graph = UvSphere::new(3, 2)
            .polygons_with_position::<E3>() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f64>>>();
        let buffer = graph
            .to_mesh_buffer_by_face::<U3, u32, Point3<f64>>()
            .unwrap();

        assert_eq!(18, buffer.as_index_slice().len());
        assert_eq!(18, buffer.as_vertex_slice().len());
    }
}
