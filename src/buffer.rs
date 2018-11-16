//! Linear buffers that can be used for rendering.
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
//! and _unstructured_ (_flat_) index buffers are supported. See `Flat` and
//! `Structured`.
//!
//! Structured index buffers store indices as `Polygon`s, which preserve the
//! topology of a mesh even if its arity is non-constant. Only triangles and
//! quads are supported.
//!
//! Flat index buffers store individual indices. Because there is no structure,
//! arity must by constant, but arbitrary N-gons are supported. Flat buffers
//! tend to be more useful for rendering, especially triangular flat buffers.
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
//! use plexus::primitive::sphere::UvSphere;
//! use plexus::prelude::*;
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
//! let buffer = graph.to_mesh_buffer_by_vertex::<U4, u32, Point3<f32>>().unwrap();
//! # }
//! ```

use num::{Integer, NumCast, ToPrimitive, Unsigned};
use std::hash::Hash;
use std::iter::FromIterator;
use std::marker::PhantomData;
use typenum::{self, NonZero};

use geometry::convert::IntoGeometry;
use primitive::decompose::IntoVertices;
use primitive::index::{FlatIndexVertices, FromIndexer, HashIndexer, IndexVertices, Indexer};
use primitive::{Arity, Map, Polygon, Polygonal};

pub use typenum::{U3, U4};

#[derive(Debug, Fail)]
pub enum BufferError {
    #[fail(display = "index into vertex data out of bounds")]
    IndexOutOfBounds,
}

/// Index buffer.
pub trait IndexBuffer {
    type Unit;

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
    type Unit = N;

    const ARITY: Option<usize> = Some(A::USIZE);
}

/// Alias for a flat and triangular index buffer.
pub type Flat3<N = usize> = Flat<U3, N>;
/// Alias for a flat and quadrilateral index buffer.
pub type Flat4<N = usize> = Flat<U4, N>;

/// Alias for a flat and triangular `MeshBuffer`.
///
/// For most applications, this alias can be used to avoid more complex and
/// verbose type parameters, because flat and triangular buffers are most
/// common.
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
///
/// let mut buffer = MeshBuffer::<Structured<usize>, Triplet<f64>>::default();
/// ```
#[derive(Debug)]
pub struct Structured<N = usize>
where
    N: Copy + Integer + NumCast + Unsigned,
{
    phantom: PhantomData<N>,
}

impl<N> IndexBuffer for Structured<N>
where
    N: Copy + Integer + NumCast + Unsigned,
{
    type Unit = Polygon<N>;

    const ARITY: Option<usize> = None;
}

/// Alias for a structured `MeshBuffer`.
pub type MeshBufferN<N, G> = MeshBuffer<Structured<N>, G>;

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
    indices: Vec<I::Unit>,
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
    /// ```
    /// use plexus::buffer::{Flat3, MeshBuffer};
    /// use plexus::geometry::Triplet;
    ///
    /// let buffer = MeshBuffer::<Flat3<u32>, Triplet<f64>>::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    pub fn arity(&self) -> Option<usize> {
        I::ARITY
    }

    /// Gets a slice of the index data.
    pub fn as_index_slice(&self) -> &[I::Unit] {
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
    /// Creates a flat `MeshBuffer` from raw index and vertex buffers.
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
    /// use plexus::buffer::MeshBuffer3;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::prelude::*;
    ///
    /// # fn main() {
    /// let cube = Cube::new();
    /// let indices = cube
    ///     .polygons_with_index()
    ///     .triangulate()
    ///     .vertices()
    ///     .collect::<Vec<_>>();
    /// let vertices = cube
    ///     .vertices_with_position()
    ///     .map(|position| -> Point3<f32> { position.into() })
    ///     .collect::<Vec<_>>();
    /// let buffer = MeshBuffer3::<usize, _>::from_raw_buffers(indices, vertices).unwrap();
    /// # }
    /// ```
    pub fn from_raw_buffers<I, J>(indices: I, vertices: J) -> Result<Self, BufferError>
    where
        I: IntoIterator,
        I::Item: ToPrimitive,
        J: IntoIterator<Item = G>,
    {
        let indices = indices
            .into_iter()
            .map(|index| <<Flat<A, N> as IndexBuffer>::Unit as NumCast>::from(index).unwrap())
            .collect::<Vec<_>>();
        let vertices = vertices
            .into_iter()
            .map(|vertex| vertex)
            .collect::<Vec<_>>();
        let len = N::from(vertices.len()).unwrap();
        if indices.iter().any(|index| *index >= len) {
            Err(BufferError::IndexOutOfBounds)
        }
        else {
            Ok(MeshBuffer { indices, vertices })
        }
    }

    /// Appends the contents of a `MeshBuffer` into another `MeshBuffer`. The
    /// source buffer is drained.
    pub fn append<U, H>(&mut self, buffer: &mut MeshBuffer<U, H>)
    where
        U: IndexBuffer,
        U::Unit: Into<<Flat<A, N> as IndexBuffer>::Unit>,
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

impl<N, G> MeshBuffer<Structured<N>, G>
where
    N: Copy + Integer + NumCast + Unsigned,
{
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
    /// use plexus::primitive::sphere::UvSphere;
    /// use plexus::prelude::*;
    ///
    /// # fn main() {
    /// let sphere = UvSphere::new(8, 8);
    /// let indices = sphere
    ///     .polygons_with_index()
    ///     .collect::<Vec<_>>();
    /// let vertices = sphere
    ///     .vertices_with_position()
    ///     .map(|position| -> Point3<f32> { position.into() })
    ///     .collect::<Vec<_>>();
    /// let buffer = MeshBufferN::<usize, _>::from_raw_buffers(indices, vertices).unwrap();
    /// # }
    /// ```
    pub fn from_raw_buffers<I, J>(indices: I, vertices: J) -> Result<Self, BufferError>
    where
        I: IntoIterator,
        I::Item: Into<<Structured<N> as IndexBuffer>::Unit>,
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
        let len = N::from(vertices.len()).unwrap();
        if indices
            .iter()
            .any(|polygon| polygon.iter().any(|index| *index >= len))
        {
            Err(BufferError::IndexOutOfBounds)
        }
        else {
            Ok(MeshBuffer { indices, vertices })
        }
    }

    /// Appends the contents of a `MeshBuffer` into another `MeshBuffer`. The
    /// source buffer is drained.
    pub fn append<U, H>(&mut self, buffer: &mut MeshBuffer<U, H>)
    where
        U: IndexBuffer,
        U::Unit: Into<<Structured<N> as IndexBuffer>::Unit>,
        H: IntoGeometry<G>,
    {
        let offset = N::from(self.vertices.len()).unwrap();
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

impl<A, N, G, P> FromIndexer<P, P> for MeshBuffer<Flat<A, N>, G>
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

impl<N, G, P> FromIndexer<P, P> for MeshBuffer<Structured<N>, G>
where
    N: Copy + Integer + NumCast + Unsigned,
    P: Map<usize> + Polygonal,
    P::Output: Map<N>,
    P::Vertex: IntoGeometry<G>,
    <Structured<N> as IndexBuffer>::Unit: Clone + Copy + From<<P::Output as Map<N>>::Output>,
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
                .map(|topology| topology.map(|index| N::from(index).unwrap())),
            vertices.into_iter().map(|vertex| vertex.into_geometry()),
        )
    }
}

impl<A, N, G, P> FromIterator<P> for MeshBuffer<Flat<A, N>, G>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Integer + NumCast + Unsigned,
    P: Arity + IntoVertices + Polygonal,
    P::Vertex: Clone + Eq + Hash + IntoGeometry<G>,
{
    fn from_iter<I>(input: I) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        Self::from_indexer(input, HashIndexer::default()).unwrap_or_else(|_| Self::default())
    }
}

impl<N, G, P> FromIterator<P> for MeshBuffer<Structured<N>, G>
where
    N: Copy + Integer + NumCast + Unsigned,
    P: Map<usize> + Polygonal,
    P::Output: Map<N>,
    P::Vertex: Clone + Eq + Hash + IntoGeometry<G>,
    <Structured<N> as IndexBuffer>::Unit: Clone + Copy + From<<P::Output as Map<N>>::Output>,
{
    fn from_iter<I>(input: I) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        Self::from_indexer(input, HashIndexer::default()).unwrap_or_else(|_| Self::default())
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use buffer::*;
    use graph::*;
    use primitive::decompose::*;
    use primitive::generate::*;
    use primitive::sphere::UvSphere;

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
            .triangulate()
            .collect::<MeshBuffer<Structured<u32>, Point3<f32>>>();

        assert_eq!(6, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
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
