//! Linear buffers that can be used for rendering.
//!
//! This module provides a `MeshBuffer` that can be read by graphics pipelines
//! to render meshes. `MeshBuffer` combines an index buffer and vertex buffer
//! (containing arbitrary data), which are exposed as slices.
//!
//! # Examples
//!
//! Generating a `MeshBuffer` from a primitive:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point3;
//! use plexus::buffer::MeshBuffer;
//! use plexus::primitive::sphere::UvSphere;
//! use plexus::prelude::*;
//!
//! # fn main() {
//! let buffer = UvSphere::new(16, 16)
//!     .polygons_with_position()
//!     .triangulate()
//!     .collect::<MeshBuffer<u32, Point3<f32>>>();
//! let indeces = buffer.as_index_slice();
//! let positions = buffer.as_vertex_slice();
//! # }
//! ```
//!
//! Converting a `Mesh` to a `MeshBuffer`:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point3;
//! use plexus::buffer::MeshBuffer;
//! use plexus::primitive::cube::Cube;
//! use plexus::graph::Mesh;
//! use plexus::prelude::*;
//!
//! # fn main() {
//! let mesh = Cube::new()
//!     .polygons_with_position()
//!     .collect::<Mesh<Point3<f32>>>();
//! let buffer = mesh.to_mesh_buffer_by_vertex::<u32, Point3<f32>>().unwrap();
//! # }
//! ```

use failure::Error;
use num::{Integer, NumCast, Unsigned};
use std::hash::Hash;
use std::iter::FromIterator;

use geometry::convert::IntoGeometry;
use primitive::{
    Arity, FromIndexer, HashIndexer, IndexVertices, Indexer, IntoVertices, MapVerticesInto,
};

#[derive(Debug, Fail)]
pub enum BufferError {
    #[fail(display = "index into vertex data out of bounds")]
    IndexOutOfBounds,
}

/// Linear buffer of mesh data.
///
/// A `MeshBuffer` is a flattened representation of a mesh that can be consumed
/// by a rendering pipeline. A `MeshBuffer` is composed of two separate
/// buffers: an index buffer and a vertex buffer. The index buffer contains
/// ordered indeces into the data in the vertex buffer and describes the
/// topology or polygons forming the mesh. The vertex buffer contains arbitrary
/// geometric data that is typically read by a rendering pipeline.
pub struct MeshBuffer<N, V>
where
    N: Integer + Unsigned,
{
    indeces: Vec<N>,
    vertices: Vec<V>,
}

impl<N, V> MeshBuffer<N, V>
where
    N: Copy + Integer + NumCast + Unsigned,
{
    /// Creates an empty `MeshBuffer`.
    ///
    /// # Examples
    ///
    /// ```
    /// use plexus::buffer::MeshBuffer;
    /// use plexus::geometry::Triplet;
    ///
    /// let buffer = MeshBuffer::<u32, Triplet<f64>>::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a `MeshBuffer` from raw index and vertex buffers.
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
    /// use plexus::buffer::MeshBuffer;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::prelude::*;
    ///
    /// # fn main() {
    /// let cube = Cube::new();
    /// let indeces = cube
    ///     .polygons_with_index()
    ///     .triangulate()
    ///     .vertices()
    ///     .collect::<Vec<_>>();
    /// let vertices = cube
    ///     .vertices_with_position()
    ///     .map(|position| -> Point3<f32> { position.into() })
    ///     .collect::<Vec<_>>();
    /// let buffer = MeshBuffer::from_raw_buffers(indeces, vertices).unwrap();
    /// # }
    /// ```
    pub fn from_raw_buffers<I, J>(indeces: I, vertices: J) -> Result<Self, Error>
    where
        I: IntoIterator<Item = N>,
        J: IntoIterator<Item = V>,
    {
        let indeces = indeces.into_iter().collect::<Vec<_>>();
        let vertices = vertices.into_iter().collect::<Vec<_>>();
        let len = N::from(vertices.len()).unwrap();
        if indeces.iter().any(|index| *index >= len) {
            Err(BufferError::IndexOutOfBounds.into())
        }
        else {
            Ok(MeshBuffer { indeces, vertices })
        }
    }

    /// Appends the contents of a `MeshBuffer` into another `MeshBuffer`. The
    /// source buffer is drained.
    pub fn append<M, U>(&mut self, other: &mut MeshBuffer<M, U>)
    where
        M: Copy + Integer + Into<N> + Unsigned,
        U: IntoGeometry<V>,
    {
        let offset = N::from(self.vertices.len()).unwrap();
        self.vertices.extend(
            other
                .vertices
                .drain(..)
                .map(|vertex| vertex.into_geometry()),
        );
        self.indeces
            .extend(other.indeces.drain(..).map(|index| index.into() + offset))
    }

    /// Gets a slice of the index data.
    pub fn as_index_slice(&self) -> &[N] {
        self.indeces.as_slice()
    }

    /// Gets a slice of the vertex data.
    pub fn as_vertex_slice(&self) -> &[V] {
        self.vertices.as_slice()
    }
}

impl<N, V> Default for MeshBuffer<N, V>
where
    N: Integer + Unsigned,
{
    fn default() -> Self {
        MeshBuffer {
            indeces: vec![],
            vertices: vec![],
        }
    }
}

impl<N, V, P> FromIndexer<P, P> for MeshBuffer<N, V>
where
    P: Arity + MapVerticesInto<usize>,
    P::Output: IntoVertices,
    P::Vertex: IntoGeometry<V>,
    N: Copy + Integer + NumCast + Unsigned,
{
    fn from_indexer<I, M>(input: I, indexer: M) -> Self
    where
        I: IntoIterator<Item = P>,
        M: Indexer<P, P::Vertex>,
    {
        let (indeces, vertices) = input.into_iter().index_vertices(indexer);
        MeshBuffer::from_raw_buffers(
            indeces
                .into_iter()
                .flat_map(|topology| topology.into_vertices())
                .map(|index| N::from(index).unwrap()),
            vertices.into_iter().map(|vertex| vertex.into_geometry()),
        ).unwrap()
    }
}

impl<N, V, P> FromIterator<P> for MeshBuffer<N, V>
where
    P: Arity + MapVerticesInto<usize>,
    P::Output: IntoVertices,
    P::Vertex: Eq + Hash + IntoGeometry<V>,
    N: Copy + Integer + NumCast + Unsigned,
{
    fn from_iter<I>(input: I) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        Self::from_indexer(input, HashIndexer::default())
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use buffer::*;
    use graph::*;
    use primitive::*;

    #[test]
    fn collect_topology_into_buffer() {
        let buffer = sphere::UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .triangulate()
            .collect::<MeshBuffer<u32, Point3<f32>>>();

        assert_eq!(18, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
    }

    #[test]
    fn convert_mesh_to_buffer_by_vertex() {
        let mesh = sphere::UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();
        let buffer = mesh.to_mesh_buffer_by_vertex::<u32, Point3<f32>>().unwrap();

        assert_eq!(18, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
    }

    #[test]
    fn convert_mesh_to_buffer_by_face() {
        let mesh = sphere::UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();
        let buffer = mesh.to_mesh_buffer_by_face::<u32, Point3<f32>>().unwrap();

        assert_eq!(18, buffer.as_index_slice().len());
        assert_eq!(18, buffer.as_vertex_slice().len());
    }
}
