//! Linear buffer that can be used for rendering.
//!
//! This module provides a `MeshBuffer` that can be read by graphics pipelines
//! to render meshes. `MeshBuffer` combines an index buffer and vertex buffer,
//! which is exposes as slices.

use num::{Integer, NumCast, Unsigned};
use std::hash::Hash;
use std::iter::FromIterator;

use generate::{FromIndexer, HashIndexer, IndexVertices, Indexer, IntoTriangles, IntoVertices,
               Topological, Triangle, Triangulate};

pub struct MeshBuffer<N, V>
where
    N: Integer + Unsigned,
{
    indeces: Vec<N>,
    vertices: Vec<V>,
}

impl<N, V> MeshBuffer<N, V>
where
    N: Integer + Unsigned,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_raw_buffers<I, J>(indeces: I, vertices: J) -> Self
    where
        I: IntoIterator<Item = N>,
        J: IntoIterator<Item = V>,
    {
        MeshBuffer {
            indeces: indeces.into_iter().collect(),
            vertices: vertices.into_iter().collect(),
        }
    }

    pub fn as_index_slice(&self) -> &[N] {
        self.indeces.as_slice()
    }

    pub fn as_vertex_slice(&self) -> &[V] {
        self.vertices.as_slice()
    }
}

impl<N, V> MeshBuffer<N, V>
where
    N: Copy + Integer + NumCast + Unsigned,
{
    pub fn append<M, U>(&mut self, other: &mut MeshBuffer<M, U>)
    where
        M: Copy + Integer + Into<N> + Unsigned,
        U: Into<V>,
    {
        let offset = N::from(self.vertices.len()).unwrap();
        self.vertices
            .extend(other.vertices.drain(..).map(|vertex| vertex.into()));
        self.indeces
            .extend(other.indeces.drain(..).map(|index| index.into() + offset))
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

impl<N, V, P> FromIndexer<P, Triangle<P::Vertex>> for MeshBuffer<N, V>
where
    P: IntoTriangles + IntoVertices + Topological,
    P::Vertex: Into<V>,
    N: Integer + NumCast + Unsigned,
{
    fn from_indexer<I, M>(input: I, indexer: M) -> Self
    where
        I: IntoIterator<Item = P>,
        M: Indexer<Triangle<P::Vertex>, P::Vertex>,
    {
        let (indeces, vertices) = input.into_iter().triangulate().index_vertices(indexer);
        MeshBuffer::from_raw_buffers(
            indeces.into_iter().map(|index| N::from(index).unwrap()),
            vertices.into_iter().map(|vertex| vertex.into()),
        )
    }
}

impl<N, V, P> FromIterator<P> for MeshBuffer<N, V>
where
    P: IntoTriangles + IntoVertices + Topological,
    P::Vertex: Eq + Hash + Into<V>,
    N: Integer + NumCast + Unsigned,
{
    fn from_iter<I>(input: I) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        // TODO: This is fast and reliable, but the requirements on `P::Vertex`
        //       are difficult to achieve. Would `LruIndexer` be a better
        //       choice?
        Self::from_indexer(input, HashIndexer::default())
    }
}

#[cfg(test)]
mod tests {
    use buffer::*;
    use generate::*;
    use geometry::*;
    use ordered::*;

    #[test]
    fn collect_topology_into_buffer() {
        let buffer = sphere::UVSphere::<f32>::with_unit_radius(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .map_vertices(|vertex| vertex.into_hash())
            .collect::<MeshBuffer<u32, Triplet<_>>>();

        assert_eq!(18, buffer.as_index_slice().len());
        assert_eq!(5, buffer.as_vertex_slice().len());
    }
}
