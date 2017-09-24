//! Half-edge graph representation of meshes.
//!
//! This module provides a flexible representation of meshes as a [half-edge
//! graph](https://en.wikipedia.org/wiki/doubly_connected_edge_list). Meshes
//! can store arbitrary geometric data associated with any topological
//! structure, including vertices, (bilateral) edges, and faces.
//!
//! These structures can be difficult to construct from individual components;
//! the `generate` module can be used to produce primitive meshes that can be
//! converted into a graph.
//!
//! # Examples
//!
//! Creating an empty mesh with no geometric data:
//!
//! ```rust
//! use plexus::graph::Mesh;
//!
//! let mut mesh = Mesh::<()>::new();
//! ```

mod geometry;
mod mesh;
mod storage;
mod topology;

pub use self::geometry::{AsPosition, Attribute, Cross, FromGeometry, FromInteriorGeometry,
                         Geometry, IntoGeometry, IntoInteriorGeometry, Normalize};
pub use self::mesh::Mesh;
pub use self::storage::{EdgeKey, FaceKey, VertexKey};
pub use self::topology::{EdgeKeyTopology, EdgeMut, EdgeRef, FaceKeyTopology, FaceMut, FaceRef,
                         OrphanEdgeMut, OrphanFaceMut, OrphanVertexMut, VertexMut, VertexRef};

pub mod prelude {
    pub use super::{FromGeometry, FromInteriorGeometry, IntoGeometry, IntoInteriorGeometry};
}

pub(crate) trait VecExt<T>
where
    T: Copy,
{
    fn duplet_circuit_windows(&self) -> DupletCircuitWindows<T>;
}

impl<T> VecExt<T> for Vec<T>
where
    T: Copy,
{
    fn duplet_circuit_windows(&self) -> DupletCircuitWindows<T> {
        DupletCircuitWindows::new(self)
    }
}

pub(crate) struct DupletCircuitWindows<'a, T>
where
    T: 'a + Copy,
{
    input: &'a Vec<T>,
    index: usize,
}

impl<'a, T> DupletCircuitWindows<'a, T>
where
    T: 'a + Copy,
{
    fn new(input: &'a Vec<T>) -> Self {
        DupletCircuitWindows {
            input: input,
            index: 0,
        }
    }
}

impl<'a, T> Iterator for DupletCircuitWindows<'a, T>
where
    T: 'a + Copy,
{
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index;
        let n = self.input.len();
        if index >= n {
            None
        }
        else {
            self.index += 1;
            Some((self.input[index], self.input[(index + 1) % n]))
        }
    }
}
