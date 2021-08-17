use std::marker::PhantomData;

use crate::entity::storage::{AsStorage, AsStorageMut, Fuse, StorageTarget};
use crate::entity::Entity;
use crate::graph::data::{GraphData, Parametric};
use crate::graph::edge::{Arc, Edge};
use crate::graph::face::Face;
use crate::graph::vertex::Vertex;

/// A complete core that owns all of its storage.
pub type OwnedCore<G> = Core<
    G,
    <Vertex<G> as Entity>::Storage,
    <Arc<G> as Entity>::Storage,
    <Edge<G> as Entity>::Storage,
    <Face<G> as Entity>::Storage,
>;

/// Adaptable graph representation that can incorporate arbitrary storage.
///
/// Cores act as a container for storage that comprises a graph and allow
/// storage to be moved (_fused_ and _unfused_) as values or references. A core
/// may or may not own its storage and may or may not provide storage for all
/// entities. When a core does not own its storage, it is _ephemeral_.
///
/// Cores are used by the mutation API to unfuse storage and guard it behind
/// per-entity APIs. Unlike `MeshGraph`, `Core` does not implement the
/// `Consistent` trait.  `MeshGraph` contains a core, but does not mutate it
/// outside of the mutation API, which maintains consistency.
///
/// A core's fields may be _unfused_ and _fused_. When a field is unfused, its
/// type is `()`. An unfused field has no value and is zero-sized. A fused field
/// has any type other than `()`. These fields should provide storage for their
/// corresponding entity. The `Fuse` trait is used to transition from `()` to
/// some other type by _fusing_ storage into a `Core`. `Fuse` implementations
/// enforce storage constraints; it is not possible to fuse values that do not
/// expose storage to yet unfused entities.
///
/// A `Core` with no unfused fields is _complete_.
pub struct Core<G, V = (), A = (), E = (), F = ()>
where
    G: GraphData,
{
    pub(in crate::graph) vertices: V,
    pub(in crate::graph) arcs: A,
    pub(in crate::graph) edges: E,
    pub(in crate::graph) faces: F,
    phantom: PhantomData<fn() -> G>,
}

impl<G> Core<G>
where
    G: GraphData,
{
    pub fn empty() -> Self {
        Core {
            vertices: (),
            arcs: (),
            edges: (),
            faces: (),
            phantom: PhantomData,
        }
    }
}

impl<G, V, A, E, F> Core<G, V, A, E, F>
where
    G: GraphData,
{
    pub fn unfuse(self) -> (V, A, E, F) {
        let Core {
            vertices,
            arcs,
            edges,
            faces,
            ..
        } = self;
        (vertices, arcs, edges, faces)
    }
}

impl<G, V, A, E, F> AsStorage<Vertex<G>> for Core<G, V, A, E, F>
where
    V: AsStorage<Vertex<G>>,
    G: GraphData,
{
    fn as_storage(&self) -> &StorageTarget<Vertex<G>> {
        self.vertices.as_storage()
    }
}

impl<G, V, A, E, F> AsStorage<Arc<G>> for Core<G, V, A, E, F>
where
    A: AsStorage<Arc<G>>,
    G: GraphData,
{
    fn as_storage(&self) -> &StorageTarget<Arc<G>> {
        self.arcs.as_storage()
    }
}

impl<G, V, A, E, F> AsStorage<Edge<G>> for Core<G, V, A, E, F>
where
    E: AsStorage<Edge<G>>,
    G: GraphData,
{
    fn as_storage(&self) -> &StorageTarget<Edge<G>> {
        self.edges.as_storage()
    }
}

impl<G, V, A, E, F> AsStorage<Face<G>> for Core<G, V, A, E, F>
where
    F: AsStorage<Face<G>>,
    G: GraphData,
{
    fn as_storage(&self) -> &StorageTarget<Face<G>> {
        self.faces.as_storage()
    }
}

impl<G, V, A, E, F> AsStorageMut<Vertex<G>> for Core<G, V, A, E, F>
where
    V: AsStorageMut<Vertex<G>>,
    G: GraphData,
{
    fn as_storage_mut(&mut self) -> &mut StorageTarget<Vertex<G>> {
        self.vertices.as_storage_mut()
    }
}

impl<G, V, A, E, F> AsStorageMut<Arc<G>> for Core<G, V, A, E, F>
where
    A: AsStorageMut<Arc<G>>,
    G: GraphData,
{
    fn as_storage_mut(&mut self) -> &mut StorageTarget<Arc<G>> {
        self.arcs.as_storage_mut()
    }
}

impl<G, V, A, E, F> AsStorageMut<Edge<G>> for Core<G, V, A, E, F>
where
    E: AsStorageMut<Edge<G>>,
    G: GraphData,
{
    fn as_storage_mut(&mut self) -> &mut StorageTarget<Edge<G>> {
        self.edges.as_storage_mut()
    }
}

impl<G, V, A, E, F> AsStorageMut<Face<G>> for Core<G, V, A, E, F>
where
    F: AsStorageMut<Face<G>>,
    G: GraphData,
{
    fn as_storage_mut(&mut self) -> &mut StorageTarget<Face<G>> {
        self.faces.as_storage_mut()
    }
}

impl<G, V, A, E, F> Default for Core<G, V, A, E, F>
where
    G: GraphData,
    V: Default,
    A: Default,
    E: Default,
    F: Default,
{
    fn default() -> Self {
        Core {
            vertices: Default::default(),
            arcs: Default::default(),
            edges: Default::default(),
            faces: Default::default(),
            phantom: PhantomData,
        }
    }
}

impl<G, V, A, E, F> Fuse<V, Vertex<G>> for Core<G, (), A, E, F>
where
    V: AsStorage<Vertex<G>>,
    G: GraphData,
{
    type Output = Core<G, V, A, E, F>;

    fn fuse(self, vertices: V) -> Self::Output {
        let Core {
            arcs, edges, faces, ..
        } = self;
        Core {
            vertices,
            arcs,
            edges,
            faces,
            phantom: PhantomData,
        }
    }
}

impl<G, V, A, E, F> Fuse<A, Arc<G>> for Core<G, V, (), E, F>
where
    A: AsStorage<Arc<G>>,
    G: GraphData,
{
    type Output = Core<G, V, A, E, F>;

    fn fuse(self, arcs: A) -> Self::Output {
        let Core {
            vertices,
            edges,
            faces,
            ..
        } = self;
        Core {
            vertices,
            arcs,
            edges,
            faces,
            phantom: PhantomData,
        }
    }
}

impl<G, V, A, E, F> Fuse<E, Edge<G>> for Core<G, V, A, (), F>
where
    E: AsStorage<Edge<G>>,
    G: GraphData,
{
    type Output = Core<G, V, A, E, F>;

    fn fuse(self, edges: E) -> Self::Output {
        let Core {
            vertices,
            arcs,
            faces,
            ..
        } = self;
        Core {
            vertices,
            arcs,
            edges,
            faces,
            phantom: PhantomData,
        }
    }
}

impl<G, V, A, E, F> Fuse<F, Face<G>> for Core<G, V, A, E, ()>
where
    F: AsStorage<Face<G>>,
    G: GraphData,
{
    type Output = Core<G, V, A, E, F>;

    fn fuse(self, faces: F) -> Self::Output {
        let Core {
            vertices,
            arcs,
            edges,
            ..
        } = self;
        Core {
            vertices,
            arcs,
            edges,
            faces,
            phantom: PhantomData,
        }
    }
}

impl<G, V, A, E, F> Parametric for Core<G, V, A, E, F>
where
    G: GraphData,
{
    type Data = G;
}
