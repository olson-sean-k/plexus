use std::marker::PhantomData;

use crate::graph::edge::{Arc, Edge};
use crate::graph::face::Face;
use crate::graph::geometry::{Geometric, GraphGeometry};
use crate::graph::vertex::Vertex;
use crate::network::storage::{AsStorage, AsStorageMut, Fuse, Storage};

/// A complete core that owns all of its storage.
pub type OwnedCore<G> =
    Core<G, Storage<Vertex<G>>, Storage<Arc<G>>, Storage<Edge<G>>, Storage<Face<G>>>;
/// A complete core with immutable references to all of its storage.
pub type RefCore<'a, G> = Core<
    G,
    &'a Storage<Vertex<G>>,
    &'a Storage<Arc<G>>,
    &'a Storage<Edge<G>>,
    &'a Storage<Face<G>>,
>;

/// Adaptable graph representation that can incorporate arbitrary storage.
///
/// A core may or may not own its storage and may or may not provide storage for
/// all topologies (vertices, arcs, edges, and faces). When a core does not own
/// its storage, it is _ephemeral_. A core that owns storage for all topologies
/// is known as an _owned core_. See the `OwnedCore` type alias.
///
/// Unlike `MeshGraph`, `Core` does not implement the `Consistent` trait.
/// `MeshGraph` contains an owned core, but does not mutate it outside of the
/// mutation API, which maintains consistency.
///
/// A core's fields may be in one of two states: _unbound_ and _bound_. When a
/// field is unbound, its type is `()`. An unbound field has no value and is
/// zero-sized. A bound field has any type other than `()`. These fields should
/// provide storage for their corresponding entity, though this is not enforced
/// directly in `Core`. The `Fuse` trait can be used to transition from `()` to
/// some other type by _fusing_ storage into a `Core`. `Fuse` implementations
/// enforce storage constraints.
///
/// A `Core` with no unbound fields is _complete_.
pub struct Core<G, V = (), A = (), E = (), F = ()>
where
    G: GraphGeometry,
{
    vertices: V,
    arcs: A,
    edges: E,
    faces: F,
    phantom: PhantomData<G>,
}

impl<G> Core<G>
where
    G: GraphGeometry,
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
    G: GraphGeometry,
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
    G: GraphGeometry,
{
    fn as_storage(&self) -> &Storage<Vertex<G>> {
        self.vertices.as_storage()
    }
}

impl<G, V, A, E, F> AsStorage<Arc<G>> for Core<G, V, A, E, F>
where
    A: AsStorage<Arc<G>>,
    G: GraphGeometry,
{
    fn as_storage(&self) -> &Storage<Arc<G>> {
        self.arcs.as_storage()
    }
}

impl<G, V, A, E, F> AsStorage<Edge<G>> for Core<G, V, A, E, F>
where
    E: AsStorage<Edge<G>>,
    G: GraphGeometry,
{
    fn as_storage(&self) -> &Storage<Edge<G>> {
        self.edges.as_storage()
    }
}

impl<G, V, A, E, F> AsStorage<Face<G>> for Core<G, V, A, E, F>
where
    F: AsStorage<Face<G>>,
    G: GraphGeometry,
{
    fn as_storage(&self) -> &Storage<Face<G>> {
        self.faces.as_storage()
    }
}

impl<G, V, A, E, F> AsStorageMut<Vertex<G>> for Core<G, V, A, E, F>
where
    V: AsStorageMut<Vertex<G>>,
    G: GraphGeometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<Vertex<G>> {
        self.vertices.as_storage_mut()
    }
}

impl<G, V, A, E, F> AsStorageMut<Arc<G>> for Core<G, V, A, E, F>
where
    A: AsStorageMut<Arc<G>>,
    G: GraphGeometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<Arc<G>> {
        self.arcs.as_storage_mut()
    }
}

impl<G, V, A, E, F> AsStorageMut<Edge<G>> for Core<G, V, A, E, F>
where
    E: AsStorageMut<Edge<G>>,
    G: GraphGeometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<Edge<G>> {
        self.edges.as_storage_mut()
    }
}

impl<G, V, A, E, F> AsStorageMut<Face<G>> for Core<G, V, A, E, F>
where
    F: AsStorageMut<Face<G>>,
    G: GraphGeometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<Face<G>> {
        self.faces.as_storage_mut()
    }
}

impl<G, V, A, E, F> Fuse<V, Vertex<G>> for Core<G, (), A, E, F>
where
    V: AsStorage<Vertex<G>>,
    G: GraphGeometry,
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
    G: GraphGeometry,
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
    G: GraphGeometry,
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
    G: GraphGeometry,
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

impl<G, V, A, E, F> Geometric for Core<G, V, A, E, F>
where
    G: GraphGeometry,
{
    type Geometry = G;
}

impl<G, V, A, E, F> GraphGeometry for Core<G, V, A, E, F>
where
    G: GraphGeometry,
{
    type Vertex = G::Vertex;
    type Arc = G::Arc;
    type Edge = G::Edge;
    type Face = G::Face;
}
