use crate::geometry::Geometry;
use crate::graph::payload::{ArcPayload, EdgePayload, FacePayload, Payload, VertexPayload};
use crate::graph::storage::convert::{AsStorage, AsStorageMut};
use crate::graph::storage::Storage;

/// A complete core that owns all of its storage.
pub type OwnedCore<G> = Core<
    Storage<VertexPayload<G>>,
    Storage<ArcPayload<G>>,
    Storage<EdgePayload<G>>,
    Storage<FacePayload<G>>,
>;
/// A complete core with immutable references to all of its storage.
pub type RefCore<'a, G> = Core<
    &'a Storage<VertexPayload<G>>,
    &'a Storage<ArcPayload<G>>,
    &'a Storage<EdgePayload<G>>,
    &'a Storage<FacePayload<G>>,
>;

pub trait Bind<T, M>
where
    T: Payload,
    M: AsStorage<T>,
{
    type Output;

    fn bind(self, source: M) -> Self::Output;
}

/// Adaptable graph representation that can incorporate arbitrary storage.
///
/// A core may or may not own its storage and may or may not provide storage
/// for all topologies (vertices, arcs, edges, and faces). When a core does not
/// own its storage, it is _ephemeral_. A core that owns storage for all
/// topologies is known as an _owned core_. See the `OwnedCore` type alias.
///
/// Unlike `MeshGraph`, `Core` does not implement the `Consistent` trait.
/// `MeshGraph` contains an owned core, but does not mutate it outside of the
/// mutation API, which maintains consistency.
///
/// A core's fields may be in one of two states: _unbound_ and _bound_. When a
/// field is unbound, its type is `()`. An unbound field has no value and is
/// zero-sized. A bound field has any type other than `()`. These fields should
/// provide storage for their corresponding topology, though this is not
/// enforced directly in `Core`. The `Bind` trait can be used to transition
/// from `()` to some other type. `Bind` implementations enforce storage
/// constraints.
///
/// A `Core` with no unbound fields is _complete_.
pub struct Core<V = (), A = (), E = (), F = ()> {
    vertices: V,
    arcs: A,
    edges: E,
    faces: F,
}

impl Core {
    pub fn empty() -> Self {
        Core {
            vertices: (),
            arcs: (),
            edges: (),
            faces: (),
        }
    }
}

impl<V, A, E, F> Core<V, A, E, F> {
    pub fn into_storage(self) -> (V, A, E, F) {
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

impl<V, A, E, F, G> AsStorage<VertexPayload<G>> for Core<V, A, E, F>
where
    V: AsStorage<VertexPayload<G>>,
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<VertexPayload<G>> {
        self.vertices.as_storage()
    }
}

impl<V, A, E, F, G> AsStorage<ArcPayload<G>> for Core<V, A, E, F>
where
    A: AsStorage<ArcPayload<G>>,
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<ArcPayload<G>> {
        self.arcs.as_storage()
    }
}

impl<V, A, E, F, G> AsStorage<EdgePayload<G>> for Core<V, A, E, F>
where
    E: AsStorage<EdgePayload<G>>,
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<EdgePayload<G>> {
        self.edges.as_storage()
    }
}

impl<V, A, E, F, G> AsStorage<FacePayload<G>> for Core<V, A, E, F>
where
    F: AsStorage<FacePayload<G>>,
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<FacePayload<G>> {
        self.faces.as_storage()
    }
}

impl<V, A, E, F, G> AsStorageMut<VertexPayload<G>> for Core<V, A, E, F>
where
    V: AsStorageMut<VertexPayload<G>>,
    G: Geometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<VertexPayload<G>> {
        self.vertices.as_storage_mut()
    }
}

impl<V, A, E, F, G> AsStorageMut<ArcPayload<G>> for Core<V, A, E, F>
where
    A: AsStorageMut<ArcPayload<G>>,
    G: Geometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<ArcPayload<G>> {
        self.arcs.as_storage_mut()
    }
}

impl<V, A, E, F, G> AsStorageMut<EdgePayload<G>> for Core<V, A, E, F>
where
    E: AsStorageMut<EdgePayload<G>>,
    G: Geometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<EdgePayload<G>> {
        self.edges.as_storage_mut()
    }
}

impl<V, A, E, F, G> AsStorageMut<FacePayload<G>> for Core<V, A, E, F>
where
    F: AsStorageMut<FacePayload<G>>,
    G: Geometry,
{
    fn as_storage_mut(&mut self) -> &mut Storage<FacePayload<G>> {
        self.faces.as_storage_mut()
    }
}

impl<V, A, E, F, G> Bind<VertexPayload<G>, V> for Core<(), A, E, F>
where
    V: AsStorage<VertexPayload<G>>,
    G: Geometry,
{
    type Output = Core<V, A, E, F>;

    fn bind(self, vertices: V) -> Self::Output {
        let Core {
            arcs, edges, faces, ..
        } = self;
        Core {
            vertices,
            arcs,
            edges,
            faces,
        }
    }
}

impl<V, A, E, F, G> Bind<ArcPayload<G>, A> for Core<V, (), E, F>
where
    A: AsStorage<ArcPayload<G>>,
    G: Geometry,
{
    type Output = Core<V, A, E, F>;

    fn bind(self, arcs: A) -> Self::Output {
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
        }
    }
}

impl<V, A, E, F, G> Bind<EdgePayload<G>, E> for Core<V, A, (), F>
where
    E: AsStorage<EdgePayload<G>>,
    G: Geometry,
{
    type Output = Core<V, A, E, F>;

    fn bind(self, edges: E) -> Self::Output {
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
        }
    }
}

impl<V, A, E, F, G> Bind<FacePayload<G>, F> for Core<V, A, E, ()>
where
    F: AsStorage<FacePayload<G>>,
    G: Geometry,
{
    type Output = Core<V, A, E, F>;

    fn bind(self, faces: F) -> Self::Output {
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
        }
    }
}
