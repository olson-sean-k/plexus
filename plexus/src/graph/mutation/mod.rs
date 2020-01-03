pub mod edge;
pub mod face;
pub mod vertex;

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use crate::graph::core::OwnedCore;
use crate::graph::geometry::GraphGeometry;
use crate::graph::mutation::face::FaceMutation;
use crate::graph::storage::alias::*;
use crate::graph::storage::payload::{Arc, Face, Vertex};
use crate::graph::storage::{AsStorage, StorageProxy};
use crate::graph::GraphError;
use crate::transact::Transact;

/// Marker trait for graph representations that promise to be in a consistent
/// state.
///
/// This trait is only implemented by representations that ensure that their
/// storage is only ever mutated via the mutation API (and therefore is
/// consistent). Note that `Core` does not implement this trait and instead
/// acts as a raw container for topological storage that can be freely
/// manipulated.
///
/// This trait allows code to make assumptions about the data it operates
/// against. For example, views expose an API to user code that assumes that
/// topologies are present and therefore unwraps values.
pub trait Consistent {}

impl<'a, T> Consistent for &'a T where T: Consistent {}

impl<'a, T> Consistent for &'a mut T where T: Consistent {}

/// Graph mutation.
pub struct Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    inner: FaceMutation<G>,
    phantom: PhantomData<M>,
}

impl<M, G> AsRef<Self> for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<M, G> AsMut<Self> for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

impl<M, G> AsStorage<Arc<G>> for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    fn as_storage(&self) -> &StorageProxy<Arc<G>> {
        self.inner.as_arc_storage()
    }
}

impl<M, G> AsStorage<Face<G>> for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    fn as_storage(&self) -> &StorageProxy<Face<G>> {
        self.inner.as_face_storage()
    }
}

impl<M, G> AsStorage<Vertex<G>> for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    fn as_storage(&self) -> &StorageProxy<Vertex<G>> {
        self.inner.as_vertex_storage()
    }
}

// TODO: This is a hack. Replace this with delegation.
impl<M, G> Deref for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    type Target = FaceMutation<G>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<M, G> DerefMut for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<M, G> From<M> for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    fn from(graph: M) -> Self {
        Mutation {
            inner: graph.into().into(),
            phantom: PhantomData,
        }
    }
}

impl<M, G> Transact<M> for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    type Output = M;
    type Error = GraphError;

    fn commit(self) -> Result<Self::Output, Self::Error> {
        self.inner.commit().map(|core| core.into())
    }
}

pub trait Mutable<G>: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>
where
    G: GraphGeometry,
{
}

impl<T, G> Mutable<G> for T
where
    T: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
}
