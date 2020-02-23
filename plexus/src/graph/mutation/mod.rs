pub mod edge;
pub mod face;
pub mod vertex;

use std::ops::{Deref, DerefMut};

use crate::graph::core::OwnedCore;
use crate::graph::geometry::{Geometric, Geometry};
use crate::graph::mutation::face::FaceMutation;
use crate::graph::storage::alias::*;
use crate::graph::storage::payload::{Arc, Face, Vertex};
use crate::graph::storage::{AsStorage, StorageProxy};
use crate::graph::GraphError;
use crate::transact::Transact;

// TODO: Remove geometric information from mutations. For entities and storage,
//       forward the `Geometry` associated type from the core. Accept geometry
//       in mutation functions (and not when in cache types nor when
//       snapshotting).

/// Marker trait for graph representations that promise to be in a consistent
/// state.
///
/// This trait is only implemented by representations that ensure that their
/// storage is only ever mutated via the mutation API (and therefore is
/// consistent). Note that `Core` does not implement this trait and instead acts
/// as a raw container for topological storage that can be freely manipulated.
///
/// This trait allows code to make assumptions about the data it operates
/// against. For example, views expose an API to user code that assumes that
/// topologies are present and therefore unwraps values.
pub trait Consistent {}

impl<'a, T> Consistent for &'a T where T: Consistent {}

impl<'a, T> Consistent for &'a mut T where T: Consistent {}

/// Graph mutation.
pub struct Mutation<M>
where
    M: Consistent + From<OwnedCore<Geometry<M>>> + Geometric + Into<OwnedCore<Geometry<M>>>,
{
    inner: FaceMutation<M>,
}

impl<M> AsRef<Self> for Mutation<M>
where
    M: Consistent + From<OwnedCore<Geometry<M>>> + Geometric + Into<OwnedCore<Geometry<M>>>,
{
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<M> AsMut<Self> for Mutation<M>
where
    M: Consistent + From<OwnedCore<Geometry<M>>> + Geometric + Into<OwnedCore<Geometry<M>>>,
{
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

impl<M> AsStorage<Arc<Geometry<M>>> for Mutation<M>
where
    M: Consistent + From<OwnedCore<Geometry<M>>> + Geometric + Into<OwnedCore<Geometry<M>>>,
{
    fn as_storage(&self) -> &StorageProxy<Arc<Geometry<M>>> {
        self.inner.as_arc_storage()
    }
}

impl<M> AsStorage<Face<Geometry<M>>> for Mutation<M>
where
    M: Consistent + From<OwnedCore<Geometry<M>>> + Geometric + Into<OwnedCore<Geometry<M>>>,
{
    fn as_storage(&self) -> &StorageProxy<Face<Geometry<M>>> {
        self.inner.as_face_storage()
    }
}

impl<M> AsStorage<Vertex<Geometry<M>>> for Mutation<M>
where
    M: Consistent + From<OwnedCore<Geometry<M>>> + Geometric + Into<OwnedCore<Geometry<M>>>,
{
    fn as_storage(&self) -> &StorageProxy<Vertex<Geometry<M>>> {
        self.inner.as_vertex_storage()
    }
}

// TODO: This is a hack. Replace this with delegation.
impl<M> Deref for Mutation<M>
where
    M: Consistent + From<OwnedCore<Geometry<M>>> + Geometric + Into<OwnedCore<Geometry<M>>>,
{
    type Target = FaceMutation<M>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<M> DerefMut for Mutation<M>
where
    M: Consistent + From<OwnedCore<Geometry<M>>> + Geometric + Into<OwnedCore<Geometry<M>>>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<M> From<M> for Mutation<M>
where
    M: Consistent + From<OwnedCore<Geometry<M>>> + Geometric + Into<OwnedCore<Geometry<M>>>,
{
    fn from(graph: M) -> Self {
        Mutation {
            inner: graph.into().into(),
        }
    }
}

impl<M> Transact<M> for Mutation<M>
where
    M: Consistent + From<OwnedCore<Geometry<M>>> + Geometric + Into<OwnedCore<Geometry<M>>>,
{
    type Output = M;
    type Error = GraphError;

    fn commit(self) -> Result<Self::Output, Self::Error> {
        self.inner.commit().map(|core| core.into())
    }
}

pub trait Mutable:
    Consistent + From<OwnedCore<Geometry<Self>>> + Geometric + Into<OwnedCore<Geometry<Self>>>
{
}

impl<M> Mutable for M where
    M: Consistent + From<OwnedCore<Geometry<M>>> + Geometric + Into<OwnedCore<Geometry<M>>>
{
}
