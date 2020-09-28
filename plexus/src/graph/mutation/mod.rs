pub mod edge;
pub mod face;
pub mod path;
pub mod vertex;

use std::ops::{Deref, DerefMut};

use crate::entity::storage::{AsStorage, StorageObject};
use crate::graph::core::OwnedCore;
use crate::graph::data::{Data, Parametric};
use crate::graph::edge::{Arc, Edge};
use crate::graph::face::Face;
use crate::graph::mutation::face::FaceMutation;
use crate::graph::vertex::Vertex;
use crate::graph::{GraphData, GraphError};
use crate::transact::Transact;

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
    M: Consistent + From<OwnedCore<Data<M>>> + Parametric + Into<OwnedCore<Data<M>>>,
{
    inner: FaceMutation<M>,
}

impl<M, G> Mutation<M>
where
    M: Consistent + From<OwnedCore<G>> + Parametric<Data = G> + Into<OwnedCore<G>>,
    G: GraphData,
{
}

impl<M, G> AsRef<Self> for Mutation<M>
where
    M: Consistent + From<OwnedCore<G>> + Parametric<Data = G> + Into<OwnedCore<G>>,
    G: GraphData,
{
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<M, G> AsMut<Self> for Mutation<M>
where
    M: Consistent + From<OwnedCore<G>> + Parametric<Data = G> + Into<OwnedCore<G>>,
    G: GraphData,
{
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

impl<M, G> AsStorage<Arc<G>> for Mutation<M>
where
    M: Consistent + From<OwnedCore<G>> + Parametric<Data = G> + Into<OwnedCore<G>>,
    G: GraphData,
{
    fn as_storage(&self) -> &StorageObject<Arc<G>> {
        self.inner.to_ref_core().unfuse().1
    }
}

impl<M, G> AsStorage<Edge<G>> for Mutation<M>
where
    M: Consistent + From<OwnedCore<G>> + Parametric<Data = G> + Into<OwnedCore<G>>,
    G: GraphData,
{
    fn as_storage(&self) -> &StorageObject<Edge<G>> {
        self.inner.to_ref_core().unfuse().2
    }
}

impl<M, G> AsStorage<Face<G>> for Mutation<M>
where
    M: Consistent + From<OwnedCore<G>> + Parametric<Data = G> + Into<OwnedCore<G>>,
    G: GraphData,
{
    fn as_storage(&self) -> &StorageObject<Face<G>> {
        self.inner.to_ref_core().unfuse().3
    }
}

impl<M, G> AsStorage<Vertex<G>> for Mutation<M>
where
    M: Consistent + From<OwnedCore<G>> + Parametric<Data = G> + Into<OwnedCore<G>>,
    G: GraphData,
{
    fn as_storage(&self) -> &StorageObject<Vertex<G>> {
        self.inner.to_ref_core().unfuse().0
    }
}

// TODO: This is a hack. Replace this with delegation.
impl<M, G> Deref for Mutation<M>
where
    M: Consistent + From<OwnedCore<G>> + Parametric<Data = G> + Into<OwnedCore<G>>,
    G: GraphData,
{
    type Target = FaceMutation<M>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<M, G> DerefMut for Mutation<M>
where
    M: Consistent + From<OwnedCore<G>> + Parametric<Data = G> + Into<OwnedCore<G>>,
    G: GraphData,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<M, G> From<M> for Mutation<M>
where
    M: Consistent + From<OwnedCore<G>> + Parametric<Data = G> + Into<OwnedCore<G>>,
    G: GraphData,
{
    fn from(graph: M) -> Self {
        Mutation {
            inner: graph.into().into(),
        }
    }
}

impl<M, G> Parametric for Mutation<M>
where
    M: Consistent + From<OwnedCore<G>> + Parametric<Data = G> + Into<OwnedCore<G>>,
    G: GraphData,
{
    type Data = G;
}

impl<M, G> Transact<M> for Mutation<M>
where
    M: Consistent + From<OwnedCore<G>> + Parametric<Data = G> + Into<OwnedCore<G>>,
    G: GraphData,
{
    type Output = M;
    type Error = GraphError;

    fn commit(self) -> Result<Self::Output, Self::Error> {
        self.inner.commit().map(|core| core.into())
    }
}

pub trait Mutable:
    Consistent + From<OwnedCore<Data<Self>>> + Parametric + Into<OwnedCore<Data<Self>>>
{
}

impl<M, G> Mutable for M
where
    M: Consistent + From<OwnedCore<G>> + Parametric<Data = G> + Into<OwnedCore<G>>,
    G: GraphData,
{
}
