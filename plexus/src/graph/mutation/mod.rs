pub mod edge;
pub mod face;
pub mod path;
pub mod vertex;

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use crate::entity::storage::{AsStorage, AsStorageMut, StorageTarget};
use crate::entity::Entity;
use crate::graph::core::OwnedCore;
use crate::graph::data::{Data, Parametric};
use crate::graph::edge::{Arc, Edge};
use crate::graph::face::Face;
use crate::graph::mutation::face::FaceMutation;
use crate::graph::vertex::Vertex;
use crate::graph::{GraphData, GraphError};
use crate::transact::{Bypass, Transact};

// TODO: The stable toolchain does not allow a type parameter `G` to be
//       introduced and bound to the associated type `Mode::Graph::Data`. The
//       compiler does not seem to consider the types equal, and requires
//       redundant type bounds on `Mode`'s associated storage types at each
//       usage. The nightly toolchain already supports this. Reintroduce a
//       `G: GraphData` type parameter in implementation blocks when this is
//       fixed. For now, this code uses `Data<P::Graph>`. See the following
//       related issues:
//
//       https://github.com/rust-lang/rust/issues/58231
//       https://github.com/rust-lang/rust/issues/70703
//       https://github.com/rust-lang/rust/issues/47897

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

pub trait Mode {
    type Graph: Parametric;
    type VertexStorage: AsStorageMut<Vertex<Data<Self::Graph>>>;
    type ArcStorage: AsStorageMut<Arc<Data<Self::Graph>>>;
    type EdgeStorage: AsStorageMut<Edge<Data<Self::Graph>>>;
    type FaceStorage: AsStorageMut<Face<Data<Self::Graph>>>;
}

pub struct Immediate<M>
where
    M: Parametric,
{
    phantom: PhantomData<fn() -> M>,
}

impl<M> Mode for Immediate<M>
where
    M: Parametric,
{
    type Graph = M;
    type VertexStorage = <Vertex<Data<M>> as Entity>::Storage;
    type ArcStorage = <Arc<Data<M>> as Entity>::Storage;
    type EdgeStorage = <Edge<Data<M>> as Entity>::Storage;
    type FaceStorage = <Face<Data<M>> as Entity>::Storage;
}

/// Graph mutation.
pub struct Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    inner: FaceMutation<P>,
}

impl<P> AsRef<Self> for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<P> AsMut<Self> for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

impl<P> AsStorage<Arc<Data<P::Graph>>> for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    fn as_storage(&self) -> &StorageTarget<Arc<Data<P::Graph>>> {
        self.inner.to_ref_core().unfuse().1
    }
}

impl<P> AsStorage<Edge<Data<P::Graph>>> for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    fn as_storage(&self) -> &StorageTarget<Edge<Data<P::Graph>>> {
        self.inner.to_ref_core().unfuse().2
    }
}

impl<P> AsStorage<Face<Data<P::Graph>>> for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    fn as_storage(&self) -> &StorageTarget<Face<Data<P::Graph>>> {
        self.inner.to_ref_core().unfuse().3
    }
}

impl<P> AsStorage<Vertex<Data<P::Graph>>> for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    fn as_storage(&self) -> &StorageTarget<Vertex<Data<P::Graph>>> {
        self.inner.to_ref_core().unfuse().0
    }
}

impl<M> Bypass<M> for Mutation<Immediate<M>>
where
    M: Consistent + From<OwnedCore<Data<M>>> + Parametric + Into<OwnedCore<Data<M>>>,
{
    fn bypass(self) -> Self::Commit {
        self.inner.bypass().into()
    }
}

// TODO: This is a hack. Replace this with delegation.
impl<P> Deref for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    type Target = FaceMutation<P>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<P> DerefMut for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<M> From<M> for Mutation<Immediate<M>>
where
    M: Consistent + From<OwnedCore<Data<M>>> + Parametric + Into<OwnedCore<Data<M>>>,
{
    fn from(graph: M) -> Self {
        Mutation {
            inner: graph.into().into(),
        }
    }
}

impl<P> Parametric for Mutation<P>
where
    P: Mode,
    P::Graph: Consistent + From<OwnedCore<Data<P::Graph>>> + Into<OwnedCore<Data<P::Graph>>>,
{
    type Data = Data<P::Graph>;
}

impl<M> Transact<M> for Mutation<Immediate<M>>
where
    M: Consistent + From<OwnedCore<Data<M>>> + Parametric + Into<OwnedCore<Data<M>>>,
{
    type Commit = M;
    type Abort = ();
    type Error = GraphError;

    fn commit(self) -> Result<Self::Commit, (Self::Abort, Self::Error)> {
        self.inner.commit().map(|core| core.into())
    }

    fn abort(self) -> Self::Abort {}
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
