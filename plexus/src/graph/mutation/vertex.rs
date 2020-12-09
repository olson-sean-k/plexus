use crate::entity::borrow::Reborrow;
use crate::entity::storage::prelude::*;
use crate::entity::storage::{AsStorage, AsStorageMut, Fuse, StorageTarget};
use crate::graph::core::Core;
use crate::graph::data::{Data, GraphData, Parametric};
use crate::graph::edge::ArcKey;
use crate::graph::mutation::edge::{self, EdgeRemoveCache};
use crate::graph::mutation::{Consistent, Immediate, Mode, Mutable, Mutation};
use crate::graph::vertex::{Vertex, VertexKey, VertexView};
use crate::graph::GraphError;
use crate::transact::{Bypass, Transact};

type ModalCore<P> = Core<Data<<P as Mode>::Graph>, <P as Mode>::VertexStorage, (), (), ()>;
#[cfg(not(all(nightly, feature = "unstable")))]
type RefCore<'a, G> = Core<G, &'a StorageTarget<Vertex<G>>, (), (), ()>;
#[cfg(all(nightly, feature = "unstable"))]
type RefCore<'a, G> = Core<G, &'a StorageTarget<'a, Vertex<G>>, (), (), ()>;

pub struct VertexMutation<P>
where
    P: Mode,
{
    storage: P::VertexStorage,
}

impl<P> VertexMutation<P>
where
    P: Mode,
{
    pub fn to_ref_core(&self) -> RefCore<Data<P::Graph>> {
        Core::empty().fuse(self.storage.as_storage())
    }

    pub fn connect_outgoing_arc(&mut self, a: VertexKey, ab: ArcKey) -> Result<(), GraphError> {
        self.with_vertex_mut(a, |vertex| vertex.arc = Some(ab))
    }

    // TODO: See `edge::split_with_cache`.
    #[allow(dead_code)]
    pub fn disconnect_outgoing_arc(&mut self, a: VertexKey) -> Result<Option<ArcKey>, GraphError> {
        self.with_vertex_mut(a, |vertex| vertex.arc.take())
    }

    fn with_vertex_mut<T, F>(&mut self, a: VertexKey, mut f: F) -> Result<T, GraphError>
    where
        F: FnMut(&mut Vertex<Data<P::Graph>>) -> T,
    {
        let vertex = self
            .storage
            .as_storage_mut()
            .get_mut(&a)
            .ok_or(GraphError::TopologyNotFound)?;
        Ok(f(vertex))
    }
}

impl<P> AsStorage<Vertex<Data<P::Graph>>> for VertexMutation<P>
where
    P: Mode,
{
    fn as_storage(&self) -> &StorageTarget<Vertex<Data<P::Graph>>> {
        self.storage.as_storage()
    }
}

impl<M> Bypass<ModalCore<Immediate<M>>> for VertexMutation<Immediate<M>>
where
    M: Parametric,
{
    fn bypass(self) -> Self::Commit {
        let VertexMutation {
            storage: vertices, ..
        } = self;
        Core::empty().fuse(vertices)
    }
}

impl<P> From<ModalCore<P>> for VertexMutation<P>
where
    P: Mode,
{
    fn from(core: ModalCore<P>) -> Self {
        let (vertices, ..) = core.unfuse();
        VertexMutation { storage: vertices }
    }
}

impl<M> Transact<ModalCore<Immediate<M>>> for VertexMutation<Immediate<M>>
where
    M: Parametric,
{
    type Commit = ModalCore<Immediate<M>>;
    type Abort = ();
    type Error = GraphError;

    fn commit(self) -> Result<Self::Commit, (Self::Abort, Self::Error)> {
        let VertexMutation {
            storage: vertices, ..
        } = self;
        // In a consistent graph, all vertices must have a leading arc.
        for (_, vertex) in vertices.as_storage().iter() {
            if vertex.arc.is_none() {
                return Err(((), GraphError::TopologyMalformed));
            }
        }
        Ok(Core::empty().fuse(vertices))
    }

    fn abort(self) -> Self::Abort {}
}

pub struct VertexRemoveCache {
    cache: Vec<EdgeRemoveCache>,
}

impl VertexRemoveCache {
    pub fn from_vertex<B>(vertex: VertexView<B>) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Vertex<Data<B>>> + Consistent + Parametric,
    {
        let _ = vertex;
        unimplemented!()
    }
}

pub fn insert<N, P>(mut mutation: N, data: <Data<P::Graph> as GraphData>::Vertex) -> VertexKey
where
    N: AsMut<Mutation<P>>,
    P: Mode,
    P::Graph: Mutable,
{
    mutation
        .as_mut()
        .storage
        .as_storage_mut()
        .insert(Vertex::new(data))
}

pub fn remove<N, P>(
    mut mutation: N,
    cache: VertexRemoveCache,
) -> Result<Vertex<Data<P::Graph>>, GraphError>
where
    N: AsMut<Mutation<P>>,
    P: Mode,
    P::Graph: Mutable,
{
    let VertexRemoveCache { cache } = cache;
    for cache in cache {
        edge::remove(mutation.as_mut(), cache)?;
    }
    unimplemented!()
}
