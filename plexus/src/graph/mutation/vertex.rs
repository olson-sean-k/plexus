use crate::graph::borrow::Reborrow;
use crate::graph::core::{Bind, Core};
use crate::graph::geometry::GraphGeometry;
use crate::graph::mutation::edge::{self, EdgeRemoveCache};
use crate::graph::mutation::{Consistent, Mutable, Mutation};
use crate::graph::storage::key::{ArcKey, VertexKey};
use crate::graph::storage::payload::VertexPayload;
use crate::graph::storage::{AsStorage, StorageProxy};
use crate::graph::view::vertex::VertexView;
use crate::graph::view::FromKeyedSource;
use crate::graph::GraphError;
use crate::transact::Transact;

type Mutant<G> = Core<StorageProxy<VertexPayload<G>>, (), (), ()>;

pub struct VertexMutation<G>
where
    G: GraphGeometry,
{
    storage: StorageProxy<VertexPayload<G>>,
}

impl<G> VertexMutation<G>
where
    G: GraphGeometry,
{
    pub fn insert_vertex(&mut self, geometry: G::Vertex) -> VertexKey {
        self.storage.insert(VertexPayload::new(geometry))
    }

    pub fn connect_outgoing_arc(&mut self, a: VertexKey, ab: ArcKey) -> Result<(), GraphError> {
        VertexView::from_keyed_source((a, &mut self.storage))
            .ok_or_else(|| GraphError::TopologyNotFound)
            .map(|mut vertex| {
                vertex.arc = Some(ab);
            })
    }

    // TODO: See `edge::split_with_cache`.
    #[allow(dead_code)]
    pub fn disconnect_outgoing_arc(&mut self, a: VertexKey) -> Result<Option<ArcKey>, GraphError> {
        VertexView::from_keyed_source((a, &mut self.storage))
            .ok_or_else(|| GraphError::TopologyNotFound)
            .map(|mut vertex| vertex.arc.take())
    }
}

impl<G> AsStorage<VertexPayload<G>> for VertexMutation<G>
where
    G: GraphGeometry,
{
    fn as_storage(&self) -> &StorageProxy<VertexPayload<G>> {
        &self.storage
    }
}

impl<G> From<Mutant<G>> for VertexMutation<G>
where
    G: GraphGeometry,
{
    fn from(core: Mutant<G>) -> Self {
        let (vertices, ..) = core.into_storage();
        VertexMutation { storage: vertices }
    }
}

impl<G> Transact<Mutant<G>> for VertexMutation<G>
where
    G: GraphGeometry,
{
    type Output = Mutant<G>;
    type Error = GraphError;

    fn commit(self) -> Result<Self::Output, Self::Error> {
        let VertexMutation {
            storage: vertices, ..
        } = self;
        // In a consistent graph, all vertices must have a leading arc.
        for (_, vertex) in vertices.iter() {
            if vertex.arc.is_none() {
                return Err(GraphError::TopologyMalformed);
            }
        }
        Ok(Core::empty().bind(vertices))
    }
}

pub struct VertexRemoveCache<G>
where
    G: GraphGeometry,
{
    cache: Vec<EdgeRemoveCache<G>>,
}

impl<G> VertexRemoveCache<G>
where
    G: GraphGeometry,
{
    pub fn snapshot<M>(storage: M, a: VertexKey) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<VertexPayload<G>> + Consistent,
    {
        let _ = (storage, a);
        unimplemented!()
    }
}

pub fn remove_with_cache<M, N, G>(
    mut mutation: N,
    cache: VertexRemoveCache<G>,
) -> Result<VertexPayload<G>, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Mutable<G>,
    G: GraphGeometry,
{
    let VertexRemoveCache { cache } = cache;
    for cache in cache {
        edge::remove_with_cache(mutation.as_mut(), cache)?;
    }
    unimplemented!()
}
