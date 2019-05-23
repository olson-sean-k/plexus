use crate::graph::borrow::Reborrow;
use crate::graph::core::{Bind, Core};
use crate::graph::geometry::GraphGeometry;
use crate::graph::mutation::edge::{self, EdgeRemoveCache};
use crate::graph::mutation::{Consistent, Mutable, Mutate, Mutation};
use crate::graph::payload::VertexPayload;
use crate::graph::storage::{ArcKey, AsStorage, Storage, VertexKey};
use crate::graph::view::vertex::VertexView;
use crate::graph::view::FromKeyedSource;
use crate::graph::GraphError;

pub struct VertexMutation<G>
where
    G: GraphGeometry,
{
    storage: Storage<VertexPayload<G>>,
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
    fn as_storage(&self) -> &Storage<VertexPayload<G>> {
        &self.storage
    }
}

impl<G> Mutate for VertexMutation<G>
where
    G: GraphGeometry,
{
    type Mutant = Core<Storage<VertexPayload<G>>, (), (), ()>;
    type Error = GraphError;

    fn commit(self) -> Result<Self::Mutant, Self::Error> {
        let VertexMutation {
            storage: vertices, ..
        } = self;
        Ok(Core::empty().bind(vertices))
    }

    fn mutate(mutant: Self::Mutant) -> Self {
        let (vertices, ..) = mutant.into_storage();
        VertexMutation { storage: vertices }
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
