use crate::geometry::Geometry;
use crate::graph::container::alias::OwnedCore;
use crate::graph::container::{Bind, Consistent, Core, Reborrow};
use crate::graph::mutation::edge::{self, EdgeRemoveCache};
use crate::graph::mutation::{Mutate, Mutation};
use crate::graph::storage::convert::AsStorage;
use crate::graph::storage::{HalfKey, Storage, VertexKey};
use crate::graph::topology::Vertex;
use crate::graph::view::convert::FromKeyedSource;
use crate::graph::view::VertexView;
use crate::graph::GraphError;

pub struct VertexMutation<G>
where
    G: Geometry,
{
    storage: Storage<Vertex<G>>,
}

impl<G> VertexMutation<G>
where
    G: Geometry,
{
    pub fn insert_vertex(&mut self, geometry: G::Vertex) -> VertexKey {
        self.storage.insert(Vertex::new(geometry))
    }

    pub fn connect_outgoing_half(&mut self, a: VertexKey, ab: HalfKey) -> Result<(), GraphError> {
        VertexView::from_keyed_source((a, &mut self.storage))
            .ok_or_else(|| GraphError::TopologyNotFound)
            .map(|mut vertex| {
                vertex.half = Some(ab);
            })
    }

    pub fn disconnect_outgoing_half(
        &mut self,
        a: VertexKey,
    ) -> Result<Option<HalfKey>, GraphError> {
        VertexView::from_keyed_source((a, &mut self.storage))
            .ok_or_else(|| GraphError::TopologyNotFound)
            .map(|mut vertex| vertex.half.take())
    }
}

impl<G> AsStorage<Vertex<G>> for VertexMutation<G>
where
    G: Geometry,
{
    fn as_storage(&self) -> &Storage<Vertex<G>> {
        &self.storage
    }
}

impl<G> Mutate for VertexMutation<G>
where
    G: Geometry,
{
    type Mutant = Core<Storage<Vertex<G>>, (), ()>;
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
    G: Geometry,
{
    cache: Vec<EdgeRemoveCache<G>>,
}

impl<G> VertexRemoveCache<G>
where
    G: Geometry,
{
    pub fn snapshot<M>(storage: M, a: VertexKey) -> Result<Self, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Vertex<G>> + Consistent,
    {
        unimplemented!()
    }
}

pub fn remove_with_cache<M, N, G>(
    mut mutation: N,
    cache: VertexRemoveCache<G>,
) -> Result<Vertex<G>, GraphError>
where
    N: AsMut<Mutation<M, G>>,
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: Geometry,
{
    let VertexRemoveCache { cache } = cache;
    for cache in cache {
        edge::remove_with_cache(mutation.as_mut(), cache)?;
    }
    unimplemented!()
}
