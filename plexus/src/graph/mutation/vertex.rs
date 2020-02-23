use crate::graph::borrow::Reborrow;
use crate::graph::core::{Core, Fuse};
use crate::graph::geometry::{Geometric, Geometry, GraphGeometry};
use crate::graph::mutation::edge::{self, EdgeRemoveCache};
use crate::graph::mutation::{Consistent, Mutable, Mutation};
use crate::graph::storage::key::{ArcKey, VertexKey};
use crate::graph::storage::payload::Vertex;
use crate::graph::storage::{AsStorage, StorageProxy};
use crate::graph::view::View;
use crate::graph::GraphError;
use crate::transact::Transact;

type Mutant<G> = Core<G, StorageProxy<Vertex<G>>, (), (), ()>;

pub struct VertexMutation<M>
where
    M: Geometric,
{
    storage: StorageProxy<Vertex<Geometry<M>>>,
}

impl<M, G> VertexMutation<M>
where
    M: Geometric<Geometry = G>,
    G: GraphGeometry,
{
    pub fn insert_vertex(&mut self, geometry: G::Vertex) -> VertexKey {
        self.storage.insert(Vertex::new(geometry))
    }

    pub fn connect_outgoing_arc(&mut self, a: VertexKey, ab: ArcKey) -> Result<(), GraphError> {
        View::bind(&mut self.storage, a)
            .ok_or_else(|| GraphError::TopologyNotFound)
            .map(|mut vertex| {
                vertex.arc = Some(ab);
            })
    }

    // TODO: See `edge::split_with_cache`.
    #[allow(dead_code)]
    pub fn disconnect_outgoing_arc(&mut self, a: VertexKey) -> Result<Option<ArcKey>, GraphError> {
        View::bind(&mut self.storage, a)
            .ok_or_else(|| GraphError::TopologyNotFound)
            .map(|mut vertex| vertex.arc.take())
    }
}

impl<M, G> AsStorage<Vertex<G>> for VertexMutation<M>
where
    M: Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn as_storage(&self) -> &StorageProxy<Vertex<G>> {
        &self.storage
    }
}

impl<M, G> From<Mutant<G>> for VertexMutation<M>
where
    M: Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(core: Mutant<G>) -> Self {
        let (vertices, ..) = core.unfuse();
        VertexMutation { storage: vertices }
    }
}

impl<M, G> Transact<Mutant<G>> for VertexMutation<M>
where
    M: Geometric<Geometry = G>,
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
        Ok(Core::empty().fuse(vertices))
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
        M::Target: AsStorage<Vertex<G>> + Consistent,
    {
        let _ = (storage, a);
        unimplemented!()
    }
}

pub fn remove_with_cache<M, N, G>(
    mut mutation: N,
    cache: VertexRemoveCache<G>,
) -> Result<Vertex<G>, GraphError>
where
    N: AsMut<Mutation<M>>,
    M: Mutable<Geometry = G>,
    G: GraphGeometry,
{
    let VertexRemoveCache { cache } = cache;
    for cache in cache {
        edge::remove_with_cache(mutation.as_mut(), cache)?;
    }
    unimplemented!()
}
