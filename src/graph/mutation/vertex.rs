use geometry::Geometry;
use graph::container::{Bind, Core};
use graph::mutation::Mutate;
use graph::storage::convert::AsStorage;
use graph::storage::{EdgeKey, Storage, VertexKey};
use graph::topology::Vertex;
use graph::GraphError;

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
    pub fn as_vertex_storage(&self) -> &Storage<Vertex<G>> {
        self.as_storage()
    }

    pub fn insert_vertex(&mut self, geometry: G::Vertex) -> VertexKey {
        self.storage.insert(Vertex::new(geometry))
    }

    pub fn connect_outgoing_edge(&mut self, a: VertexKey, ab: EdgeKey) -> Result<(), GraphError> {
        if a == ab.to_vertex_keys().0 {
            let vertex = self
                .storage
                .get_mut(&a)
                .ok_or_else(|| GraphError::TopologyNotFound)?;
            vertex.edge = Some(ab);
            Ok(())
        }
        else {
            Err(GraphError::TopologyMalformed)
        }
    }

    pub fn disconnect_outgoing_edge(
        &mut self,
        a: VertexKey,
    ) -> Result<Option<EdgeKey>, GraphError> {
        let edge = self
            .storage
            .get_mut(&a)
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .edge
            .take();
        Ok(edge)
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
