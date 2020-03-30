use crate::graph::entity::{Arc, Edge, Face, Vertex};
use crate::graph::geometry::GraphGeometry;
use crate::network::storage::{AsStorage, AsStorageMut, Storage};

pub trait AsVertexStorage<G>: AsStorage<Vertex<G>>
where
    G: GraphGeometry,
{
    #[inline(always)]
    fn as_vertex_storage(&self) -> &Storage<Vertex<G>> {
        self.as_storage()
    }
}

impl<T, G> AsVertexStorage<G> for T
where
    T: AsStorage<Vertex<G>>,
    G: GraphGeometry,
{
}

pub trait AsVertexStorageMut<G>: AsStorageMut<Vertex<G>>
where
    G: GraphGeometry,
{
    #[inline(always)]
    fn as_vertex_storage_mut(&mut self) -> &mut Storage<Vertex<G>> {
        self.as_storage_mut()
    }
}

impl<T, G> AsVertexStorageMut<G> for T
where
    T: AsStorageMut<Vertex<G>>,
    G: GraphGeometry,
{
}

pub trait AsArcStorage<G>: AsStorage<Arc<G>>
where
    G: GraphGeometry,
{
    #[inline(always)]
    fn as_arc_storage(&self) -> &Storage<Arc<G>> {
        self.as_storage()
    }
}

impl<T, G> AsArcStorage<G> for T
where
    T: AsStorage<Arc<G>>,
    G: GraphGeometry,
{
}

pub trait AsArcStorageMut<G>: AsStorageMut<Arc<G>>
where
    G: GraphGeometry,
{
    #[inline(always)]
    fn as_arc_storage_mut(&mut self) -> &mut Storage<Arc<G>> {
        self.as_storage_mut()
    }
}

impl<T, G> AsArcStorageMut<G> for T
where
    T: AsStorageMut<Arc<G>>,
    G: GraphGeometry,
{
}

pub trait AsEdgeStorage<G>: AsStorage<Edge<G>>
where
    G: GraphGeometry,
{
    #[inline(always)]
    fn as_edge_storage(&self) -> &Storage<Edge<G>> {
        self.as_storage()
    }
}

impl<T, G> AsEdgeStorage<G> for T
where
    T: AsStorage<Edge<G>>,
    G: GraphGeometry,
{
}

pub trait AsEdgeStorageMut<G>: AsStorageMut<Edge<G>>
where
    G: GraphGeometry,
{
    #[inline(always)]
    fn as_edge_storage_mut(&mut self) -> &mut Storage<Edge<G>> {
        self.as_storage_mut()
    }
}

impl<T, G> AsEdgeStorageMut<G> for T
where
    T: AsStorageMut<Edge<G>>,
    G: GraphGeometry,
{
}

pub trait AsFaceStorage<G>: AsStorage<Face<G>>
where
    G: GraphGeometry,
{
    #[inline(always)]
    fn as_face_storage(&self) -> &Storage<Face<G>> {
        self.as_storage()
    }
}

impl<T, G> AsFaceStorage<G> for T
where
    T: AsStorage<Face<G>>,
    G: GraphGeometry,
{
}

pub trait AsFaceStorageMut<G>: AsStorageMut<Face<G>>
where
    G: GraphGeometry,
{
    #[inline(always)]
    fn as_face_storage_mut(&mut self) -> &mut Storage<Face<G>> {
        self.as_storage_mut()
    }
}

impl<T, G> AsFaceStorageMut<G> for T
where
    T: AsStorageMut<Face<G>>,
    G: GraphGeometry,
{
}
