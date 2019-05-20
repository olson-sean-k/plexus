use crate::graph::geometry::Geometry;
use crate::graph::payload::{ArcPayload, EdgePayload, FacePayload, VertexPayload};
use crate::graph::storage::{AsStorage, AsStorageMut, Storage};

pub trait AsVertexStorage<G>: AsStorage<VertexPayload<G>>
where
    G: Geometry,
{
    fn as_vertex_storage(&self) -> &Storage<VertexPayload<G>> {
        self.as_storage()
    }
}

impl<T, G> AsVertexStorage<G> for T
where
    T: AsStorage<VertexPayload<G>>,
    G: Geometry,
{
}

pub trait AsVertexStorageMut<G>: AsStorageMut<VertexPayload<G>>
where
    G: Geometry,
{
    fn as_vertex_storage_mut(&mut self) -> &mut Storage<VertexPayload<G>> {
        self.as_storage_mut()
    }
}

impl<T, G> AsVertexStorageMut<G> for T
where
    T: AsStorageMut<VertexPayload<G>>,
    G: Geometry,
{
}

pub trait AsArcStorage<G>: AsStorage<ArcPayload<G>>
where
    G: Geometry,
{
    fn as_arc_storage(&self) -> &Storage<ArcPayload<G>> {
        self.as_storage()
    }
}

impl<T, G> AsArcStorage<G> for T
where
    T: AsStorage<ArcPayload<G>>,
    G: Geometry,
{
}

pub trait AsArcStorageMut<G>: AsStorageMut<ArcPayload<G>>
where
    G: Geometry,
{
    fn as_arc_storage_mut(&mut self) -> &mut Storage<ArcPayload<G>> {
        self.as_storage_mut()
    }
}

impl<T, G> AsArcStorageMut<G> for T
where
    T: AsStorageMut<ArcPayload<G>>,
    G: Geometry,
{
}

pub trait AsEdgeStorage<G>: AsStorage<EdgePayload<G>>
where
    G: Geometry,
{
    fn as_edge_storage(&self) -> &Storage<EdgePayload<G>> {
        self.as_storage()
    }
}

impl<T, G> AsEdgeStorage<G> for T
where
    T: AsStorage<EdgePayload<G>>,
    G: Geometry,
{
}

pub trait AsEdgeStorageMut<G>: AsStorageMut<EdgePayload<G>>
where
    G: Geometry,
{
    fn as_edge_storage_mut(&mut self) -> &mut Storage<EdgePayload<G>> {
        self.as_storage_mut()
    }
}

impl<T, G> AsEdgeStorageMut<G> for T
where
    T: AsStorageMut<EdgePayload<G>>,
    G: Geometry,
{
}

pub trait AsFaceStorage<G>: AsStorage<FacePayload<G>>
where
    G: Geometry,
{
    fn as_face_storage(&self) -> &Storage<FacePayload<G>> {
        self.as_storage()
    }
}

impl<T, G> AsFaceStorage<G> for T
where
    T: AsStorage<FacePayload<G>>,
    G: Geometry,
{
}

pub trait AsFaceStorageMut<G>: AsStorageMut<FacePayload<G>>
where
    G: Geometry,
{
    fn as_face_storage_mut(&mut self) -> &mut Storage<FacePayload<G>> {
        self.as_storage_mut()
    }
}

impl<T, G> AsFaceStorageMut<G> for T
where
    T: AsStorageMut<FacePayload<G>>,
    G: Geometry,
{
}
