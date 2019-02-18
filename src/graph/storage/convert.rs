use crate::graph::payload::Payload;
use crate::graph::storage::Storage;

pub trait FromInnerKey<K> {
    fn from_inner_key(key: K) -> Self;
}

pub trait IntoOpaqueKey<K> {
    fn into_opaque_key(self) -> K;
}

impl<K, I> IntoOpaqueKey<I> for K
where
    I: FromInnerKey<K>,
{
    fn into_opaque_key(self) -> I {
        I::from_inner_key(self)
    }
}

pub trait AsStorage<T>
where
    T: Payload,
{
    fn as_storage(&self) -> &Storage<T>;
}

impl<'a, T, U> AsStorage<T> for &'a U
where
    T: Payload,
    U: AsStorage<T>,
{
    fn as_storage(&self) -> &Storage<T> {
        <U as AsStorage<T>>::as_storage(self)
    }
}

impl<'a, T, U> AsStorage<T> for &'a mut U
where
    T: Payload,
    U: AsStorage<T>,
{
    fn as_storage(&self) -> &Storage<T> {
        <U as AsStorage<T>>::as_storage(self)
    }
}

pub trait AsStorageMut<T>
where
    T: Payload,
{
    fn as_storage_mut(&mut self) -> &mut Storage<T>;
}

impl<'a, T, U> AsStorageMut<T> for &'a mut U
where
    T: Payload,
    U: AsStorageMut<T>,
{
    fn as_storage_mut(&mut self) -> &mut Storage<T> {
        <U as AsStorageMut<T>>::as_storage_mut(self)
    }
}

// These aliases are not truly aliases, but thin traits. They are useful when
// `as_storage` or `as_storage_mut` are ambiguous and allow for a particular
// topology to be easily isolated.
pub mod alias {
    use super::*;

    use crate::geometry::Geometry;
    use crate::graph::payload::{ArcPayload, EdgePayload, FacePayload, VertexPayload};

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
}
