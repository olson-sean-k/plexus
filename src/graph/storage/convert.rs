use crate::graph::storage::Storage;
use crate::graph::topology::Topological;

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
    T: Topological,
{
    fn as_storage(&self) -> &Storage<T>;
}

impl<'a, T, U> AsStorage<T> for &'a U
where
    T: Topological,
    U: AsStorage<T>,
{
    fn as_storage(&self) -> &Storage<T> {
        <U as AsStorage<T>>::as_storage(self)
    }
}

impl<'a, T, U> AsStorage<T> for &'a mut U
where
    T: Topological,
    U: AsStorage<T>,
{
    fn as_storage(&self) -> &Storage<T> {
        <U as AsStorage<T>>::as_storage(self)
    }
}

pub trait AsStorageMut<T>
where
    T: Topological,
{
    fn as_storage_mut(&mut self) -> &mut Storage<T>;
}

impl<'a, T, U> AsStorageMut<T> for &'a mut U
where
    T: Topological,
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
    use crate::graph::topology::{Arc, Edge, Face, Vertex};

    pub trait AsVertexStorage<G>: AsStorage<Vertex<G>>
    where
        G: Geometry,
    {
        fn as_vertex_storage(&self) -> &Storage<Vertex<G>> {
            self.as_storage()
        }
    }

    impl<T, G> AsVertexStorage<G> for T
    where
        T: AsStorage<Vertex<G>>,
        G: Geometry,
    {
    }

    pub trait AsVertexStorageMut<G>: AsStorageMut<Vertex<G>>
    where
        G: Geometry,
    {
        fn as_vertex_storage_mut(&mut self) -> &mut Storage<Vertex<G>> {
            self.as_storage_mut()
        }
    }

    impl<T, G> AsVertexStorageMut<G> for T
    where
        T: AsStorageMut<Vertex<G>>,
        G: Geometry,
    {
    }

    pub trait AsArcStorage<G>: AsStorage<Arc<G>>
    where
        G: Geometry,
    {
        fn as_arc_storage(&self) -> &Storage<Arc<G>> {
            self.as_storage()
        }
    }

    impl<T, G> AsArcStorage<G> for T
    where
        T: AsStorage<Arc<G>>,
        G: Geometry,
    {
    }

    pub trait AsArcStorageMut<G>: AsStorageMut<Arc<G>>
    where
        G: Geometry,
    {
        fn as_arc_storage_mut(&mut self) -> &mut Storage<Arc<G>> {
            self.as_storage_mut()
        }
    }

    impl<T, G> AsArcStorageMut<G> for T
    where
        T: AsStorageMut<Arc<G>>,
        G: Geometry,
    {
    }

    pub trait AsEdgeStorage<G>: AsStorage<Edge<G>>
    where
        G: Geometry,
    {
        fn as_edge_storage(&self) -> &Storage<Edge<G>> {
            self.as_storage()
        }
    }

    impl<T, G> AsEdgeStorage<G> for T
    where
        T: AsStorage<Edge<G>>,
        G: Geometry,
    {
    }

    pub trait AsEdgeStorageMut<G>: AsStorageMut<Edge<G>>
    where
        G: Geometry,
    {
        fn as_edge_storage_mut(&mut self) -> &mut Storage<Edge<G>> {
            self.as_storage_mut()
        }
    }

    impl<T, G> AsEdgeStorageMut<G> for T
    where
        T: AsStorageMut<Edge<G>>,
        G: Geometry,
    {
    }

    pub trait AsFaceStorage<G>: AsStorage<Face<G>>
    where
        G: Geometry,
    {
        fn as_face_storage(&self) -> &Storage<Face<G>> {
            self.as_storage()
        }
    }

    impl<T, G> AsFaceStorage<G> for T
    where
        T: AsStorage<Face<G>>,
        G: Geometry,
    {
    }

    pub trait AsFaceStorageMut<G>: AsStorageMut<Face<G>>
    where
        G: Geometry,
    {
        fn as_face_storage_mut(&mut self) -> &mut Storage<Face<G>> {
            self.as_storage_mut()
        }
    }

    impl<T, G> AsFaceStorageMut<G> for T
    where
        T: AsStorageMut<Face<G>>,
        G: Geometry,
    {
    }
}
