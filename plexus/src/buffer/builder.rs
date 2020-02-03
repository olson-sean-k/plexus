use arrayvec::Array;
use num::{Integer, NumCast, Unsigned};
use std::hash::Hash;
use theon::FromItems;
use typenum::{self, NonZero};

use crate::buffer::{BufferError, MeshBuffer};
use crate::builder::{FacetBuilder, MeshBuilder, SurfaceBuilder};
use crate::index::{Flat, Grouping, IndexBuffer};
use crate::primitive::{NGon, Polygon, Topological};
use crate::transact::{ClosedInput, Transact};
use crate::IntoGeometry;

// TODO: It should not be possible to manufacture keys without placing
//       additional constraints on the type bounds of `FacetBuilder` (for
//       example, `FacetBuilder<Key = usize>`). Is it important to check for
//       out-of-bounds indices in `insert_facet`?

pub type VertexKey<R> = <Vec<<R as Grouping>::Item> as IndexBuffer<R>>::Index;

pub struct BufferBuilder<R, G>
where
    R: Grouping,
{
    indices: Vec<R::Item>,
    vertices: Vec<G>,
}

impl<R, G> Default for BufferBuilder<R, G>
where
    R: Grouping,
    Vec<R::Item>: IndexBuffer<R>,
{
    fn default() -> Self {
        BufferBuilder {
            indices: Default::default(),
            vertices: Default::default(),
        }
    }
}

impl<R, G> ClosedInput for BufferBuilder<R, G>
where
    R: Grouping,
    Vec<R::Item>: IndexBuffer<R>,
{
    type Input = ();
}

impl<A, N, G> FacetBuilder<N> for BufferBuilder<Flat<A, N>, G>
where
    A: NonZero + typenum::Unsigned,
    N: Copy + Hash + Integer + NumCast + Unsigned,
    Vec<N>: IndexBuffer<Flat<A, N>>,
{
    type Facet = ();
    type Key = ();

    fn insert_facet<T, U>(&mut self, keys: T, _: U) -> Result<Self::Key, Self::Error>
    where
        T: AsRef<[N]>,
        U: IntoGeometry<Self::Facet>,
    {
        let keys = keys.as_ref();
        if keys.len() == A::USIZE {
            self.indices.extend(keys.iter());
            Ok(())
        }
        else {
            Err(BufferError::ArityConflict {
                expected: A::USIZE,
                actual: keys.len(),
            })
        }
    }
}

impl<A, G> FacetBuilder<A::Item> for BufferBuilder<NGon<A>, G>
where
    A: Array,
    A::Item: Copy + Hash + Integer + NumCast + Unsigned,
    NGon<A>: FromItems<Item = A::Item> + Grouping<Item = NGon<A>>,
    Vec<NGon<A>>: IndexBuffer<NGon<A>>,
{
    type Facet = ();
    type Key = ();

    fn insert_facet<T, U>(&mut self, keys: T, _: U) -> Result<Self::Key, Self::Error>
    where
        T: AsRef<[A::Item]>,
        U: IntoGeometry<Self::Facet>,
    {
        let keys = keys.as_ref();
        if keys.len() == A::CAPACITY {
            let ngon = NGon::<A>::from_items(keys.iter().cloned()).unwrap();
            self.indices.push(ngon);
            Ok(())
        }
        else {
            Err(BufferError::ArityConflict {
                expected: A::CAPACITY,
                actual: keys.len(),
            })
        }
    }
}

impl<K, G> FacetBuilder<K> for BufferBuilder<Polygon<K>, G>
where
    K: Copy + Hash + Integer + NumCast + Unsigned,
    Polygon<K>: Grouping<Item = Polygon<K>>,
    Vec<Polygon<K>>: IndexBuffer<Polygon<K>>,
{
    type Facet = ();
    type Key = ();

    fn insert_facet<T, U>(&mut self, keys: T, _: U) -> Result<Self::Key, Self::Error>
    where
        T: AsRef<[<Polygon<K> as Topological>::Vertex]>,
        U: IntoGeometry<Self::Facet>,
    {
        let keys = keys.as_ref();
        let polygon = match keys.len() {
            3 => Ok(Polygon::N3(NGon::from_items(keys.iter().cloned()).unwrap())),
            4 => Ok(Polygon::N4(NGon::from_items(keys.iter().cloned()).unwrap())),
            _ => Err(BufferError::ArityConflict {
                expected: 0, // TODO: Cannot report a non-uniform arity.
                actual: keys.len(),
            }),
        }?;
        self.indices.push(polygon);
        Ok(())
    }
}

impl<R, G> MeshBuilder for BufferBuilder<R, G>
where
    Self: SurfaceBuilder<Vertex = G, Facet = ()>,
    R: Grouping,
    VertexKey<R>: Hash,
    Vec<R::Item>: IndexBuffer<R>,
{
    type Builder = Self;

    type Vertex = G;
    type Facet = ();

    fn surface_with<F, T, E>(&mut self, f: F) -> Result<T, Self::Error>
    where
        Self::Error: From<E>,
        F: FnOnce(&mut Self::Builder) -> Result<T, E>,
    {
        f(self).map_err(|error| error.into())
    }
}

impl<R, G> SurfaceBuilder for BufferBuilder<R, G>
where
    Self: FacetBuilder<VertexKey<R>, Facet = ()>,
    R: Grouping,
    VertexKey<R>: Hash,
    Vec<R::Item>: IndexBuffer<R>,
{
    type Builder = Self;
    type Key = VertexKey<R>;

    type Vertex = G;
    type Facet = ();

    fn facets_with<F, T, E>(&mut self, f: F) -> Result<T, Self::Error>
    where
        Self::Error: From<E>,
        F: FnOnce(&mut Self::Builder) -> Result<T, E>,
    {
        f(self).map_err(|error| error.into())
    }

    fn insert_vertex<T>(&mut self, geometry: T) -> Result<Self::Key, Self::Error>
    where
        T: IntoGeometry<Self::Vertex>,
    {
        let key = <VertexKey<R> as NumCast>::from(self.vertices.len()).unwrap();
        self.vertices.push(geometry.into_geometry());
        Ok(key)
    }
}

impl<R, G> Transact<<Self as ClosedInput>::Input> for BufferBuilder<R, G>
where
    R: Grouping,
    Vec<R::Item>: IndexBuffer<R>,
{
    type Output = MeshBuffer<R, G>;
    type Error = BufferError;

    fn commit(self) -> Result<Self::Output, Self::Error> {
        let BufferBuilder { indices, vertices } = self;
        Ok(MeshBuffer::from_raw_buffers_unchecked(indices, vertices))
    }
}
