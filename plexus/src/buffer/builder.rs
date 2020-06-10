use num::{Integer, NumCast, Unsigned};
use std::hash::Hash;
use typenum::{self, NonZero};

use crate::buffer::{BufferError, MeshBuffer};
use crate::builder::{FacetBuilder, MeshBuilder, SurfaceBuilder};
use crate::index::{Flat, Grouping, IndexBuffer};
use crate::primitive::Topological;
use crate::transact::{ClosedInput, Transact};
use crate::{Arity, IntoGeometry};

// TODO: It should not be possible to manufacture keys without placing
//       additional constraints on the type bounds of `FacetBuilder` (for
//       example, `FacetBuilder<Key = usize>`). Is it important to check for
//       out-of-bounds indices in `insert_facet`?

pub type VertexKey<R> = <Vec<<R as Grouping>::Group> as IndexBuffer<R>>::Index;

pub struct BufferBuilder<R, G>
where
    R: Grouping,
{
    indices: Vec<R::Group>,
    vertices: Vec<G>,
}

impl<R, G> Default for BufferBuilder<R, G>
where
    R: Grouping,
    Vec<R::Group>: IndexBuffer<R>,
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
    Vec<R::Group>: IndexBuffer<R>,
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

impl<P, G> FacetBuilder<P::Vertex> for BufferBuilder<P, G>
where
    P: Grouping<Group = P> + Topological,
    P::Vertex: Copy + Hash + Integer + NumCast + Unsigned,
    Vec<P>: IndexBuffer<P>,
{
    type Facet = ();
    type Key = ();

    fn insert_facet<T, U>(&mut self, keys: T, _: U) -> Result<Self::Key, Self::Error>
    where
        T: AsRef<[P::Vertex]>,
        U: IntoGeometry<Self::Facet>,
    {
        let arity = keys.as_ref().len();
        P::try_from_slice(keys)
            .ok_or_else(|| BufferError::ArityConflict {
                expected: P::ARITY.into_interval().0,
                actual: arity,
            })
            .map(|polygon| self.indices.push(polygon))
    }
}

impl<R, G> MeshBuilder for BufferBuilder<R, G>
where
    Self: SurfaceBuilder<Vertex = G, Facet = ()>,
    R: Grouping,
    VertexKey<R>: Hash,
    Vec<R::Group>: IndexBuffer<R>,
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
    Vec<R::Group>: IndexBuffer<R>,
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
    Vec<R::Group>: IndexBuffer<R>,
{
    type Output = MeshBuffer<R, G>;
    type Error = BufferError;

    fn commit(self) -> Result<Self::Output, Self::Error> {
        let BufferBuilder { indices, vertices } = self;
        Ok(MeshBuffer::from_raw_buffers_unchecked(indices, vertices))
    }
}
