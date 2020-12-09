use crate::builder::{FacetBuilder, MeshBuilder, SurfaceBuilder};
use crate::geometry::{FromGeometry, IntoGeometry};
use crate::graph::data::GraphData;
use crate::graph::face::FaceKey;
use crate::graph::mutation::face::{self, FaceInsertCache};
use crate::graph::mutation::vertex;
use crate::graph::mutation::{Immediate, Mutation};
use crate::graph::vertex::VertexKey;
use crate::graph::{GraphError, MeshGraph};
use crate::transact::{ClosedInput, Transact};

pub struct GraphBuilder<G>
where
    G: GraphData,
{
    mutation: Mutation<Immediate<MeshGraph<G>>>,
}

impl<G> Default for GraphBuilder<G>
where
    G: GraphData,
{
    fn default() -> Self {
        GraphBuilder {
            mutation: Mutation::from(MeshGraph::default()),
        }
    }
}

impl<G> ClosedInput for GraphBuilder<G>
where
    G: GraphData,
{
    type Input = ();
}

impl<G> MeshBuilder for GraphBuilder<G>
where
    G: GraphData,
{
    type Builder = Self;

    type Vertex = G::Vertex;
    type Facet = G::Face;

    fn surface_with<F, T, E>(&mut self, f: F) -> Result<T, Self::Error>
    where
        Self::Error: From<E>,
        F: FnOnce(&mut Self::Builder) -> Result<T, E>,
    {
        f(self).map_err(|error| error.into())
    }
}

impl<G> Transact<<Self as ClosedInput>::Input> for GraphBuilder<G>
where
    G: GraphData,
{
    type Commit = MeshGraph<G>;
    type Abort = ();
    type Error = GraphError;

    fn commit(self) -> Result<Self::Commit, (Self::Abort, Self::Error)> {
        let GraphBuilder { mutation } = self;
        mutation.commit()
    }

    fn abort(self) -> Self::Abort {}
}

impl<G> SurfaceBuilder for GraphBuilder<G>
where
    G: GraphData,
{
    type Builder = Self;
    type Key = VertexKey;

    type Vertex = G::Vertex;
    type Facet = G::Face;

    fn facets_with<F, T, E>(&mut self, f: F) -> Result<T, Self::Error>
    where
        Self::Error: From<E>,
        F: FnOnce(&mut Self::Builder) -> Result<T, E>,
    {
        f(self).map_err(|error| error.into())
    }

    fn insert_vertex<T>(&mut self, data: T) -> Result<Self::Key, Self::Error>
    where
        Self::Vertex: FromGeometry<T>,
    {
        Ok(vertex::insert(&mut self.mutation, data.into_geometry()))
    }
}

impl<G> FacetBuilder<VertexKey> for GraphBuilder<G>
where
    G: GraphData,
{
    type Facet = G::Face;
    type Key = FaceKey;

    fn insert_facet<T, U>(&mut self, keys: T, data: U) -> Result<Self::Key, Self::Error>
    where
        Self::Facet: FromGeometry<U>,
        T: AsRef<[VertexKey]>,
    {
        let cache = FaceInsertCache::from_storage(&self.mutation, keys.as_ref())?;
        let data = data.into_geometry();
        face::insert_with(&mut self.mutation, cache, || (Default::default(), data))
    }
}
