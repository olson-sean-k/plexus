use crate::builder::{FacetBuilder, MeshBuilder, SurfaceBuilder};
use crate::graph::geometry::GraphGeometry;
use crate::graph::key::{FaceKey, VertexKey};
use crate::graph::mutation::face::FaceInsertCache;
use crate::graph::mutation::Mutation;
use crate::graph::{GraphError, MeshGraph};
use crate::transact::{ClosedInput, Transact};
use crate::IntoGeometry;

pub struct GraphBuilder<G>
where
    G: GraphGeometry,
{
    mutation: Mutation<MeshGraph<G>>,
}

impl<G> Default for GraphBuilder<G>
where
    G: GraphGeometry,
{
    fn default() -> Self {
        GraphBuilder {
            mutation: Mutation::from(MeshGraph::default()),
        }
    }
}

impl<G> ClosedInput for GraphBuilder<G>
where
    G: GraphGeometry,
{
    type Input = ();
}

impl<G> MeshBuilder for GraphBuilder<G>
where
    G: GraphGeometry,
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
    G: GraphGeometry,
{
    type Output = MeshGraph<G>;
    type Error = GraphError;

    fn commit(self) -> Result<Self::Output, Self::Error> {
        let GraphBuilder { mutation } = self;
        mutation.commit()
    }
}

impl<G> SurfaceBuilder for GraphBuilder<G>
where
    G: GraphGeometry,
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

    fn insert_vertex<T>(&mut self, geometry: T) -> Result<Self::Key, Self::Error>
    where
        T: IntoGeometry<Self::Vertex>,
    {
        Ok(self.mutation.insert_vertex(geometry.into_geometry()))
    }
}

impl<G> FacetBuilder<VertexKey> for GraphBuilder<G>
where
    G: GraphGeometry,
{
    type Facet = G::Face;
    type Key = FaceKey;

    fn insert_facet<T, U>(&mut self, keys: T, geometry: U) -> Result<Self::Key, Self::Error>
    where
        T: AsRef<[VertexKey]>,
        U: IntoGeometry<Self::Facet>,
    {
        let cache = FaceInsertCache::snapshot(&self.mutation, keys.as_ref())?;
        let geometry = geometry.into_geometry();
        self.mutation
            .insert_face_with(cache, || (Default::default(), geometry))
    }
}
