use itertools::Itertools;
use smallvec::SmallVec;

use crate::entity::borrow::Reborrow;
use crate::entity::storage::AsStorage;
use crate::entity::view::Bind;
use crate::graph::edge::Arc;
use crate::graph::face::{Face, FaceKey};
use crate::graph::geometry::{Geometric, Geometry, GraphGeometry};
use crate::graph::mutation::edge;
use crate::graph::mutation::face::{self, FaceInsertCache};
use crate::graph::mutation::vertex;
use crate::graph::mutation::{Consistent, Mutable, Mutation};
use crate::graph::path::Path;
use crate::graph::vertex::{Vertex, VertexKey, VertexView};
use crate::graph::GraphError;
use crate::IteratorExt as _;

pub struct PathExtrudeCache {
    // Avoid allocations for single arc extrusions.
    sources: SmallVec<[VertexKey; 2]>,
}

impl PathExtrudeCache {
    pub fn from_path<B>(path: Path<B>) -> Result<Self, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Geometry<B>>>
            + AsStorage<Face<Geometry<B>>>
            + AsStorage<Vertex<Geometry<B>>>
            + Consistent
            + Geometric,
    {
        if path.arcs().any(|arc| !arc.is_boundary_arc()) {
            Err(GraphError::TopologyMalformed)
        }
        else {
            Ok(PathExtrudeCache {
                sources: path.vertices().keys().collect(),
            })
        }
    }
}

pub fn extrude_with<M, N, F>(
    mut mutation: N,
    cache: PathExtrudeCache,
    f: F,
) -> Result<FaceKey, GraphError>
where
    N: AsMut<Mutation<M>>,
    M: Mutable,
    F: Fn(<Geometry<M> as GraphGeometry>::Vertex) -> <Geometry<M> as GraphGeometry>::Vertex,
{
    fn extrude_and_insert<M, N, F>(
        mut mutation: N,
        a: VertexKey,
        f: F,
    ) -> Result<VertexKey, GraphError>
    where
        N: AsMut<Mutation<M>>,
        M: Mutable,
        F: Fn(<Geometry<M> as GraphGeometry>::Vertex) -> <Geometry<M> as GraphGeometry>::Vertex,
    {
        let geometry = VertexView::bind(mutation.as_mut(), a)
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .geometry;
        Ok(vertex::insert(mutation.as_mut(), f(geometry)))
    }

    let PathExtrudeCache { sources } = cache;
    let mut destinations = SmallVec::<[_; 2]>::with_capacity(sources.len());
    for (a, b) in sources.iter().cloned().rev().tuple_windows() {
        let a = extrude_and_insert(mutation.as_mut(), a, &f)?;
        let b = extrude_and_insert(mutation.as_mut(), b, &f)?;
        let (_, (ab, _)) = edge::get_or_insert_with(mutation.as_mut(), (a, b), Default::default)?;
        destinations.push(ab);
    }
    let cache = FaceInsertCache::from_storage(
        mutation.as_mut(),
        sources.into_iter().chain(
            destinations
                .as_slice()
                .get(0)
                .map(|ab| ab.into_source())
                .into_iter()
                .chain(destinations.into_iter().map(|ab| ab.into_destination())),
        ),
    )?;
    face::insert_with(mutation.as_mut(), cache, Default::default)
}
