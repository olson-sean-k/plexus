use smallvec::SmallVec;

use crate::entity::borrow::Reborrow;
use crate::entity::storage::AsStorage;
use crate::entity::view::Bind;
use crate::graph::edge::Arc;
use crate::graph::face::{Face, FaceKey};
use crate::graph::geometry::{Geometric, Geometry, GraphGeometry};
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
    let PathExtrudeCache { sources } = cache;
    let destinations: SmallVec<[_; 2]> = sources
        .iter()
        .cloned()
        .rev()
        .map(|source| -> Result<_, GraphError> {
            let geometry = VertexView::bind(mutation.as_mut(), source)
                .ok_or_else(|| GraphError::TopologyNotFound)?
                .geometry;
            Ok(vertex::insert(mutation.as_mut(), f(geometry)))
        })
        .collect::<Result<_, _>>()?;
    let cache =
        FaceInsertCache::from_storage(mutation.as_mut(), sources.into_iter().chain(destinations))?;
    face::insert_with(mutation.as_mut(), cache, Default::default)
}
