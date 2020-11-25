use smallvec::SmallVec;

use crate::entity::borrow::Reborrow;
use crate::entity::storage::AsStorage;
use crate::entity::view::Bind;
use crate::graph::data::{Data, GraphData, Parametric};
use crate::graph::edge::Arc;
use crate::graph::face::{Face, FaceKey};
use crate::graph::mutation::face::{self, FaceInsertCache};
use crate::graph::mutation::vertex;
use crate::graph::mutation::{Consistent, Mode, Mutable, Mutation};
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
        B::Target: AsStorage<Arc<Data<B>>>
            + AsStorage<Face<Data<B>>>
            + AsStorage<Vertex<Data<B>>>
            + Consistent
            + Parametric,
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

pub fn extrude_contour_with<N, P, F>(
    mut mutation: N,
    cache: PathExtrudeCache,
    f: F,
) -> Result<FaceKey, GraphError>
where
    N: AsMut<Mutation<P>>,
    P: Mode,
    P::Graph: Mutable,
    F: Fn(&<Data<P::Graph> as GraphData>::Vertex) -> <Data<P::Graph> as GraphData>::Vertex,
{
    let PathExtrudeCache { sources } = cache;
    let destinations: SmallVec<[_; 2]> = sources
        .iter()
        .cloned()
        .rev()
        .map(|source| -> Result<_, GraphError> {
            let vertex =
                VertexView::bind(mutation.as_mut(), source).ok_or(GraphError::TopologyNotFound)?;
            let data = f(vertex.get());
            Ok(vertex::insert(mutation.as_mut(), data))
        })
        .collect::<Result<_, _>>()?;
    let cache =
        FaceInsertCache::from_storage(mutation.as_mut(), sources.into_iter().chain(destinations))?;
    face::insert_with(mutation.as_mut(), cache, Default::default)
}
