pub mod convert;
mod edge;
mod face;
mod vertex;

use crate::graph::storage::OpaqueKey;
use crate::graph::GraphError;

pub use self::edge::{ArcNeighborhood, ArcView, EdgeView, OrphanArcView, OrphanEdgeView};
pub use self::face::{FaceNeighborhood, FaceView, InteriorPathView, OrphanFaceView};
pub use self::vertex::{OrphanVertexView, VertexView};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Selector<K> {
    ByKey(K),
    ByIndex(usize),
}

impl<K> Selector<K> {
    pub fn key_or_else<E, F>(self, f: F) -> Result<K, GraphError>
    where
        E: Into<GraphError>,
        F: Fn(usize) -> Result<K, E>,
    {
        match self {
            Selector::ByKey(key) => Ok(key),
            Selector::ByIndex(index) => f(index).map_err(|error| error.into()),
        }
    }

    pub fn index_or_else<E, F>(self, f: F) -> Result<usize, GraphError>
    where
        E: Into<GraphError>,
        F: Fn(K) -> Result<usize, E>,
    {
        match self {
            Selector::ByKey(key) => f(key).map_err(|error| error.into()),
            Selector::ByIndex(index) => Ok(index),
        }
    }
}

impl<K> From<K> for Selector<K>
where
    K: OpaqueKey,
{
    fn from(key: K) -> Self {
        Selector::ByKey(key)
    }
}

impl<K> From<usize> for Selector<K> {
    fn from(index: usize) -> Self {
        Selector::ByIndex(index)
    }
}
