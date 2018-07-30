use failure::{Error, Fail};
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use geometry::Geometry;
use graph::storage::convert::AsStorage;
use graph::storage::{EdgeKey, FaceKey, VertexKey};
use graph::topology::{Edge, Face, Vertex};
use graph::view::{Inconsistent, VertexView};
use graph::{GraphError, Perimeter};

/// Vertex-bounded region connectivity.
///
/// Describes the per-vertex edge connectivity of a region bounded by a set of
/// vertices. This is primarily used to connect exterior edges when a face is
/// inserted into a mesh.
pub type Connectivity = HashMap<VertexKey, Vec<EdgeKey>>;

pub type Singularity = (VertexKey, Vec<FaceKey>);

/// Vertex-bounded region.
///
/// A unique set of vertices that forms a bounded region that can represent a
/// face in a mesh.
#[derive(Clone, Copy, Debug)]
pub struct Region<'a, M, G>
where
    M: AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    storage: M,
    vertices: &'a [VertexKey],
    face: Option<FaceKey>,
    phantom: PhantomData<G>,
}

impl<'a, M, G> Region<'a, M, G>
where
    M: AsStorage<Edge<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub fn from_keyed_storage(vertices: &'a [VertexKey], storage: M) -> Result<Self, Error> {
        // A face requires at least three vertices (edges). This invariant
        // should be maintained by any code that is able to mutate the mesh,
        // such that code manipulating faces (via `FaceView`) may assume this
        // is true. Panics resulting from faces with fewer than three vertices
        // are bugs.
        if vertices.len() < 3 {
            return Err(GraphError::TopologyMalformed
                .context("non-polygonal arity")
                .into());
        }
        if vertices.len() != vertices.iter().unique().count() {
            return Err(GraphError::TopologyMalformed
                .context("non-manifold bounds")
                .into());
        }
        // Fail if any vertex is not present.
        if vertices
            .iter()
            .any(|vertex| !AsStorage::<Vertex<G>>::as_storage(&storage).contains_key(vertex))
        {
            return Err(GraphError::TopologyNotFound.into());
        }
        let faces = vertices
            .perimeter()
            .flat_map(|ab| AsStorage::<Edge<G>>::as_storage(&storage).get(&ab.into()))
            .flat_map(|edge| edge.face)
            .collect::<HashSet<_>>();
        // Fail if the edges refer to more than one face.
        if faces.len() > 1 {
            return Err(GraphError::TopologyMalformed
                .context("non-closed region")
                .into());
        }
        Ok(Region {
            storage,
            vertices,
            face: faces.into_iter().next(),
            phantom: PhantomData,
        })
    }

    pub fn vertices(&self) -> &[VertexKey] {
        self.vertices
    }

    pub fn face(&self) -> &Option<FaceKey> {
        &self.face
    }
}

impl<'a, M, G> Region<'a, M, G>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub fn reachable_connectivity(&self) -> ((Connectivity, Connectivity), Option<Singularity>) {
        // Get the outgoing and incoming edges of the vertices forming the
        // perimeter.
        let outgoing = self
            .vertices
            .iter()
            .cloned()
            .map(|key| {
                (
                    key,
                    VertexView::<_, _, Inconsistent>::from_keyed_storage(key, &self.storage)
                        .unwrap()
                        .reachable_incoming_edges()
                        .flat_map(|edge| edge.into_reachable_opposite_edge())
                        .map(|edge| edge.key())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<HashMap<_, _>>();
        let incoming = self
            .vertices
            .iter()
            .cloned()
            .map(|key| {
                (
                    key,
                    VertexView::<_, _, Inconsistent>::from_keyed_storage(key, &self.storage)
                        .unwrap()
                        .reachable_incoming_edges()
                        .map(|edge| edge.key())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<HashMap<_, _>>();
        // If only one vertex has any outgoing edges, then this face shares
        // exactly one vertex with other faces and is therefore non-manifold.
        //
        // This kind of non-manifold is not supported, but sometimes occurs
        // during a batch mutation. Details of the singularity vertex are
        // emitted and handled by calling code, either raising an error or
        // waiting to validate after a batch mutation is complete.
        let singularity = {
            let mut outgoing = outgoing.iter().filter(|&(_, edges)| !edges.is_empty());
            if let Some((&vertex, _)) = outgoing.next() {
                outgoing.next().map_or_else(
                    || {
                        let faces = VertexView::<_, _, Inconsistent>::from_keyed_storage(
                            vertex,
                            &self.storage,
                        ).unwrap()
                            .reachable_neighboring_faces()
                            .map(|face| face.key())
                            .collect::<Vec<_>>();
                        Some((vertex, faces))
                    },
                    |_| None,
                )
            }
            else {
                None
            }
        };
        ((incoming, outgoing), singularity)
    }
}
