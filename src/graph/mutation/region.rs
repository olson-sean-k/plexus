use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use crate::geometry::Geometry;
use crate::graph::container::Reborrow;
use crate::graph::storage::convert::alias::*;
use crate::graph::storage::convert::AsStorage;
use crate::graph::storage::{FaceKey, HalfKey, VertexKey};
use crate::graph::topology::{Face, Half, Vertex};
use crate::graph::view::convert::FromKeyedSource;
use crate::graph::view::VertexView;
use crate::graph::GraphError;
use crate::IteratorExt;

// TODO: This type needs some serious refactoring. Here are a few important
//       points to keep in mind:
//
//       1. Some notion of a region may become an important part of the public
//          API. A `RegionView` type may even be useful.
//       2. There are two similar but distinct notions of a region that must be
//          considered: a vertex-bounded region (which could encompass multiple
//          faces) and a "potential face", which is conceptually a face that
//          hasn't yet been inserted into a graph.
//
//       The API for `Region` is currently a bit of a mess. It's primary use is
//       to provide a basic API for "potential faces" and singularity detection
//       for insertions and removals.

/// Vertex-bounded region connectivity.
///
/// Describes the per-vertex half-edge connectivity of a region bounded by a
/// set of vertices. This is primarily used to connect exterior half-edges when
/// a face is inserted into a mesh.
pub type Connectivity = HashMap<VertexKey, Vec<HalfKey>>;

pub type Singularity = (VertexKey, Vec<FaceKey>);

/// Vertex-bounded region.
///
/// A unique set of vertices that forms a bounded region that can represent a
/// face in a mesh.
#[derive(Clone, Copy, Debug)]
pub struct Region<'a, M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Half<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    storage: M,
    vertices: &'a [VertexKey],
    face: Option<FaceKey>,
    phantom: PhantomData<G>,
}

impl<'a, M, G> Region<'a, M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Half<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub fn from_keyed_storage(vertices: &'a [VertexKey], storage: M) -> Result<Self, GraphError> {
        // A face requires at least three vertices (edges). This invariant
        // should be maintained by any code that is able to mutate the mesh,
        // such that code manipulating faces (via `FaceView`) may assume this
        // is true. Panics resulting from faces with fewer than three vertices
        // are bugs.
        if vertices.len() < 3 {
            // Non-polygonal arity.
            return Err(GraphError::TopologyMalformed);
        }
        if vertices.len() != vertices.iter().unique().count() {
            // Non-manifold bounds.
            return Err(GraphError::TopologyMalformed);
        }
        // Fail if any vertex is not present.
        if vertices
            .iter()
            .any(|vertex| !storage.reborrow().as_vertex_storage().contains_key(vertex))
        {
            return Err(GraphError::TopologyNotFound);
        }
        let faces = vertices
            .iter()
            .cloned()
            .perimeter()
            .flat_map(|span| storage.reborrow().as_half_storage().get(&span.into()))
            .flat_map(|half| half.face)
            .collect::<HashSet<_>>();
        // Fail if the half-edges refer to more than one face.
        if faces.len() > 1 {
            // Non-closed region.
            return Err(GraphError::TopologyMalformed);
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
    M: Reborrow,
    M::Target: AsStorage<Half<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
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
                    VertexView::from_keyed_source((key, self.storage.reborrow()))
                        .unwrap()
                        .reachable_incoming_halves()
                        .flat_map(|half| half.into_reachable_opposite_half())
                        .map(|half| half.key())
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
                    VertexView::from_keyed_source((key, self.storage.reborrow()))
                        .unwrap()
                        .reachable_incoming_halves()
                        .map(|half| half.key())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<HashMap<_, _>>();
        // If only one vertex has any outgoing half-edges, then this face
        // shares exactly one vertex with other faces. The vertex is a
        // singularity and forms a pinwheel.
        //
        // TODO: Half-edge graphs can support pinwheels and this topology is
        //       common. If possible, remove this detection mechanism. If this
        //       data is used to reject pinwheel formations, remove that as
        //       well.
        //
        //       Note that this data is still used when `FaceMutation`s are
        //       committed to detect and reject unreachable faces.
        let singularity = {
            let mut outgoing = outgoing.iter().filter(|&(_, halves)| !halves.is_empty());
            if let Some((&vertex, _)) = outgoing.next() {
                outgoing.next().map_or_else(
                    || {
                        let faces =
                            VertexView::from_keyed_source((vertex, self.storage.reborrow()))
                                .unwrap()
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
