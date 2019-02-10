//! Higher order geometric traits.
//!
//! See the `geometry::compose` module. This module's contents are re-exported
//! there.

use std::ops::{Add, Sub};

use crate::geometry::alias::*;
use crate::geometry::convert::AsPosition;
use crate::geometry::ops::{Average, Cross, Interpolate, Normalize, Project};
use crate::geometry::Geometry;
use crate::graph::container::Reborrow;
use crate::graph::storage::convert::AsStorage;
use crate::graph::topology::{Arc, Edge, Face, Vertex};
use crate::graph::view::{ArcView, EdgeView, FaceView};
use crate::graph::GraphError;

// TODO: Some traits should operate directly on arcs instead of edges (and vice
//       versa.

pub trait FaceNormal: Geometry {
    type Normal;

    fn normal<M>(face: FaceView<M, Self>) -> Result<Self::Normal, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Arc<Self>> + AsStorage<Face<Self>> + AsStorage<Vertex<Self>>;
}

impl<G> FaceNormal for G
where
    G: Geometry,
    G::Vertex: AsPosition,
    VertexPosition<G>: Clone + Sub,
    <VertexPosition<G> as Sub>::Output: Cross,
    <<VertexPosition<G> as Sub>::Output as Cross>::Output: Normalize,
{
    type Normal = <<VertexPosition<G> as Sub>::Output as Cross>::Output;

    fn normal<M>(face: FaceView<M, Self>) -> Result<Self::Normal, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Arc<Self>> + AsStorage<Face<Self>> + AsStorage<Vertex<Self>>,
    {
        let positions = face
            .reachable_vertices()
            .take(3)
            .map(|vertex| vertex.geometry.as_position().clone())
            .collect::<Vec<_>>();
        let (a, b, c) = (&positions[0], &positions[1], &positions[2]);
        let ab = a.clone() - b.clone();
        let bc = b.clone() - c.clone();
        Ok(ab.cross(bc).normalize())
    }
}

pub trait FaceCentroid: Geometry {
    type Centroid;

    fn centroid<M>(face: FaceView<M, Self>) -> Result<Self::Centroid, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Arc<Self>> + AsStorage<Face<Self>> + AsStorage<Vertex<Self>>;
}

impl<G> FaceCentroid for G
where
    G: Geometry,
    G::Vertex: Average,
{
    type Centroid = G::Vertex;

    fn centroid<M>(face: FaceView<M, Self>) -> Result<Self::Centroid, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Arc<Self>> + AsStorage<Face<Self>> + AsStorage<Vertex<Self>>,
    {
        Ok(G::Vertex::average(
            face.reachable_vertices()
                .map(|vertex| vertex.geometry.clone()),
        ))
    }
}

pub trait EdgeMidpoint: Geometry {
    type Midpoint;

    fn midpoint<M>(edge: EdgeView<M, Self>) -> Result<Self::Midpoint, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Arc<Self>> + AsStorage<Edge<Self>> + AsStorage<Vertex<Self>>;
}

impl<G> EdgeMidpoint for G
where
    G: Geometry,
    G::Vertex: AsPosition,
    VertexPosition<G>: Clone + Interpolate,
{
    type Midpoint = <VertexPosition<G> as Interpolate>::Output;

    fn midpoint<M>(edge: EdgeView<M, Self>) -> Result<Self::Midpoint, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Arc<Self>> + AsStorage<Edge<Self>> + AsStorage<Vertex<Self>>,
    {
        let a = edge
            .reachable_arc()
            .ok_or_else(|| GraphError::TopologyNotFound)
            .and_then(|arc| {
                arc.reachable_source_vertex()
                    .ok_or_else(|| GraphError::TopologyNotFound)
                    .map(|vertex| vertex.geometry.as_position().clone())
            })?;
        let b = edge
            .reachable_arc()
            .ok_or_else(|| GraphError::TopologyNotFound)
            .and_then(|arc| {
                arc.reachable_destination_vertex()
                    .ok_or_else(|| GraphError::TopologyNotFound)
                    .map(|vertex| vertex.geometry.as_position().clone())
            })?;
        Ok(a.midpoint(b))
    }
}

pub trait ArcNormal: Geometry {
    type Normal;

    fn normal<M>(arc: ArcView<M, Self>) -> Result<Self::Normal, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Arc<Self>> + AsStorage<Vertex<Self>>;
}

impl<G> ArcNormal for G
where
    G: Geometry,
    G::Vertex: AsPosition,
    VertexPosition<G>: Clone
        + Add<
            <<VertexPosition<G> as Sub>::Output as Project>::Output,
            Output = VertexPosition<G>,
        > + Sub,
    <VertexPosition<G> as Sub>::Output: Normalize + Project,
{
    type Normal = <VertexPosition<G> as Sub>::Output;

    fn normal<M>(arc: ArcView<M, Self>) -> Result<Self::Normal, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<Arc<Self>> + AsStorage<Vertex<Self>>,
    {
        let a = arc
            .reachable_source_vertex()
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .geometry
            .as_position()
            .clone();
        let b = arc
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .geometry
            .as_position()
            .clone();
        let c = arc
            .reachable_opposite_arc()
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .reachable_previous_arc()
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .geometry
            .as_position()
            .clone();
        let ab = a - b.clone();
        let cb = c.clone() - b.clone();
        let p = b + ab.project(cb);
        Ok((p - c).normalize())
    }
}
