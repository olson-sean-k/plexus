//! Higher order geometric traits.
//!
//! See the `geometry::compose` module. This module's contents are re-exported
//! there.

use crate::geometry::alias::{Vector, VertexPosition};
use crate::geometry::convert::AsPosition;
use crate::geometry::ops::{Cross, Interpolate, Normalize, Project};
use crate::geometry::space::{EuclideanSpace, Origin};
use crate::geometry::Geometry;
use crate::graph::container::Reborrow;
use crate::graph::payload::{ArcPayload, EdgePayload, FacePayload, VertexPayload};
use crate::graph::storage::convert::AsStorage;
use crate::graph::view::{ArcView, EdgeView, FaceView};
use crate::graph::{GraphError, OptionExt};

pub trait FaceNormal: Geometry {
    type Normal;

    fn normal<M>(face: FaceView<M, Self>) -> Result<Self::Normal, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<Self>>
            + AsStorage<FacePayload<Self>>
            + AsStorage<VertexPayload<Self>>;
}

impl<G> FaceNormal for G
where
    G: Geometry,
    G::Vertex: AsPosition,
    Vector<VertexPosition<G>>: Cross<Output = Vector<VertexPosition<G>>>,
    VertexPosition<G>: EuclideanSpace,
{
    type Normal = Vector<VertexPosition<G>>;

    fn normal<M>(face: FaceView<M, Self>) -> Result<Self::Normal, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<Self>>
            + AsStorage<FacePayload<Self>>
            + AsStorage<VertexPayload<Self>>,
    {
        let mut positions = face
            .reachable_vertices()
            .take(3)
            .map(|vertex| vertex.geometry.as_position().clone());
        let a = positions
            .next()
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let b = positions
            .next()
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let c = positions
            .next()
            .ok_or_else(|| GraphError::TopologyNotFound)?;
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
        M::Target: AsStorage<ArcPayload<Self>>
            + AsStorage<FacePayload<Self>>
            + AsStorage<VertexPayload<Self>>;
}

impl<G> FaceCentroid for G
where
    G: Geometry,
    G::Vertex: AsPosition,
    VertexPosition<G>: EuclideanSpace,
{
    type Centroid = VertexPosition<G>;

    fn centroid<M>(face: FaceView<M, Self>) -> Result<Self::Centroid, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<Self>>
            + AsStorage<FacePayload<Self>>
            + AsStorage<VertexPayload<Self>>,
    {
        Ok(VertexPosition::<G>::centroid(
            face.reachable_vertices()
                .map(|vertex| vertex.geometry.as_position().clone()),
        )
        .expect_consistent())
    }
}

pub trait EdgeMidpoint: Geometry {
    type Midpoint;

    fn midpoint<M>(edge: EdgeView<M, Self>) -> Result<Self::Midpoint, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<Self>>
            + AsStorage<EdgePayload<Self>>
            + AsStorage<VertexPayload<Self>>;
}

impl<G> EdgeMidpoint for G
where
    G: Geometry,
    G::Vertex: AsPosition,
    VertexPosition<G>: EuclideanSpace,
{
    type Midpoint = VertexPosition<G>;

    fn midpoint<M>(edge: EdgeView<M, Self>) -> Result<Self::Midpoint, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<Self>>
            + AsStorage<EdgePayload<Self>>
            + AsStorage<VertexPayload<Self>>,
    {
        let a = edge
            .reachable_arc()
            .ok_or_else(|| GraphError::TopologyNotFound)
            .and_then(|arc| {
                arc.reachable_source_vertex()
                    .ok_or_else(|| GraphError::TopologyNotFound)
                    .map(|vertex| vertex.geometry.as_position().coordinates())
            })?;
        let b = edge
            .reachable_arc()
            .ok_or_else(|| GraphError::TopologyNotFound)
            .and_then(|arc| {
                arc.reachable_destination_vertex()
                    .ok_or_else(|| GraphError::TopologyNotFound)
                    .map(|vertex| vertex.geometry.as_position().coordinates())
            })?;
        Ok(VertexPosition::<G>::origin() + a.midpoint(b))
    }
}

pub trait ArcNormal: Geometry {
    type Normal;

    fn normal<M>(arc: ArcView<M, Self>) -> Result<Self::Normal, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<Self>> + AsStorage<VertexPayload<Self>>;
}

impl<G> ArcNormal for G
where
    G: Geometry,
    G::Vertex: AsPosition,
    Vector<VertexPosition<G>>: Project<Output = Vector<VertexPosition<G>>>,
    VertexPosition<G>: EuclideanSpace,
{
    type Normal = Vector<VertexPosition<G>>;

    fn normal<M>(arc: ArcView<M, Self>) -> Result<Self::Normal, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<Self>> + AsStorage<VertexPayload<Self>>,
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
