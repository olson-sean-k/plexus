//! Higher order geometric traits.
//!
//! This module defines higher order traits for operations on a mesh. It also
//! provides aliases for geometric types to improve readability of type
//! constraints.

use failure::Error;
use std::ops::Sub;

use geometry::Geometry;
use geometry::convert::AsPosition;
use geometry::ops::{Average, Cross, Interpolate, Normalize};
use graph::GraphError;
use graph::topology::{EdgeRef, FaceRef};
use self::alias::*;

pub trait FaceNormal: Geometry {
    type Normal;

    fn normal(face: FaceRef<Self>) -> Result<Self::Normal, Error>;
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

    fn normal(face: FaceRef<Self>) -> Result<Self::Normal, Error> {
        let positions = face.vertices()
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

    fn centroid(face: FaceRef<Self>) -> Result<Self::Centroid, Error>;
}

impl<G> FaceCentroid for G
where
    G: Geometry,
    G::Vertex: Average,
{
    type Centroid = G::Vertex;

    fn centroid(face: FaceRef<Self>) -> Result<Self::Centroid, Error> {
        Ok(G::Vertex::average(
            face.vertices().map(|vertex| vertex.geometry.clone()),
        ))
    }
}

pub trait EdgeMidpoint: Geometry {
    type Midpoint;

    fn midpoint(edge: EdgeRef<Self>) -> Result<Self::Midpoint, Error>;
}

impl<G> EdgeMidpoint for G
where
    G: Geometry,
    G::Vertex: AsPosition,
    VertexPosition<G>: Clone + Interpolate,
{
    type Midpoint = <VertexPosition<G> as Interpolate>::Output;

    fn midpoint(edge: EdgeRef<Self>) -> Result<Self::Midpoint, Error> {
        let a = edge.source_vertex().geometry.as_position().clone();
        let b = edge.destination_vertex().geometry.as_position().clone();
        Ok(a.midpoint(b))
    }
}

pub trait EdgeLateral: Geometry {
    type Lateral;

    fn lateral(edge: EdgeRef<Self>) -> Result<Self::Lateral, Error>;
}

impl<G> EdgeLateral for G
where
    G: FaceNormal + Geometry,
    G::Vertex: AsPosition,
    VertexPosition<G>: Clone + Sub,
    <VertexPosition<G> as Sub>::Output: Cross,
    <VertexPosition<G> as Sub>::Output: Cross<<G as FaceNormal>::Normal>,
    <<VertexPosition<G> as Sub>::Output as Cross>::Output: Normalize,
    <<VertexPosition<G> as Sub>::Output as Cross<<G as FaceNormal>::Normal>>::Output: Normalize,
{
    type Lateral = <<VertexPosition<G> as Sub>::Output as Cross<<G as FaceNormal>::Normal>>::Output;

    fn lateral(edge: EdgeRef<Self>) -> Result<Self::Lateral, Error> {
        let a = edge.source_vertex().geometry.as_position().clone();
        let b = edge.destination_vertex().geometry.as_position().clone();
        let ab = a - b;
        let normal = <G as FaceNormal>::normal(edge.into_opposite_edge()
            .into_face()
            .ok_or_else(|| Error::from(GraphError::TopologyNotFound))?)?;
        Ok(ab.cross(normal).normalize())
    }
}

pub mod alias {
    use std::ops::Mul;

    use super::*;

    pub type VertexPosition<G> = <<G as Geometry>::Vertex as AsPosition>::Target;
    pub type ScaledFaceNormal<G, T> = <<G as FaceNormal>::Normal as Mul<T>>::Output;
    pub type ScaledEdgeLateral<G, T> = <<G as EdgeLateral>::Lateral as Mul<T>>::Output;
}
