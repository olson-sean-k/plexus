//! Higher order geometric traits.
//!
//! This module defines higher order traits for operations on a mesh. It also
//! provides aliases for geometric types to improve readability of type
//! constraints.

use failure::Error;
use std::ops::{Add, Sub};

use self::alias::*;
use geometry::convert::AsPosition;
use geometry::ops::{Average, Cross, Interpolate, Normalize, Project};
use geometry::Geometry;
use graph::mesh::Mesh;
use graph::view::{Consistent, EdgeView, FaceView, ReadStorage};

pub trait FaceNormal: Geometry {
    type Normal;

    fn normal<M>(face: &FaceView<M, Self, Consistent>) -> Result<Self::Normal, Error>
    where
        M: AsRef<Mesh<Self>> + ReadStorage<Self>;
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

    fn normal<M>(face: &FaceView<M, Self, Consistent>) -> Result<Self::Normal, Error>
    where
        M: AsRef<Mesh<Self>> + ReadStorage<Self>,
    {
        let positions = face
            .vertices()
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

    fn centroid<M>(face: &FaceView<M, Self, Consistent>) -> Result<Self::Centroid, Error>
    where
        M: AsRef<Mesh<Self>> + ReadStorage<Self>;
}

impl<G> FaceCentroid for G
where
    G: Geometry,
    G::Vertex: Average,
{
    type Centroid = G::Vertex;

    fn centroid<M>(face: &FaceView<M, Self, Consistent>) -> Result<Self::Centroid, Error>
    where
        M: AsRef<Mesh<Self>> + ReadStorage<Self>,
    {
        Ok(G::Vertex::average(
            face.vertices().map(|vertex| vertex.geometry.clone()),
        ))
    }
}

pub trait EdgeMidpoint: Geometry {
    type Midpoint;

    fn midpoint<M>(edge: &EdgeView<M, Self, Consistent>) -> Result<Self::Midpoint, Error>
    where
        M: AsRef<Mesh<Self>> + ReadStorage<Self>;
}

impl<G> EdgeMidpoint for G
where
    G: Geometry,
    G::Vertex: AsPosition,
    VertexPosition<G>: Clone + Interpolate,
{
    type Midpoint = <VertexPosition<G> as Interpolate>::Output;

    fn midpoint<M>(edge: &EdgeView<M, Self, Consistent>) -> Result<Self::Midpoint, Error>
    where
        M: AsRef<Mesh<Self>> + ReadStorage<Self>,
    {
        let a = edge.source_vertex().geometry.as_position().clone();
        let b = edge.destination_vertex().geometry.as_position().clone();
        Ok(a.midpoint(b))
    }
}

pub trait EdgeLateral: Geometry {
    type Lateral;

    fn lateral<M>(edge: &EdgeView<M, Self, Consistent>) -> Result<Self::Lateral, Error>
    where
        M: AsRef<Mesh<Self>> + ReadStorage<Self>;
}

impl<G> EdgeLateral for G
where
    G: Geometry,
    G::Vertex: AsPosition,
    VertexPosition<G>: Clone
        + Add<
            <<VertexPosition<G> as Sub>::Output as Project>::Output,
            Output = VertexPosition<G>,
        >
        + Sub,
    <VertexPosition<G> as Sub>::Output: Normalize + Project,
{
    type Lateral = <VertexPosition<G> as Sub>::Output;

    fn lateral<M>(edge: &EdgeView<M, Self, Consistent>) -> Result<Self::Lateral, Error>
    where
        M: AsRef<Mesh<Self>> + ReadStorage<Self>,
    {
        let a = edge.source_vertex().geometry.as_position().clone();
        let b = edge.destination_vertex().geometry.as_position().clone();
        let c = edge
            .opposite_edge()
            .previous_edge()
            .destination_vertex()
            .geometry
            .as_position()
            .clone();
        let ab = a - b.clone();
        let cb = c.clone() - b.clone();
        let p = b + ab.project(cb);
        Ok((p - c).normalize())
    }
}

pub mod alias {
    use std::ops::Mul;

    use super::*;

    pub type VertexPosition<G> = <<G as Geometry>::Vertex as AsPosition>::Target;
    pub type ScaledFaceNormal<G, T> = <<G as FaceNormal>::Normal as Mul<T>>::Output;
    pub type ScaledEdgeLateral<G, T> = <<G as EdgeLateral>::Lateral as Mul<T>>::Output;
}
