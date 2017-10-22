/// Higher order geometric traits.
///
/// This module defines higher order traits for operations on a mesh. It also
/// provides aliases for geometric types to improve readability of type
/// constraints.

use std::ops::Sub;

use geometry::Geometry;
use geometry::convert::AsPosition;
use geometry::ops::{Average, Cross, Normalize};
use graph::topology::FaceRef;
use self::alias::*;

pub trait FaceNormal: Geometry {
    type Normal;

    fn normal(face: FaceRef<Self>) -> Result<Self::Normal, ()>;
}

impl<G> FaceNormal for G
where
    G: Geometry,
    G::Vertex: AsPosition,
    VertexPosition<G>: Clone + Sub,
    <VertexPosition<G> as Sub>::Output: Cross,
    <<VertexPosition<G> as Sub>::Output as Cross>::Output: Normalize,
{
    type Normal = <<<G::Vertex as AsPosition>::Target as Sub>::Output as Cross>::Output;

    fn normal(face: FaceRef<Self>) -> Result<Self::Normal, ()> {
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

    fn centroid(face: FaceRef<Self>) -> Result<Self::Centroid, ()>;
}

impl<G> FaceCentroid for G
where
    G: Geometry,
    G::Vertex: Average,
{
    type Centroid = G::Vertex;

    fn centroid(face: FaceRef<Self>) -> Result<Self::Centroid, ()> {
        Ok(G::Vertex::average(
            face.vertices().map(|vertex| vertex.geometry.clone()),
        ))
    }
}

pub mod alias {
    use std::ops::Mul;

    use super::*;

    pub type VertexPosition<G> =
        <<G as Geometry>::Vertex as AsPosition>::Target;
    pub type ScaledFaceNormal<G, T> =
        <<G as FaceNormal>::Normal as Mul<T>>::Output;
}
