//! Higher order geometric traits.
//!
//! This module defines higher order traits for operations on a graph. It also
//! provides aliases for geometric types to improve readability of type
//! constraints. These traits can be used as contraints to prove to the
//! compiler that certain operations are supported without specifying
//! complicated relationships.
//!
//! The traits in this module have blanket implementations that apply when
//! certain geometric and operational traits are implemented. For example, if a
//! type implements `AsPosition` and the `Output` type of that implementation
//! also implements `Cross` and `Normalize`, then a `Geometry` using that type
//! as its `Vertex` attribute will likely implement the `FaceNormal` trait in
//! this module.
//!
//! # Examples
//!
//! A function that subdivides faces in a graph by splitting edges at their
//! midpoints:
//!
//! ```rust
//! # extern crate plexus;
//! # extern crate smallvec;
//! use plexus::geometry::alias::VertexPosition;
//! use plexus::geometry::convert::AsPosition;
//! use plexus::geometry::Geometry;
//! use plexus::graph::{EdgeMidpoint, FaceView, MeshGraph};
//! use plexus::prelude::*;
//! use smallvec::SmallVec;
//!
//! # fn main() {
//! // Requires `EdgeMidpoint` for `split_at_midpoint`.
//! pub fn circumscribe<G>(face: FaceView<&mut MeshGraph<G>, G>) -> FaceView<&mut MeshGraph<G>, G>
//! where
//!     G: EdgeMidpoint<Midpoint = VertexPosition<G>> + Geometry,
//!     G::Vertex: AsPosition,
//! {
//!     let arity = face.arity();
//!     let mut arc = face.into_arc();
//!     let mut splits = SmallVec::<[_; 4]>::with_capacity(arity);
//!     for _ in 0..arity {
//!         let vertex = arc.split_at_midpoint();
//!         splits.push(vertex.key());
//!         arc = vertex.into_outgoing_arc().into_next_arc();
//!     }
//!     let mut face = arc.into_face().unwrap();
//!     for (a, b) in splits.into_iter().perimeter() {
//!         face = face.split(ByKey(a), ByKey(b)).unwrap().into_face().unwrap();
//!     }
//!     face
//! }
//! # }
//! ```
//!
//! Geometric traits and primitives.
//!
//! These traits are used to support geometric operations in generators and
//! graphs. Implementing these traits implicitly implements internal traits,
//! which in turn enable geometric features.
//!
//! To use types as geometry in a `MeshGraph` only requires implementing the
//! `Geometry` trait. Implementing operations like `Cross` and `Normalize` for
//! the `Vertex` attribute enables geometric features like extrusion,
//! centroids, midpoints, etc.

use theon::ops::{Cross, Interpolate, Project};
use theon::space::{EuclideanSpace, InnerSpace, Vector};

use crate::graph::borrow::Reborrow;
use crate::graph::payload::{ArcPayload, EdgePayload, FacePayload, VertexPayload};
use crate::graph::storage::AsStorage;
use crate::graph::view::{ArcView, EdgeView, FaceView, VertexView};
use crate::graph::GraphError;

pub type VertexPosition<G> = <<G as Geometry>::Vertex as AsPosition>::Target;

/// Graph geometry.
///
/// Specifies the types used to represent geometry for vertices, arcs, edges,
/// and faces in a graph. Arbitrary types can be used, including `()` for no
/// geometry at all.
///
/// Geometry is vertex-based. Geometric operations depend on understanding the
/// positional data in vertices, and operational traits mostly apply to such
/// positional data.
///
/// # Examples
///
/// ```rust
/// # extern crate decorum;
/// # extern crate nalgebra;
/// # extern crate num;
/// # extern crate plexus;
/// use decorum::R64;
/// use nalgebra::{Point3, Vector4};
/// use num::One;
/// use plexus::geometry::convert::{AsPosition, IntoGeometry};
/// use plexus::geometry::Geometry;
/// use plexus::graph::MeshGraph;
/// use plexus::prelude::*;
/// use plexus::primitive::sphere::UvSphere;
///
/// // Vertex-only geometry with a position and color.
/// #[derive(Clone, Copy, Eq, Hash, PartialEq)]
/// pub struct VertexGeometry {
///     pub position: Point3<R64>,
///     pub color: Vector4<R64>,
/// }
///
/// impl Geometry for VertexGeometry {
///     type Vertex = Self;
///     type Arc = ();
///     type Edge = ();
///     type Face = ();
/// }
///
/// impl AsPosition for VertexGeometry {
///     type Target = Point3<R64>;
///
///     fn as_position(&self) -> &Self::Target {
///         &self.position
///     }
///
///     fn as_position_mut(&mut self) -> &mut Self::Target {
///         &mut self.position
///     }
/// }
///
/// # fn main() {
/// // Create a mesh from a sphere primitive and map the geometry data.
/// let mut graph = UvSphere::new(8, 8)
///     .polygons_with_position()
///     .map_vertices(|position| VertexGeometry {
///         position: position.into_geometry(),
///         color: Vector4::new(One::one(), One::one(), One::one(), One::one()),
///     })
///     .collect::<MeshGraph<VertexGeometry>>();
/// # }
/// ```
pub trait Geometry: Sized {
    type Vertex: Clone;
    type Arc: Clone + Default;
    type Edge: Clone + Default;
    type Face: Clone + Default;
}

impl Geometry for () {
    type Vertex = ();
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> Geometry for (T, T, T)
where
    T: Clone,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

/// Exposes a reference to positional vertex data.
///
/// To enable geometric features, this trait must be implemented for the type
/// representing vertex data. Additionally, geometric operations should be
/// implemented for the `Target` type.
pub trait AsPosition {
    type Target;

    fn as_position(&self) -> &Self::Target;
    fn as_position_mut(&mut self) -> &mut Self::Target;
}

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
        let ab = a - b;
        let bc = b - c;
        ab.cross(bc).normalize().ok_or_else(|| GraphError::Geometry)
    }
}

pub trait VertexCentroid: Geometry {
    type Centroid;

    fn centroid<M>(vertex: VertexView<M, Self>) -> Result<Self::Centroid, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<Self>> + AsStorage<VertexPayload<Self>>;
}

impl<G> VertexCentroid for G
where
    G: Geometry,
    G::Vertex: AsPosition,
    VertexPosition<G>: EuclideanSpace,
{
    type Centroid = VertexPosition<G>;

    fn centroid<M>(vertex: VertexView<M, Self>) -> Result<Self::Centroid, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<Self>> + AsStorage<VertexPayload<Self>>,
    {
        VertexPosition::<G>::centroid(
            vertex
                .reachable_incoming_arcs()
                .flat_map(|arc| arc.into_reachable_source_vertex())
                .map(|vertex| vertex.geometry.as_position().clone()),
        )
        .ok_or_else(|| GraphError::TopologyNotFound)
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
        VertexPosition::<G>::centroid(
            face.reachable_vertices()
                .map(|vertex| vertex.geometry.as_position().clone()),
        )
        .ok_or_else(|| GraphError::TopologyNotFound)
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
    VertexPosition<G>: EuclideanSpace + Interpolate<Output = VertexPosition<G>>,
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
            .reachable_next_arc()
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .reachable_destination_vertex()
            .ok_or_else(|| GraphError::TopologyNotFound)?
            .geometry
            .as_position()
            .clone();
        let ab = a - b;
        let cb = c - b;
        let p = b + ab.project(cb);
        (p - c).normalize().ok_or_else(|| GraphError::Geometry)
    }
}
