//! Geometric traits.
//!
//! This module provides traits that specify the geometry of `MeshGraph`s and
//! express the spatial operations supported by that geometry. The most basic
//! and only required trait is `Geometry`, which uses associated types to
//! specify the type of the `geometry` field in the payloads for vertices,
//! arcs, edges, and faces in a graph.
//!
//! To support useful spatial operations, types that implement `Geometry` may
//! also implement `AsPosition`. The `AsPosition` trait exposes positional data
//! in vertices. If that positional data also implements spatial traits from
//! the [theon](https://crates.io/crates/theon) crate, then spatial operations
//! will be enabled, such as `smooth`, `split_at_midpoint`, and
//! `poke_with_offset`.
//!
//! This module also defines traits that define the capabilities of geometry in
//! a `MeshGraph`. These traits enable generic code without the need to express
//! complicated relationships between types representing a Euclidean space. For
//! example, the `FaceCentroid` trait is defined for `Geometry` types that
//! expose positional data that implements the necessary traits to compute the
//! centroid of a face.
//!
//! # Examples
//!
//! A function that subdivides faces in a graph by splitting edges at their
//! midpoints:
//!
//! ```rust
//! # extern crate plexus;
//! # extern crate smallvec;
//! #
//! use plexus::graph::{AsPosition, EdgeMidpoint, FaceView, GraphGeometry, MeshGraph, VertexPosition};
//! use plexus::prelude::*;
//! use smallvec::SmallVec;
//!
//! # fn main() {
//! // Requires `EdgeMidpoint` for `split_at_midpoint`.
//! pub fn circumscribe<G>(face: FaceView<&mut MeshGraph<G>, G>) -> FaceView<&mut MeshGraph<G>, G>
//! where
//!     G: EdgeMidpoint<Midpoint = VertexPosition<G>> + GraphGeometry,
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

// TODO: Integrate this module documentation into the `graph` module.

use theon::ops::{Cross, Interpolate, Project};
use theon::space::{EuclideanSpace, InnerSpace, Vector};

use crate::graph::borrow::Reborrow;
use crate::graph::payload::{ArcPayload, EdgePayload, FacePayload, VertexPayload};
use crate::graph::storage::AsStorage;
use crate::graph::view::edge::{ArcView, EdgeView};
use crate::graph::view::face::FaceView;
use crate::graph::view::vertex::VertexView;
use crate::graph::GraphError;

pub type VertexPosition<G> = <<G as GraphGeometry>::Vertex as AsPosition>::Target;

/// Graph geometry.
///
/// Specifies the types used to represent geometry for vertices, arcs, edges,
/// and faces in a graph. Arbitrary types can be used, including `()` for no
/// geometry at all.
///
/// Geometry is vertex-based. Geometric operations depend on understanding the
/// positional data in vertices exposed by the `AsPosition` trait.
///
/// # Examples
///
/// ```rust
/// # extern crate decorum;
/// # extern crate nalgebra;
/// # extern crate num;
/// # extern crate plexus;
/// #
/// use decorum::N64;
/// use nalgebra::{Point3, Vector4};
/// use num::Zero;
/// use plexus::graph::{AsPosition, GraphGeometry, MeshGraph};
/// use plexus::prelude::*;
/// use plexus::primitive::sphere::UvSphere;
/// use plexus::IntoGeometry;
///
/// // Vertex-only geometry with position and color data.
/// #[derive(Clone, Copy, Eq, Hash, PartialEq)]
/// pub struct Vertex {
///     pub position: Point3<N64>,
///     pub color: Vector4<N64>,
/// }
///
/// impl GraphGeometry for Vertex {
///     type Vertex = Self;
///     type Arc = ();
///     type Edge = ();
///     type Face = ();
/// }
///
/// impl AsPosition for Vertex {
///     type Target = Point3<N64>;
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
///     .polygons_with_position::<Point3<N64>>()
///     .map_vertices(|position| Vertex {
///         position,
///         color: Zero::zero(),
///     })
///     .collect::<MeshGraph<Vertex>>();
/// # }
/// ```
pub trait GraphGeometry: Sized {
    type Vertex: Clone;
    type Arc: Clone + Default;
    type Edge: Clone + Default;
    type Face: Clone + Default;
}

impl GraphGeometry for () {
    type Vertex = ();
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> GraphGeometry for (T, T, T)
where
    T: Clone,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

/// Exposes positional data in vertex geometry.
///
/// To enable geometric features, this trait must be implemented for the type
/// representing vertex data. Additionally, geometric operations should be
/// implemented for the `Target` type. See the
/// [theon](https://crates.io/crates/theon) crate.
pub trait AsPosition {
    type Target;

    fn as_position(&self) -> &Self::Target;
    fn as_position_mut(&mut self) -> &mut Self::Target;
}

pub trait FaceNormal: GraphGeometry {
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
    G: GraphGeometry,
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

pub trait VertexCentroid: GraphGeometry {
    type Centroid;

    fn centroid<M>(vertex: VertexView<M, Self>) -> Result<Self::Centroid, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<Self>> + AsStorage<VertexPayload<Self>>;
}

impl<G> VertexCentroid for G
where
    G: GraphGeometry,
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

pub trait FaceCentroid: GraphGeometry {
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
    G: GraphGeometry,
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

pub trait EdgeMidpoint: GraphGeometry {
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
    G: GraphGeometry,
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

pub trait ArcNormal: GraphGeometry {
    type Normal;

    fn normal<M>(arc: ArcView<M, Self>) -> Result<Self::Normal, GraphError>
    where
        M: Reborrow,
        M::Target: AsStorage<ArcPayload<Self>> + AsStorage<VertexPayload<Self>>;
}

impl<G> ArcNormal for G
where
    G: GraphGeometry,
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
