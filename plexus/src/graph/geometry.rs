//! Geometric traits.
//!
//! This module provides traits that specify the geometry of `MeshGraph`s and
//! express the spatial operations supported by that geometry. The most basic
//! and only required trait is `GraphGeometry`, which uses associated types to
//! specify the type of the `geometry` field in the payloads for vertices, arcs,
//! edges, and faces in a graph.
//!
//! To support useful spatial operations, types that implement `GraphGeometry`
//! may also implement `AsPosition`. The `AsPosition` trait exposes positional
//! data in vertices. If that positional data also implements spatial traits
//! from the [`theon`][1] crate, then spatial operations will be enabled, such
//! as `smooth`, `split_at_midpoint`, and `poke_with_offset`.
//!
//! This module also defines traits that define the capabilities of geometry in
//! a `MeshGraph`. These traits enable generic code without the need to express
//! complicated relationships between types representing a Euclidean space. For
//! example, the `FaceCentroid` trait is defined for `GraphGeometry` types that
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
//! use plexus::graph::{EdgeMidpoint, FaceView, GraphGeometry, MeshGraph};
//! use plexus::prelude::*;
//! use plexus::AsPosition;
//! use smallvec::SmallVec;
//!
//! // Requires `EdgeMidpoint` for `split_at_midpoint`.
//! pub fn circumscribe<G>(face: FaceView<&mut MeshGraph<G>>) -> FaceView<&mut MeshGraph<G>>
//! where
//!     G: EdgeMidpoint + GraphGeometry,
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
//! ```
//!
//! [1]: https://crates.io/crates/theon

// TODO: Integrate this module documentation into the `graph` module.

// Geometric traits like `FaceNormal` and `EdgeMidpoint` are defined in such a
// way to reduce the contraints necessary for writing generic user code. With
// few exceptions, these traits only depend on `AsPosition` being implemented by
// the `Vertex` type in their definition. If a more complex implementation is
// necessary, constraints are specified there so that they do not pollute user
// code.

use theon::adjunct::FromItems;
use theon::ops::{Cross, Interpolate, Project};
use theon::query::Plane;
use theon::space::{EuclideanSpace, FiniteDimensional, InnerSpace, Vector, VectorSpace};
use theon::{AsPosition, Position};
use typenum::U3;

use crate::graph::edge::{Arc, ArcView, Edge, Edgoid};
use crate::graph::face::{Face, Ringoid};
use crate::graph::mutation::Consistent;
use crate::graph::vertex::{Vertex, VertexView};
use crate::graph::{GraphError, OptionExt as _};
use crate::network::borrow::Reborrow;
use crate::network::storage::AsStorage;

pub type Geometry<M> = <M as Geometric>::Geometry;
pub type VertexPosition<G> = Position<<G as GraphGeometry>::Vertex>;

pub trait Geometric {
    type Geometry: GraphGeometry;
}

impl<B> Geometric for B
where
    B: Reborrow,
    B::Target: Geometric,
{
    type Geometry = <B::Target as Geometric>::Geometry;
}

// TODO: Require `Clone` instead of `Copy` once non-`Copy` types are supported
//       by the slotmap crate. See https://github.com/orlp/slotmap/issues/27
/// Graph geometry.
///
/// Specifies the types used to represent geometry for vertices, arcs, edges,
/// and faces in a graph. Arbitrary types can be used, including `()` for no
/// geometry at all.
///
/// Geometric operations depend on understanding the positional data in vertices
/// exposed by the `AsPosition` trait. If the `Vertex` type implements
/// `AsPosition`, then geometric operations supported by the `Position` type are
/// exposed by graph APIs.
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
/// use plexus::graph::{GraphGeometry, MeshGraph};
/// use plexus::prelude::*;
/// use plexus::primitive::generate::Position;
/// use plexus::primitive::sphere::UvSphere;
/// use plexus::{AsPosition, IntoGeometry};
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
///     type Position = Point3<N64>;
///
///     fn as_position(&self) -> &Self::Position {
///         &self.position
///     }
///
///     fn as_position_mut(&mut self) -> &mut Self::Position {
///         &mut self.position
///     }
/// }
///
/// // Create a mesh from a sphere primitive and map the geometry data.
/// let mut graph = UvSphere::new(8, 8)
///     .polygons::<Position<Point3<N64>>>()
///     .map_vertices(|position| Vertex {
///         position,
///         color: Zero::zero(),
///     })
///     .collect::<MeshGraph<Vertex>>();
/// ```
pub trait GraphGeometry: Sized {
    type Vertex: Copy;
    type Arc: Copy + Default;
    type Edge: Copy + Default;
    type Face: Copy + Default;
}

impl GraphGeometry for () {
    type Vertex = ();
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> GraphGeometry for (T, T)
where
    T: Copy,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> GraphGeometry for (T, T, T)
where
    T: Copy,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> GraphGeometry for [T; 2]
where
    T: Copy,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> GraphGeometry for [T; 3]
where
    T: Copy,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

pub trait VertexCentroid: GraphGeometry
where
    Self::Vertex: AsPosition,
{
    fn centroid<B>(vertex: VertexView<B>) -> Result<VertexPosition<Self>, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Self>>
            + AsStorage<Vertex<Self>>
            + Consistent
            + Geometric<Geometry = Self>;
}

impl<G> VertexCentroid for G
where
    G: GraphGeometry,
    G::Vertex: AsPosition,
{
    fn centroid<B>(vertex: VertexView<B>) -> Result<VertexPosition<Self>, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Self>>
            + AsStorage<Vertex<Self>>
            + Consistent
            + Geometric<Geometry = Self>,
    {
        Ok(VertexPosition::<Self>::centroid(
            vertex
                .neighboring_vertices()
                .map(|vertex| *vertex.geometry.as_position()),
        )
        .expect_consistent())
    }
}

pub trait VertexNormal: FaceNormal
where
    Self::Vertex: AsPosition,
{
    fn normal<B>(vertex: VertexView<B>) -> Result<Vector<VertexPosition<Self>>, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Self>>
            + AsStorage<Face<Self>>
            + AsStorage<Vertex<Self>>
            + Consistent
            + Geometric<Geometry = Self>;
}

impl<G> VertexNormal for G
where
    G: FaceNormal,
    G::Vertex: AsPosition,
{
    fn normal<B>(vertex: VertexView<B>) -> Result<Vector<VertexPosition<Self>>, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Self>>
            + AsStorage<Face<Self>>
            + AsStorage<Vertex<Self>>
            + Consistent
            + Geometric<Geometry = Self>,
    {
        Vector::<VertexPosition<Self>>::mean(
            vertex
                .neighboring_faces()
                .map(<Self as FaceNormal>::normal)
                .collect::<Result<Vec<_>, _>>()?,
        )
        .expect_consistent()
        .normalize()
        .ok_or_else(|| GraphError::Geometry)
    }
}

pub trait ArcNormal: GraphGeometry
where
    Self::Vertex: AsPosition,
{
    fn normal<B>(arc: ArcView<B>) -> Result<Vector<VertexPosition<Self>>, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Self>>
            + AsStorage<Vertex<Self>>
            + Consistent
            + Geometric<Geometry = Self>;
}

impl<G> ArcNormal for G
where
    G: GraphGeometry,
    G::Vertex: AsPosition,
    VertexPosition<G>: EuclideanSpace,
    Vector<VertexPosition<G>>: Project<Output = Vector<VertexPosition<G>>>,
{
    fn normal<B>(arc: ArcView<B>) -> Result<Vector<VertexPosition<Self>>, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Self>>
            + AsStorage<Vertex<Self>>
            + Consistent
            + Geometric<Geometry = Self>,
    {
        let (a, b) = FromItems::from_items(arc.vertices().map(|vertex| *vertex.position()))
            .expect_consistent();
        let c = *arc.next_arc().destination_vertex().position();
        let ab = a - b;
        let cb = c - b;
        let p = b + ab.project(cb);
        (p - c).normalize().ok_or_else(|| GraphError::Geometry)
    }
}

pub trait EdgeMidpoint: GraphGeometry
where
    Self::Vertex: AsPosition,
{
    fn midpoint<T, B>(edge: T) -> Result<VertexPosition<Self>, GraphError>
    where
        T: Edgoid<B>,
        B: Reborrow,
        B::Target: AsStorage<Arc<Self>>
            + AsStorage<Edge<Self>>
            + AsStorage<Vertex<Self>>
            + Consistent
            + Geometric<Geometry = Self>;
}

impl<G> EdgeMidpoint for G
where
    G: GraphGeometry,
    G::Vertex: AsPosition,
    VertexPosition<G>: Interpolate<Output = VertexPosition<G>>,
{
    fn midpoint<T, B>(edge: T) -> Result<VertexPosition<Self>, GraphError>
    where
        T: Edgoid<B>,
        B: Reborrow,
        B::Target: AsStorage<Arc<Self>>
            + AsStorage<Edge<Self>>
            + AsStorage<Vertex<Self>>
            + Consistent
            + Geometric<Geometry = Self>,
    {
        let arc = edge.into_arc();
        let (a, b) =
            FromItems::from_items(arc.vertices().map(|vertex| *vertex.position())).unwrap();
        Ok(a.midpoint(b))
    }
}

pub trait FaceCentroid: GraphGeometry
where
    Self::Vertex: AsPosition,
{
    fn centroid<T, B>(ring: T) -> Result<VertexPosition<Self>, GraphError>
    where
        T: Ringoid<B>,
        B: Reborrow,
        B::Target: AsStorage<Arc<Self>>
            + AsStorage<Vertex<Self>>
            + Consistent
            + Geometric<Geometry = Self>;
}

impl<G> FaceCentroid for G
where
    G: GraphGeometry,
    G::Vertex: AsPosition,
{
    fn centroid<T, B>(ring: T) -> Result<VertexPosition<Self>, GraphError>
    where
        T: Ringoid<B>,
        B: Reborrow,
        B::Target: AsStorage<Arc<Self>>
            + AsStorage<Vertex<Self>>
            + Consistent
            + Geometric<Geometry = Self>,
    {
        let ring = ring.into_ring();
        Ok(
            VertexPosition::<Self>::centroid(ring.vertices().map(|vertex| *vertex.position()))
                .expect_consistent(),
        )
    }
}

pub trait FaceNormal: GraphGeometry
where
    Self::Vertex: AsPosition,
{
    fn normal<T, B>(ring: T) -> Result<Vector<VertexPosition<Self>>, GraphError>
    where
        T: Ringoid<B>,
        B: Reborrow,
        B::Target: AsStorage<Arc<Self>>
            + AsStorage<Vertex<Self>>
            + Consistent
            + Geometric<Geometry = Self>;
}

impl<G> FaceNormal for G
where
    G: FaceCentroid + GraphGeometry,
    G::Vertex: AsPosition,
    Vector<VertexPosition<G>>: Cross<Output = Vector<VertexPosition<G>>>,
    VertexPosition<G>: EuclideanSpace,
{
    fn normal<T, B>(ring: T) -> Result<Vector<VertexPosition<Self>>, GraphError>
    where
        T: Ringoid<B>,
        B: Reborrow,
        B::Target: AsStorage<Arc<Self>>
            + AsStorage<Vertex<Self>>
            + Consistent
            + Geometric<Geometry = Self>,
    {
        let ring = ring.into_ring();
        let (a, b) =
            FromItems::from_items(ring.vertices().take(2).map(|vertex| *vertex.position()))
                .expect_consistent();
        let c = G::centroid(ring)?;
        let ab = a - b;
        let bc = b - c;
        ab.cross(bc).normalize().ok_or_else(|| GraphError::Geometry)
    }
}

pub trait FacePlane: GraphGeometry
where
    Self::Vertex: AsPosition,
    VertexPosition<Self>: FiniteDimensional<N = U3>,
{
    fn plane<T, B>(ring: T) -> Result<Plane<VertexPosition<Self>>, GraphError>
    where
        T: Ringoid<B>,
        B: Reborrow,
        B::Target: AsStorage<Arc<Self>>
            + AsStorage<Vertex<Self>>
            + Consistent
            + Geometric<Geometry = Self>;
}

// TODO: The `lapack` feature only supports Linux. See
//       https://github.com/olson-sean-k/theon/issues/1
#[cfg(target_os = "linux")]
mod array {
    use super::*;

    use smallvec::SmallVec;
    use theon::adjunct::{FromItems, IntoItems};
    use theon::lapack::Lapack;
    use theon::space::Scalar;

    impl<G> FacePlane for G
    where
        G: GraphGeometry,
        G::Vertex: AsPosition,
        VertexPosition<G>: EuclideanSpace + FiniteDimensional<N = U3>,
        Scalar<VertexPosition<G>>: Lapack,
        Vector<VertexPosition<G>>: FromItems + IntoItems,
    {
        fn plane<T, B>(ring: T) -> Result<Plane<VertexPosition<G>>, GraphError>
        where
            T: Ringoid<B>,
            B: Reborrow,
            B::Target: AsStorage<Arc<Self>>
                + AsStorage<Vertex<Self>>
                + Consistent
                + Geometric<Geometry = Self>,
        {
            let ring = ring.into_ring();
            let points = ring
                .vertices()
                .map(|vertex| *vertex.geometry.as_position())
                .collect::<SmallVec<[_; 4]>>();
            Plane::from_points(points).ok_or_else(|| GraphError::Geometry)
        }
    }
}
