use crate::entity::borrow::Reborrow;

pub type Data<M> = <M as Parametric>::Data;

// TODO: Require `Clone` instead of `Copy` once non-`Copy` types are supported
//       by the slotmap crate. See https://github.com/orlp/slotmap/issues/27
/// Graph data.
///
/// Specifies the types used to represent data in vertices, arcs, edges, and
/// faces in a graph. Arbitrary types can be used, including `()` for no data at
/// all.
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
/// use plexus::geometry::{AsPosition, IntoGeometry};
/// use plexus::graph::{GraphData, MeshGraph};
/// use plexus::prelude::*;
/// use plexus::primitive::generate::Position;
/// use plexus::primitive::sphere::UvSphere;
///
/// // Vertex-only geometry with position and color data.
/// #[derive(Clone, Copy, Eq, Hash, PartialEq)]
/// pub struct Vertex {
///     pub position: Point3<N64>,
///     pub color: Vector4<N64>,
/// }
///
/// impl GraphData for Vertex {
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
pub trait GraphData: Sized {
    type Vertex: Copy;
    type Arc: Copy + Default;
    type Edge: Copy + Default;
    type Face: Copy + Default;
}

impl GraphData for () {
    type Vertex = ();
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> GraphData for (T, T)
where
    T: Copy,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> GraphData for (T, T, T)
where
    T: Copy,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> GraphData for [T; 2]
where
    T: Copy,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl<T> GraphData for [T; 3]
where
    T: Copy,
{
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

pub trait Parametric {
    type Data: GraphData;
}

impl<B> Parametric for B
where
    B: Reborrow,
    B::Target: Parametric,
{
    type Data = <B::Target as Parametric>::Data;
}
