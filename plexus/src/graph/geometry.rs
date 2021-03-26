//! Geometric graph traits.

// Geometric traits like `FaceNormal` and `EdgeMidpoint` are defined in such a
// way to reduce the contraints necessary for writing generic user code. With
// few exceptions, these traits only depend on `AsPosition` being implemented by
// the `Vertex` type in their definition. If a more complex implementation is
// necessary, constraints are specified there so that they do not pollute user
// code.

use theon::ops::{Cross, Interpolate, Project};
use theon::query::Plane;
use theon::space::{EuclideanSpace, FiniteDimensional, InnerSpace, Vector, VectorSpace};
use theon::{AsPosition, Position};
use typenum::U3;

use crate::entity::borrow::Reborrow;
use crate::entity::storage::AsStorage;
use crate::graph::data::{GraphData, Parametric};
use crate::graph::edge::{Arc, ArcView, Edge, ToArc};
use crate::graph::face::{Face, ToRing};
use crate::graph::mutation::Consistent;
use crate::graph::vertex::{Vertex, VertexView};
use crate::graph::{GraphError, OptionExt as _, ResultExt as _};
use crate::IteratorExt as _;

pub type VertexPosition<G> = Position<<G as GraphData>::Vertex>;

pub trait VertexCentroid: GraphData
where
    Self::Vertex: AsPosition,
{
    fn centroid<B>(vertex: VertexView<B>) -> Result<VertexPosition<Self>, GraphError>
    where
        B: Reborrow,
        B::Target:
            AsStorage<Arc<Self>> + AsStorage<Vertex<Self>> + Consistent + Parametric<Data = Self>;
}

impl<G> VertexCentroid for G
where
    G: GraphData,
    G::Vertex: AsPosition,
{
    fn centroid<B>(vertex: VertexView<B>) -> Result<VertexPosition<Self>, GraphError>
    where
        B: Reborrow,
        B::Target:
            AsStorage<Arc<Self>> + AsStorage<Vertex<Self>> + Consistent + Parametric<Data = Self>,
    {
        Ok(VertexPosition::<Self>::centroid(
            vertex
                .adjacent_vertices()
                .map(|vertex| *vertex.data.as_position()),
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
            + Parametric<Data = Self>;
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
            + Parametric<Data = Self>,
    {
        Vector::<VertexPosition<Self>>::mean(
            vertex
                .adjacent_faces()
                .map(<Self as FaceNormal>::normal)
                .collect::<Result<Vec<_>, _>>()?,
        )
        .expect_consistent()
        .normalize()
        .ok_or(GraphError::Geometry)
    }
}

pub trait ArcNormal: GraphData
where
    Self::Vertex: AsPosition,
{
    fn normal<B>(arc: ArcView<B>) -> Result<Vector<VertexPosition<Self>>, GraphError>
    where
        B: Reborrow,
        B::Target:
            AsStorage<Arc<Self>> + AsStorage<Vertex<Self>> + Consistent + Parametric<Data = Self>;
}

impl<G> ArcNormal for G
where
    G: GraphData,
    G::Vertex: AsPosition,
    VertexPosition<G>: EuclideanSpace,
    Vector<VertexPosition<G>>: Project<Output = Vector<VertexPosition<G>>>,
{
    fn normal<B>(arc: ArcView<B>) -> Result<Vector<VertexPosition<Self>>, GraphError>
    where
        B: Reborrow,
        B::Target:
            AsStorage<Arc<Self>> + AsStorage<Vertex<Self>> + Consistent + Parametric<Data = Self>,
    {
        let (a, b) = arc
            .adjacent_vertices()
            .map(|vertex| *vertex.position())
            .try_collect()
            .expect_consistent();
        let c = *arc.next_arc().destination_vertex().position();
        let ab = a - b;
        let cb = c - b;
        let p = b + ab.project(cb);
        (p - c).normalize().ok_or(GraphError::Geometry)
    }
}

pub trait EdgeMidpoint: GraphData
where
    Self::Vertex: AsPosition,
{
    fn midpoint<B, T>(edge: T) -> Result<VertexPosition<Self>, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Self>>
            + AsStorage<Edge<Self>>
            + AsStorage<Vertex<Self>>
            + Consistent
            + Parametric<Data = Self>,
        T: ToArc<B>;
}

impl<G> EdgeMidpoint for G
where
    G: GraphData,
    G::Vertex: AsPosition,
    VertexPosition<G>: Interpolate<Output = VertexPosition<G>>,
{
    fn midpoint<B, T>(edge: T) -> Result<VertexPosition<Self>, GraphError>
    where
        B: Reborrow,
        B::Target: AsStorage<Arc<Self>>
            + AsStorage<Edge<Self>>
            + AsStorage<Vertex<Self>>
            + Consistent
            + Parametric<Data = Self>,
        T: ToArc<B>,
    {
        let arc = edge.into_arc();
        let (a, b) = arc
            .adjacent_vertices()
            .map(|vertex| *vertex.position())
            .try_collect()
            .expect_consistent();
        Ok(a.midpoint(b))
    }
}

pub trait FaceCentroid: GraphData
where
    Self::Vertex: AsPosition,
{
    fn centroid<B, T>(ring: T) -> Result<VertexPosition<Self>, GraphError>
    where
        B: Reborrow,
        B::Target:
            AsStorage<Arc<Self>> + AsStorage<Vertex<Self>> + Consistent + Parametric<Data = Self>,
        T: ToRing<B>;
}

impl<G> FaceCentroid for G
where
    G: GraphData,
    G::Vertex: AsPosition,
{
    fn centroid<B, T>(ring: T) -> Result<VertexPosition<Self>, GraphError>
    where
        B: Reborrow,
        B::Target:
            AsStorage<Arc<Self>> + AsStorage<Vertex<Self>> + Consistent + Parametric<Data = Self>,
        T: ToRing<B>,
    {
        let ring = ring.into_ring();
        Ok(
            VertexPosition::<Self>::centroid(ring.vertices().map(|vertex| *vertex.position()))
                .expect_consistent(),
        )
    }
}

pub trait FaceNormal: GraphData
where
    Self::Vertex: AsPosition,
{
    fn normal<B, T>(ring: T) -> Result<Vector<VertexPosition<Self>>, GraphError>
    where
        B: Reborrow,
        B::Target:
            AsStorage<Arc<Self>> + AsStorage<Vertex<Self>> + Consistent + Parametric<Data = Self>,
        T: ToRing<B>;
}

impl<G> FaceNormal for G
where
    G: FaceCentroid + GraphData,
    G::Vertex: AsPosition,
    Vector<VertexPosition<G>>: Cross<Output = Vector<VertexPosition<G>>>,
    VertexPosition<G>: EuclideanSpace,
{
    fn normal<B, T>(ring: T) -> Result<Vector<VertexPosition<Self>>, GraphError>
    where
        B: Reborrow,
        B::Target:
            AsStorage<Arc<Self>> + AsStorage<Vertex<Self>> + Consistent + Parametric<Data = Self>,
        T: ToRing<B>,
    {
        let ring = ring.into_ring();
        let (a, b) = ring
            .vertices()
            .take(2)
            .map(|vertex| *vertex.position())
            .try_collect()
            .expect_consistent();
        let c = G::centroid(ring)?;
        let ab = a - b;
        let bc = b - c;
        ab.cross(bc).normalize().ok_or(GraphError::Geometry)
    }
}

pub trait FacePlane: GraphData
where
    Self::Vertex: AsPosition,
    VertexPosition<Self>: FiniteDimensional<N = U3>,
{
    fn plane<B, T>(ring: T) -> Result<Plane<VertexPosition<Self>>, GraphError>
    where
        B: Reborrow,
        B::Target:
            AsStorage<Arc<Self>> + AsStorage<Vertex<Self>> + Consistent + Parametric<Data = Self>,
        T: ToRing<B>;
}

// TODO: The `lapack` feature depends on `ndarray-linalg` and Intel MKL. MKL is
//       dynamically linked, but the linkage fails during doctests and may fail
//       when launching a binary. The `lapack` feature and this implementation
//       have been disabled until an upstream fix is available. See
//       https://github.com/olson-sean-k/plexus/issues/58 and
//       https://github.com/rust-ndarray/ndarray-linalg/issues/229
// TODO: The `lapack` feature only supports Linux. See
//       https://github.com/olson-sean-k/theon/issues/1
//
//#[cfg(target_os = "linux")]
//mod lapack {
//    use super::*;
//
//    use smallvec::SmallVec;
//    use theon::adjunct::{FromItems, IntoItems};
//    use theon::lapack::Lapack;
//    use theon::space::Scalar;
//
//    impl<G> FacePlane for G
//    where
//        G: GraphData,
//        G::Vertex: AsPosition,
//        VertexPosition<G>: EuclideanSpace + FiniteDimensional<N = U3>,
//        Scalar<VertexPosition<G>>: Lapack,
//        Vector<VertexPosition<G>>: FromItems + IntoItems,
//    {
//        fn plane<B, T>(ring: T) -> Result<Plane<VertexPosition<G>>, GraphError>
//        where
//            B: Reborrow,
//            B::Target: AsStorage<Arc<Self>>
//                + AsStorage<Vertex<Self>>
//                + Consistent
//                + Parametric<Data = Self>,
//            T: ToRing<B>,
//        {
//            let ring = ring.into_ring();
//            let points = ring
//                .vertices()
//                .map(|vertex| *vertex.data.as_position())
//                .collect::<SmallVec<[_; 4]>>();
//            Plane::from_points(points).ok_or(GraphError::Geometry)
//        }
//    }
//}
