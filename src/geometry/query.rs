use crate::geometry::convert::AsPosition;
use crate::geometry::{Geometry, Ray, Space};

pub trait RayIntersection<S>
where
    S: Space,
    <S as Geometry>::Vertex: AsPosition<Target = S::Point>,
{
    fn intersection(&self, ray: Ray<S>) -> Option<(S::Scalar, S::Scalar)>;
}
