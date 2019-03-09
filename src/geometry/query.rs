use crate::geometry::convert::AsPosition;
use crate::geometry::{Ray, Space};

pub trait RayIntersection<S>
where
    S: Space,
    S::Vertex: AsPosition<Target = S::Point>,
{
    fn intersection(&self, ray: Ray<S>) -> Option<(S::Scalar, S::Scalar)>;
}
