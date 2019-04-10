use crate::geometry::alias::Vector;
use crate::geometry::space::EuclideanSpace;

pub trait Intersection<T> {
    type Output;

    fn intersection(&self, _: &T) -> Option<Self::Output>;
}

pub struct Ray<S>
where
    S: EuclideanSpace,
{
    pub origin: S,
    pub extent: Vector<S>,
}

impl<S> Clone for Ray<S>
where
    S: EuclideanSpace,
{
    fn clone(&self) -> Self {
        Ray {
            origin: self.origin.clone(),
            extent: self.extent.clone(),
        }
    }
}

impl<S> Copy for Ray<S>
where
    S: Copy + EuclideanSpace,
    Vector<S>: Copy,
{
}
