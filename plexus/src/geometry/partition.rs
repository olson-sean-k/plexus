use approx::abs_diff_eq;
use num::traits::real::Real;
use num::Zero;
use std::cmp::Ordering;
use theon::query::{Line, Plane};
use theon::space::{EuclideanSpace, FiniteDimensional};
use typenum::{U1, U2, U3};

// "Left" and "right" are arbitrary here and refer to the partitioned spaces
// formed by a geometric entity. This is a point, line, and plane in one, two,
// three dimensions, respectively.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum BinaryPartition {
    Left,
    Right,
}

pub trait PointPartition<S>
where
    S: EuclideanSpace,
{
    fn partition(&self, point: S) -> Option<BinaryPartition>;
}

impl<S> PointPartition<S> for S
where
    S: EuclideanSpace + FiniteDimensional<N = U1>,
{
    // TODO: Should `EmptyOrd` be used here?
    fn partition(&self, point: S) -> Option<BinaryPartition> {
        let ax = self.into_x();
        let px = point.into_x();
        match px.partial_cmp(&ax) {
            Some(Ordering::Less) => Some(BinaryPartition::Left),
            Some(Ordering::Greater) => Some(BinaryPartition::Right),
            _ => None,
        }
    }
}

impl<S> PointPartition<S> for Line<S>
where
    S: EuclideanSpace + FiniteDimensional<N = U2>,
{
    fn partition(&self, point: S) -> Option<BinaryPartition> {
        // Compute the determinant of a matrix composed of points along the line
        // and the queried point. This can also be thought of as a two-
        // dimensional cross product.
        // TODO: Perhaps this should be exposed by Theon instead.
        let (ax, ay) = self.origin.into_xy();
        let (bx, by) = (self.origin + *self.direction.get()).into_xy();
        let (px, py) = point.into_xy();
        let determinant = ((bx - ax) * (py - ay)) - ((by - ay) * (px - ax));
        if abs_diff_eq!(determinant, Zero::zero()) {
            None
        }
        else {
            Some(if determinant.is_sign_positive() {
                BinaryPartition::Left
            }
            else {
                BinaryPartition::Right
            })
        }
    }
}

impl<S> PointPartition<S> for Plane<S>
where
    S: EuclideanSpace + FiniteDimensional<N = U3>,
{
    fn partition(&self, point: S) -> Option<BinaryPartition> {
        let _ = point;
        todo!()
    }
}
