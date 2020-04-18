use plexus::integration::nalgebra;

use nalgebra::{Isometry3, Matrix4, Perspective3, Point3, RealField, Scalar, Vector3};

#[derive(Clone, Copy, Debug)]
pub struct Camera<T>
where
    T: RealField + Scalar,
{
    projection: Perspective3<T>,
    view: Isometry3<T>,
}

impl<T> Camera<T>
where
    T: RealField + Scalar,
{
    pub fn new(aspect: T, fov: T, near: T, far: T) -> Self {
        Camera {
            projection: Perspective3::new(aspect, fov, near, far),
            view: Isometry3::look_at_rh(
                &Point3::origin(),
                &Point3::new(T::zero(), T::zero(), -T::one()),
                &Vector3::y(),
            ),
        }
    }

    pub fn look_at(mut self, from: Point3<T>, to: Point3<T>) -> Self {
        self.view = Isometry3::look_at_rh(&from, &to, &Vector3::y());
        self
    }

    pub fn transform(&self) -> Matrix4<T> {
        self.projection.into_inner() * self.view.to_homogeneous()
    }
}
