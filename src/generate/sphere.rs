use decorum::Real;
use num::NumCast;
use num::traits::FloatConst;
use std::cmp;
use std::marker::PhantomData;

use generate::generate::{IndexPolygonGenerator, PolygonGenerator, PositionPolygonGenerator,
                         PositionVertexGenerator, VertexGenerator};
use generate::topology::{Polygon, Quad, Triangle};
use generate::unit::Unit;
use geometry::Triplet;

#[derive(Clone)]
pub struct UVSphere<T = f32>
where
    T: FloatConst + NumCast + Real + Unit,
{
    nu: usize, // Meridians.
    nv: usize, // Parallels.
    unit: T,
    phantom: PhantomData<T>,
}

impl<T> UVSphere<T>
where
    T: FloatConst + NumCast + Real + Unit,
{
    fn new(nu: usize, nv: usize, upper: T) -> Self {
        let nu = cmp::max(3, nu);
        let nv = cmp::max(2, nv);
        UVSphere {
            nu: nu,
            nv: nv,
            unit: upper,
            phantom: PhantomData,
        }
    }

    pub fn with_unit_radius(nu: usize, nv: usize) -> Self {
        Self::new(nu, nv, T::unit_radius().1)
    }

    pub fn with_unit_width(nu: usize, nv: usize) -> Self {
        Self::new(nu, nv, T::unit_width().1)
    }

    fn vertex_with_position(&self, u: usize, v: usize) -> Triplet<T> {
        let u = (T::from(u).unwrap() / T::from(self.nu).unwrap()) * T::PI() * (T::one() + T::one());
        let v = (T::from(v).unwrap() / T::from(self.nv).unwrap()) * T::PI();
        Triplet(
            self.unit * u.cos() * v.sin(),
            self.unit * u.sin() * v.sin(),
            self.unit * v.cos(),
        )
    }

    fn vertex_with_index(&self, u: usize, v: usize) -> usize {
        if v == 0 {
            0
        }
        else if v == self.nv {
            ((self.nv - 1) * self.nu) + 1
        }
        else {
            ((v - 1) * self.nu) + (u % self.nu) + 1
        }
    }

    fn map_polygon_index(&self, index: usize) -> (usize, usize) {
        (index % self.nu, index / self.nu)
    }
}

impl<T> VertexGenerator for UVSphere<T>
where
    T: FloatConst + NumCast + Real + Unit,
{
    fn vertex_count(&self) -> usize {
        (self.nv - 1) * self.nu + 2
    }
}

impl<T> PositionVertexGenerator for UVSphere<T>
where
    T: FloatConst + NumCast + Real + Unit,
{
    type Output = Triplet<T>;

    fn vertex_with_position(&self, index: usize) -> Self::Output {
        if index == 0 {
            self.vertex_with_position(0, 0)
        }
        else if index == self.vertex_count() - 1 {
            self.vertex_with_position(0, self.nv)
        }
        else {
            let index = index - 1;
            self.vertex_with_position(index % self.nu, (index / self.nu) + 1)
        }
    }
}

impl<T> PolygonGenerator for UVSphere<T>
where
    T: FloatConst + NumCast + Real + Unit,
{
    fn polygon_count(&self) -> usize {
        self.nu * self.nv
    }
}

impl<T> PositionPolygonGenerator for UVSphere<T>
where
    T: FloatConst + NumCast + Real + Unit,
{
    type Output = Polygon<Triplet<T>>;

    fn polygon_with_position(&self, index: usize) -> Self::Output {
        // Prevent floating point rounding errors by wrapping the incremented
        // values for `(u, v)` into `(p, q)`. This is important for indexing
        // geometry, because small differences in the computation of spatial
        // vertices will produce redundant output vertices. There should be
        // exactly `(nv - 1) * nu + 2` unique values of `(u, v)` used to
        // generate positions.
        //
        // There are two important observations:
        //
        //   1. `u` must wrap, but `v` need not. There are `nu` meridians of
        //      points and polygons, but there are `nv` parallels of polygons
        //      and `nv + 1` parallels of points.
        //   2. `u`, which represents a meridian, is meaningless at the poles,
        //      and can be normalized to zero.
        let (u, v) = self.map_polygon_index(index);
        let (p, q) = ((u + 1) % self.nu, v + 1);

        // Generate the vertices at the requested meridian and parallel. The
        // lower bound of `(u, v)` is always used, so compute that in advance
        // (`lower`). Emit triangles at the poles, otherwise quads.
        let lower = self.vertex_with_position(u, v);
        if v == 0 {
            Polygon::Triangle(Triangle::new(
                lower,
                self.vertex_with_position(u, q),
                self.vertex_with_position(p, q),
            ))
        }
        else if v == self.nv - 1 {
            Polygon::Triangle(Triangle::new(
                // Normalize `u` at the pole, using `(0, nv)` in place of
                // `(p, q)`.
                self.vertex_with_position(0, self.nv),
                self.vertex_with_position(p, v),
                lower,
            ))
        }
        else {
            Polygon::Quad(Quad::new(
                lower,
                self.vertex_with_position(u, q),
                self.vertex_with_position(p, q),
                self.vertex_with_position(p, v),
            ))
        }
    }
}

impl<T> IndexPolygonGenerator for UVSphere<T>
where
    T: FloatConst + NumCast + Real + Unit,
{
    type Output = Polygon<usize>;

    fn polygon_with_index(&self, index: usize) -> <Self as IndexPolygonGenerator>::Output {
        let (u, v) = self.map_polygon_index(index);
        let (p, q) = (u + 1, v + 1);

        let low = self.vertex_with_index(u, v);
        let high = self.vertex_with_index(p, q);
        if v == 0 {
            Polygon::Triangle(Triangle::new(low, self.vertex_with_index(u, q), high))
        }
        else if v == self.nv - 1 {
            Polygon::Triangle(Triangle::new(high, self.vertex_with_index(p, v), low))
        }
        else {
            Polygon::Quad(Quad::new(
                low,
                self.vertex_with_index(u, q),
                high,
                self.vertex_with_index(p, v),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::iter::FromIterator;

    use generate::*;

    #[test]
    fn vertex_count() {
        assert_eq!(
            5,
            sphere::UVSphere::<f32>::with_unit_radius(3, 2)
                .vertices_with_position() // 5 conjoint vertices.
                .count()
        );
    }

    #[test]
    fn polygon_vertex_count() {
        assert_eq!(
            18,
            sphere::UVSphere::<f32>::with_unit_radius(3, 2)
                .polygons_with_position() // 6 triangles, 18 vertices.
                .vertices()
                .count()
        );
    }

    #[test]
    fn index_to_vertex_mapping() {
        assert_eq!(
            5,
            BTreeSet::from_iter(
                sphere::UVSphere::<f32>::with_unit_radius(3, 2)
                    .polygons_with_index() // 18 vertices, 5 indeces.
                    .vertices()
            ).len()
        )
    }
}
