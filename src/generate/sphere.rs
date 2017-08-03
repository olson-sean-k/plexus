use num::Float;
use num::traits::FloatConst;
use std::cmp;
use std::marker::PhantomData;

use generate::generate::{IndexedPolygonGenerator, VertexGenerator, PolygonGenerator,
                         SpatialVertexGenerator, SpatialPolygonGenerator};
use generate::topology::{Polygon, Triangle, Quad};

#[derive(Clone)]
pub struct UVSphere<T>
where
    T: Float + FloatConst,
{
    nu: usize, // Meridians.
    nv: usize, // Parallels.
    phantom: PhantomData<T>,
}

impl<T> UVSphere<T>
where
    T: Float + FloatConst,
{
    pub fn with_unit_radius(nu: usize, nv: usize) -> Self {
        let nu = cmp::max(3, nu);
        let nv = cmp::max(2, nv);
        UVSphere {
            nu: nu,
            nv: nv,
            phantom: PhantomData,
        }
    }

    fn spatial_vertex(&self, u: usize, v: usize) -> (T, T, T) {
        let u = (T::from(u).unwrap() / T::from(self.nu).unwrap()) * T::PI() * (T::one() + T::one());
        let v = (T::from(v).unwrap() / T::from(self.nv).unwrap()) * T::PI();
        (u.cos() * v.sin(), u.sin() * v.sin(), v.cos())
    }

    fn indexed_vertex(&self, u: usize, v: usize) -> usize {
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
    T: Float + FloatConst,
{
    fn vertex_count(&self) -> usize {
        (self.nv - 1) * self.nu + 2
    }
}

impl<T> SpatialVertexGenerator for UVSphere<T>
where
    T: Float + FloatConst,
{
    type Output = (T, T, T);

    fn spatial_vertex(&self, index: usize) -> Self::Output {
        if index == 0 {
            self.spatial_vertex(0, 0)
        }
        else if index == self.vertex_count() - 1 {
            self.spatial_vertex(0, self.nv)
        }
        else {
            let index = index - 1;
            self.spatial_vertex(index % self.nu, (index / self.nu) + 1)
        }
    }
}

impl<T> PolygonGenerator for UVSphere<T>
where
    T: Float + FloatConst,
{
    fn polygon_count(&self) -> usize {
        self.nu * self.nv
    }
}

impl<T> SpatialPolygonGenerator for UVSphere<T>
where
    T: Float + FloatConst,
{
    type Output = Polygon<(T, T, T)>;

    fn spatial_polygon(&self, index: usize) -> Self::Output {
        // Prevent floating point rounding errors by wrapping the incremented
        // values for `(u, v)` into `(p, q)`. This is important for indexing
        // geometry, because small differences in the computation of spatial
        // vertices will produce unique output vertices.
        let (u, v) = self.map_polygon_index(index);
        let (p, q) = (u + 1, (v + 1) % self.nv);

        // Generate the vertices at the requested meridian and parallel. The
        // upper and lower bounds of (u, v) are always used, so generate them
        // in advance (`low` and `high`). Emit triangles at the poles,
        // otherwise quads.
        let low = self.spatial_vertex(u, v);
        let high = self.spatial_vertex(p, q);
        if v == 0 {
            Polygon::Triangle(Triangle::new(low, self.spatial_vertex(u, q), high))
        }
        else if v == self.nv - 1 {
            Polygon::Triangle(Triangle::new(high, self.spatial_vertex(p, v), low))
        }
        else {
            Polygon::Quad(Quad::new(
                low,
                self.spatial_vertex(u, q),
                high,
                self.spatial_vertex(p, v),
            ))
        }
    }
}

impl<T> IndexedPolygonGenerator for UVSphere<T>
where
    T: Float + FloatConst,
{
    type Output = Polygon<usize>;

    fn indexed_polygon(&self, index: usize) -> <Self as IndexedPolygonGenerator>::Output {
        let (u, v) = self.map_polygon_index(index);
        let (p, q) = (u + 1, v + 1);

        let low = self.indexed_vertex(u, v);
        let high = self.indexed_vertex(p, q);
        if v == 0 {
            Polygon::Triangle(Triangle::new(low, self.indexed_vertex(u, q), high))
        }
        else if v == self.nv - 1 {
            Polygon::Triangle(Triangle::new(high, self.indexed_vertex(p, v), low))
        }
        else {
            Polygon::Quad(Quad::new(
                low,
                self.indexed_vertex(u, q),
                high,
                self.indexed_vertex(p, v),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::iter::FromIterator;

    use super::super::*;

    #[test]
    fn vertex_count() {
        assert_eq!(
            5,
            sphere::UVSphere::<f32>::with_unit_radius(3, 2)
                .spatial_vertices() // 5 conjoint vertices.
                .count()
        );
    }

    #[test]
    fn polygon_vertex_count() {
        assert_eq!(
            18,
            sphere::UVSphere::<f32>::with_unit_radius(3, 2)
                .spatial_polygons() // 6 triangles, 18 vertices.
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
                    .indexed_polygons() // 18 vertices, 5 indeces.
                    .vertices()
            ).len()
        )
    }
}
