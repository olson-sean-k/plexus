use num::Num;

use generate::generate::{Generate, IndexedPolygonGenerator, VertexGenerator, PolygonGenerator,
                         SpatialVertexGenerator, SpatialPolygonGenerator,
                         TexturedPolygonGenerator};
use generate::topology::{MapTopology, Quad};

pub trait Unit: Copy + Num {
    fn unit_radius() -> (Self, Self);
    fn unit_width() -> (Self, Self);
}

macro_rules! unit {
    (integer => $($t:ty),*) => {$(
        impl Unit for $t {
            fn unit_radius() -> (Self, Self) {
                use num::{One, Zero};
                (Self::zero(), Self::one() + Self::one())
            }

            fn unit_width() -> (Self, Self) {
                use num::{One, Zero};
                (Self::zero(), Self::one())
            }
        }
    )*};
    (real => $($t:ty),*) => {$(
        impl Unit for $t {
            fn unit_radius() -> (Self, Self) {
                use num::One;
                (-Self::one(), Self::one())
            }

            fn unit_width() -> (Self, Self) {
                use num::One;
                let half = Self::one() / (Self::one() + Self::one());
                (-half, half)
            }
        }
    )*};
}

unit!(integer => i8, i16, i32, i64, u8, u16, u32, u64);
unit!(real => f32, f64);

#[derive(Clone, Copy)]
pub enum Plane {
    XY,
    NXY,
    ZY,
    NZY,
    XZ,
    XNZ,
}

#[derive(Clone)]
pub struct Cube<T>
where
    T: Unit,
{
    lower: T,
    upper: T,
}

impl<T> Cube<T>
where
    T: Unit,
{
    fn new(lower: T, upper: T) -> Self {
        Cube {
            lower: lower,
            upper: upper,
        }
    }

    pub fn with_unit_radius() -> Self {
        let (lower, upper) = T::unit_radius();
        Cube::new(lower, upper)
    }

    pub fn with_unit_width() -> Self {
        let (lower, upper) = T::unit_width();
        Cube::new(lower, upper)
    }

    pub fn planar_polygons(&self) -> Generate<Self, Quad<Plane>> {
        Generate::new(self, 0..self.polygon_count(), Cube::planar_polygon)
    }

    fn planar_polygon(&self, index: usize) -> Quad<Plane> {
        match index {
            0 => Quad::converged(Plane::XY),  // front
            1 => Quad::converged(Plane::NZY), // right
            2 => Quad::converged(Plane::XNZ), // top
            3 => Quad::converged(Plane::ZY),  // left
            4 => Quad::converged(Plane::XZ),  // bottom
            5 => Quad::converged(Plane::NXY), // back
            _ => panic!(),
        }
    }
}

impl<T> VertexGenerator for Cube<T>
where
    T: Unit,
{
    fn vertex_count(&self) -> usize {
        8
    }
}

impl<T> SpatialVertexGenerator for Cube<T>
where
    T: Unit,
{
    type Output = (T, T, T);

    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn spatial_vertex(&self, index: usize) -> Self::Output {
        let x = if index & 0b100 == 0b100 { self.upper } else { self.lower };
        let y = if index & 0b010 == 0b010 { self.upper } else { self.lower };
        let z = if index & 0b001 == 0b001 { self.upper } else { self.lower };
        (x, y, z)
    }
}

impl<T> PolygonGenerator for Cube<T>
where
    T: Unit,
{
    fn polygon_count(&self) -> usize {
        6
    }
}

impl<T> SpatialPolygonGenerator for Cube<T>
where
    T: Unit,
{
    type Output = Quad<(T, T, T)>;

    fn spatial_polygon(&self, index: usize) -> Self::Output {
        self.indexed_polygon(index)
            .map_topology(|index| self.spatial_vertex(index))
    }
}

impl<T> IndexedPolygonGenerator for Cube<T>
where
    T: Unit,
{
    type Output = Quad<usize>;

    fn indexed_polygon(&self, index: usize) -> <Self as IndexedPolygonGenerator>::Output {
        match index {
            0 => Quad::new(5, 7, 3, 1), // front
            1 => Quad::new(6, 7, 5, 4), // right
            2 => Quad::new(3, 7, 6, 2), // top
            3 => Quad::new(0, 1, 3, 2), // left
            4 => Quad::new(4, 5, 1, 0), // bottom
            5 => Quad::new(0, 2, 6, 4), // back
            _ => panic!(),
        }
    }
}

impl<T> TexturedPolygonGenerator for Cube<T>
where
    T: Unit,
{
    type Output = Quad<(f32, f32)>;

    fn textured_polygon(&self, index: usize) -> <Self as TexturedPolygonGenerator>::Output {
        let uu = (1.0, 1.0);
        let ul = (1.0, 0.0);
        let ll = (0.0, 0.0);
        let lu = (0.0, 1.0);
        match index {
            0 | 4 | 5 => Quad::new(uu, ul, ll, lu), // front | bottom | back
            1 => Quad::new(ul, ll, lu, uu), // right
            2 | 3 => Quad::new(lu, uu, ul, ll), // top | left
            _ => panic!(),
        }
    }
}
