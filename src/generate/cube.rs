use generate::generate::{Generate, IndexPolygonGenerator, PolygonGenerator,
                         PositionPolygonGenerator, PositionVertexGenerator,
                         TexturePolygonGenerator, VertexGenerator};
use generate::topology::{MapVerticesInto, Quad};
use generate::unit::Unit;
use geometry::{Duplet, Triplet};

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
pub struct Cube<T = f32>
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

    pub fn polygons_with_plane(&self) -> Generate<Self, Quad<Plane>> {
        Generate::new(self, 0..self.polygon_count(), Cube::polygon_with_plane)
    }

    fn polygon_with_plane(&self, index: usize) -> Quad<Plane> {
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

impl<T> PositionVertexGenerator for Cube<T>
where
    T: Unit,
{
    type Output = Triplet<T>;

    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn vertex_with_position(&self, index: usize) -> Self::Output {
        let x = if index & 0b100 == 0b100 { self.upper } else { self.lower };
        let y = if index & 0b010 == 0b010 { self.upper } else { self.lower };
        let z = if index & 0b001 == 0b001 { self.upper } else { self.lower };
        Triplet(x, y, z)
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

impl<T> PositionPolygonGenerator for Cube<T>
where
    T: Unit,
{
    type Output = Quad<Triplet<T>>;

    fn polygon_with_position(&self, index: usize) -> Self::Output {
        self.polygon_with_index(index)
            .map_vertices_into(|index| self.vertex_with_position(index))
    }
}

impl<T> IndexPolygonGenerator for Cube<T>
where
    T: Unit,
{
    type Output = Quad<usize>;

    fn polygon_with_index(&self, index: usize) -> <Self as IndexPolygonGenerator>::Output {
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

impl<T> TexturePolygonGenerator for Cube<T>
where
    T: Unit,
{
    type Output = Quad<Duplet<f32>>;

    fn polygon_with_texture(&self, index: usize) -> <Self as TexturePolygonGenerator>::Output {
        let uu = Duplet(1.0, 1.0);
        let ul = Duplet(1.0, 0.0);
        let ll = Duplet(0.0, 0.0);
        let lu = Duplet(0.0, 1.0);
        match index {
            0 | 4 | 5 => Quad::new(uu, ul, ll, lu), // front | bottom | back
            1 => Quad::new(ul, ll, lu, uu),         // right
            2 | 3 => Quad::new(lu, uu, ul, ll),     // top | left
            _ => panic!(),
        }
    }
}
