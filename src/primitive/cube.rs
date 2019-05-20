//! Cube primitives.
//!
//! # Examples
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! use nalgebra::Point3;
//! use plexus::graph::MeshGraph;
//! use plexus::index::HashIndexer;
//! use plexus::prelude::*;
//! use plexus::primitive::cube::Cube;
//!
//! # fn main() {
//! let graph = Cube::new()
//!     .polygons_with_position()
//!     .collect_with_indexer::<MeshGraph<Point3<f32>>, _>(HashIndexer::default())
//!     .unwrap();
//! # }
//! ```

use num::{One, Zero};
use theon::space::{Basis, EuclideanSpace, FiniteDimensional, Scalar, Vector, VectorSpace};
use typenum::{U2, U3};

use crate::primitive::generate::{
    Generate, NormalGenerator, NormalIndexGenerator, NormalPolygonGenerator, NormalVertexGenerator,
    PolygonGenerator, PolygonsWithNormal, PolygonsWithPosition, PolygonsWithUvMap,
    PositionGenerator, PositionIndexGenerator, PositionPolygonGenerator, PositionVertexGenerator,
    UvMapGenerator, UvMapPolygonGenerator, VerticesWithNormal, VerticesWithPosition,
};
use crate::primitive::{Converged, Map, Quad};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Plane {
    XY,  // front
    NXY, // back
    YZ,  // right
    NYZ, // left
    XZ,  // bottom
    NXZ, // top
}

impl Plane {
    pub fn normal<S>(&self) -> Vector<S>
    where
        S: EuclideanSpace,
        <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U3>,
        Vector<S>: Basis,
    {
        match *self {
            Plane::XY => Vector::<S>::z(),   // front
            Plane::NXY => -Vector::<S>::z(), // back
            Plane::YZ => Vector::<S>::x(),   // right
            Plane::NYZ => -Vector::<S>::x(), // left
            Plane::XZ => -Vector::<S>::y(),  // bottom
            Plane::NXZ => Vector::<S>::y(),  // top
        }
    }
}

#[derive(Clone, Copy)]
pub struct Bounds<S>
where
    S: EuclideanSpace,
{
    lower: Scalar<S>,
    upper: Scalar<S>,
}

impl<S> Bounds<S>
where
    S: EuclideanSpace,
{
    pub fn with_radius(radius: Scalar<S>) -> Self {
        Bounds {
            lower: -radius,
            upper: radius,
        }
    }

    pub fn with_width(width: Scalar<S>) -> Self {
        Self::with_radius(width / (Scalar::<S>::one() + One::one()))
    }

    pub fn unit_radius() -> Self {
        Self::with_radius(One::one())
    }

    pub fn unit_width() -> Self {
        Self::with_width(One::one())
    }
}

impl<S> Default for Bounds<S>
where
    S: EuclideanSpace,
{
    fn default() -> Self {
        Self::unit_width()
    }
}

#[derive(Clone, Copy)]
pub struct Cube;

impl Cube {
    pub fn new() -> Self {
        Cube
    }

    pub fn polygon_with_plane(&self, index: usize) -> Quad<Plane> {
        Quad::converged(self.vertex_with_plane(index))
    }

    pub fn polygons_with_plane(&self) -> Generate<Self, (), Quad<Plane>> {
        Generate::new(self, (), self.polygon_count(), |cube, _, index| {
            cube.polygon_with_plane(index)
        })
    }

    pub fn vertex_with_plane_count(&self) -> usize {
        self.polygon_count()
    }

    pub fn vertex_with_plane(&self, index: usize) -> Plane {
        match index {
            0 => Plane::XY,  // front
            1 => Plane::YZ,  // right
            2 => Plane::NXZ, // top
            3 => Plane::NYZ, // left
            4 => Plane::XZ,  // bottom
            5 => Plane::NXY, // back
            _ => panic!(),
        }
    }

    pub fn vertices_with_plane(&self) -> Generate<Self, (), Plane> {
        Generate::new(
            self,
            (),
            self.vertex_with_plane_count(),
            |cube, _, index| cube.vertex_with_plane(index),
        )
    }
}

impl Default for Cube {
    fn default() -> Self {
        Cube::new()
    }
}

impl PolygonGenerator for Cube {
    fn polygon_count(&self) -> usize {
        6
    }
}

impl<S> NormalGenerator<S> for Cube
where
    S: EuclideanSpace,
    <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U3>,
{
    type State = ();
}

impl<S> NormalVertexGenerator<S> for Cube
where
    S: EuclideanSpace,
    <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U3>,
    Vector<S>: Basis,
{
    fn vertex_with_normal_from(&self, _: &Self::State, index: usize) -> Vector<S> {
        // There is a unique normal for each face (plane).
        self.vertex_with_plane(index).normal::<S>()
    }

    fn vertex_with_normal_count(&self) -> usize {
        self.polygon_count()
    }
}

impl<S> NormalPolygonGenerator<S> for Cube
where
    S: EuclideanSpace,
    <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U3>,
    Vector<S>: Basis,
{
    type Output = Quad<Vector<S>>;

    fn polygon_with_normal_from(&self, state: &Self::State, index: usize) -> Self::Output {
        self.index_for_normal(index)
            .map(|index| NormalVertexGenerator::<S>::vertex_with_normal_from(self, state, index))
    }
}

impl NormalIndexGenerator for Cube {
    type Output = Quad<usize>;

    fn index_for_normal(&self, index: usize) -> Self::Output {
        assert!(index < self.polygon_count());
        Quad::converged(index)
    }
}

impl<S> PositionGenerator<S> for Cube
where
    S: EuclideanSpace,
    <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U3>,
{
    type State = Bounds<S>;
}

impl<S> PositionVertexGenerator<S> for Cube
where
    S: EuclideanSpace,
    <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U3>,
    Vector<S>: Basis,
{
    fn vertex_with_position_from(&self, state: &Self::State, index: usize) -> S {
        let x = if index & 0b100 == 0b100 {
            state.upper
        }
        else {
            state.lower
        };
        let y = if index & 0b010 == 0b010 {
            state.upper
        }
        else {
            state.lower
        };
        let z = if index & 0b001 == 0b001 {
            state.upper
        }
        else {
            state.lower
        };
        S::from_xyz(x, y, z)
    }

    fn vertex_with_position_count(&self) -> usize {
        8
    }
}

impl<S> PositionPolygonGenerator<S> for Cube
where
    S: EuclideanSpace,
    <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U3>,
    Vector<S>: Basis,
{
    type Output = Quad<S>;

    fn polygon_with_position_from(&self, state: &Self::State, index: usize) -> Self::Output {
        self.index_for_position(index).map(|index| {
            PositionVertexGenerator::<S>::vertex_with_position_from(self, state, index)
        })
    }
}

impl PositionIndexGenerator for Cube {
    type Output = Quad<usize>;

    fn index_for_position(&self, index: usize) -> Self::Output {
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

impl<S> UvMapGenerator<S> for Cube
where
    S: EuclideanSpace,
    <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U2>,
{
    type State = ();
}

impl<S> UvMapPolygonGenerator<S> for Cube
where
    S: EuclideanSpace,
    <S as EuclideanSpace>::CoordinateSpace: FiniteDimensional<N = U2>,
    Vector<S>: Basis + Converged,
{
    type Output = Quad<Vector<S>>;

    fn polygon_with_uv_map_from(&self, _: &Self::State, index: usize) -> Self::Output {
        let uu = Vector::<S>::converged(One::one());
        let ul = Vector::<S>::from_xy(One::one(), Zero::zero());
        let ll = Vector::<S>::converged(Zero::zero());
        let lu = Vector::<S>::from_xy(Zero::zero(), One::one());
        match index {
            0 | 4 | 5 => Quad::new(uu, ul, ll, lu), // front | bottom | back
            1 => Quad::new(ul, ll, lu, uu),         // right
            2 | 3 => Quad::new(lu, uu, ul, ll),     // top | left
            _ => panic!(),
        }
    }
}

impl VerticesWithNormal for Cube {}
impl VerticesWithPosition for Cube {}

impl PolygonsWithNormal for Cube {}
impl PolygonsWithPosition for Cube {}
impl PolygonsWithUvMap for Cube {}

// TODO: THIS IS A SANITY TEST. Remove this. It allows for a quick check
//       without needing to refactor all generator code.
fn test() {
    use decorum::N64;
    use nalgebra::Point3;

    use crate::graph::MeshGraph;
    use crate::index::{Flat3, HashIndexer};
    use crate::prelude::*;
    use crate::primitive;
    use crate::primitive::cube::Cube;

    let mut graph = Cube::new()
        .polygons_with_position::<Point3<N64>>()
        .triangulate()
        .collect::<MeshGraph>();

    let cube = Cube::new();
    let (_, _) = primitive::zip_vertices((
        cube.polygons_with_position::<Point3<N64>>(),
        cube.polygons_with_normal::<Point3<N64>>(),
    ))
    .triangulate()
    .index_vertices::<Flat3, _>(HashIndexer::default());
}
