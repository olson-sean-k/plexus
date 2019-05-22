extern crate decorum;
extern crate nalgebra;
extern crate plexus;

use decorum::N64;
use nalgebra::{Point3, Vector2};
use plexus::index::{Flat3, HashIndexer};
use plexus::prelude::*;
use plexus::primitive;
use plexus::primitive::cube::{Bounds, Cube, Plane};

fn map_unit_uv(position: Point3<N64>, plane: Plane, unit: N64) -> Vector2<N64> {
    let map = |position: N64| -> N64 { position / unit };
    match plane {
        Plane::XY => Vector2::new(map(position.x), map(position.y)),
        Plane::NXY => Vector2::new(-map(position.x), map(position.y)),
        Plane::YZ => Vector2::new(map(position.y), map(position.z)),
        Plane::NYZ => Vector2::new(-map(position.y), map(position.z)),
        Plane::XZ => Vector2::new(map(position.x), map(position.z)),
        Plane::NXZ => Vector2::new(-map(position.x), map(position.z)),
    }
}

fn main() {
    let width = 8.0.into();
    let cube = Cube::default();
    // Zip positions and planes into the vertices of a stream of polygons.
    let polygons = primitive::zip_vertices((
        cube.polygons_with_position_from::<Point3<N64>>(Bounds::with_width(width)),
        cube.polygons_with_plane(),
    ));
    // Use the position and plane to map texture coordinates and then
    // triangulate the polygons and index them.
    let (_, _) = polygons
        .map_vertices(|(position, plane)| (position, plane, map_unit_uv(position, plane, width)))
        .triangulate()
        .index_vertices::<Flat3, _>(HashIndexer::default());
}
