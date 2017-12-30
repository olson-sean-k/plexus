extern crate decorum;
extern crate nalgebra;
extern crate plexus;

use decorum::R32;
use nalgebra::{Point2, Point3};
use plexus::generate::{self, HashIndexer};
use plexus::generate::cube::{Cube, Plane};
use plexus::prelude::*;

fn map_unit_uv(position: Point3<R32>, plane: Plane, unit: R32) -> Point2<R32> {
    let map = |position: R32| -> R32 { position / unit };
    match plane {
        Plane::XY => Point2::new(map(position.x), map(position.y)),
        Plane::NXY => Point2::new(-map(position.x), map(position.y)),
        Plane::ZY => Point2::new(map(position.z), map(position.y)),
        Plane::NZY => Point2::new(-map(position.z), map(position.y)),
        Plane::XZ => Point2::new(map(position.x), map(position.z)),
        Plane::XNZ => Point2::new(map(position.x), -map(position.z)),
    }
}

fn main() {
    let cube = Cube::default();
    // Zip positions and planes into the vertices of a stream of polygons.
    let polygons = generate::zip_vertices((
        cube.polygons_with_position()
            .map_vertices(|position| -> Point3<R32> { position.into() })
            .map_vertices(|position| position * 8.0.into()),
        cube.polygons_with_plane(),
    ));
    // Use the position and plane to map texture coordinates and then
    // triangulate the polygons and index them.
    let (_, _) = polygons
        .map_vertices(|(position, plane)| {
            (position, plane, map_unit_uv(position, plane, 8.0.into()))
        })
        .triangulate()
        .index_vertices(HashIndexer::default());
}
