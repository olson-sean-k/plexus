![Plexus](https://raw.githubusercontent.com/olson-sean-k/plexus/master/doc/plexus.png)

**Plexus** is a Rust library for generating and manipulating 3D meshes.

[![Build Status](https://travis-ci.org/olson-sean-k/plexus.svg?branch=master)](https://travis-ci.org/olson-sean-k/plexus)
[![Documentation](https://docs.rs/plexus/badge.svg)](https://docs.rs/plexus)
[![Crate](https://img.shields.io/crates/v/plexus.svg)](https://crates.io/crates/plexus)

## Generation and Iterator Expressions

Meshes can be generated from primitives like cubes and spheres using iterator
expressions. Primitives emit topological structures like `Triangle`s or
`Quad`s, which contain arbitrary geometric data in their vertices. These can be
transformed and decomposed into other topologies and geometric data via
triangulation, tesselation, and conversion into rendering pipeline data.

```rust
use plexus::r32;
use plexus::buffer::conjoint::ConjointBuffer;
use plexus::generate::{sphere, MapVertices, SpatialPolygons};

// Example module in the local crate that provides rendering.
use render::{self, Vertex};

// Construct a buffer of index and vertex data from a sphere primitive. `r32`
// is a type alias for `OrderedFloat<f32>`, which can be hashed by `collect()`.
// Note that `(r32, r32, r32)` is convertible to `Vertex` via the `From` trait
// in this example.
let buffer = sphere::UVSphere::<f32>::with_unit_radius(16, 16)
    .spatial_polygons()
    .map_verticies(|(x, y, z)| (r32::from(x), r32::from(y), r32::from(z)))
    .map_verticies(|(x, y, z)| (x * 10.0, y * 10.0, z * 10.0))
    .collect::<ConjointBuffer<u64, Vertex>>();
render::draw(buffer.as_index_slice(), buffer.as_vertex_slice());
```

## Half-Edge Graph Meshes

Generators are flexible and easy to use, but only represent vertex geometry and
are difficult to query and manipulate once complete. A `Mesh`, represented as a
[half-edge graph](https://en.wikipedia.org/wiki/doubly_connected_edge_list),
supports arbitrary geometry for vertices, edges, and faces. The graph can also
be queried and manipulated in ways that generators and iterator expressions
cannot.

```rust
use plexus::r32;
use plexus::generate::{sphere, MapVertices, SpatialPolygons};
use plexus::graph::{FaceKey, FaceRef, Mesh};

// Example module in the local crate that provides a custom mesh geometry.
use render::MeshGeometry;

// Construct a mesh from a sphere primitive.
let mesh = sphere::UVSphere::<f32>::with_unit_radius(3, 2)
    .spatial_polygons()
    .map_vertices(|(x, y, z)| (r32::from(x), r32::from(y), r32::from(z)))
    .collect::<Mesh<(r32, r32, r32)>>()
    .into_geometry::<MeshGeometry>();
// Get one of the faces and iterate over its neighboring faces using a
// "circulator".
// TODO: Do not use `FaceKey::default()`.
let face = mesh.face(FaceKey::default()).unwrap();
for face in face.faces() {
    println!("Neighbor: {:?}", face.key);
}
```
