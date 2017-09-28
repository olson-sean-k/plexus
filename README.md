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
use nalgebra::Point3;
use plexus::buffer::MeshBuffer;
use plexus::generate::sphere;
use plexus::prelude::*; // Common traits.

// Example module in the local crate that provides basic rendering.
use render::{self, Color, Vertex};

// Construct a buffer of index and vertex data from a sphere primitive.
let buffer = sphere::UVSphere::<f32>::with_unit_radius(16, 16)
    .polygons_with_position()
    .map_vertices(|position| -> Point3<_> { position.into() })
    .map_vertices(|position| position * 10.0)
    .map_vertices(|position| Vertex::new(position, Color::white()))
    .collect::<MeshBuffer<u32, Vertex>>();
render::draw(buffer.as_index_slice(), buffer.as_vertex_slice());
```

## Half-Edge Graph Meshes

Generators are flexible and easy to use, but only represent vertex geometry and
are difficult to query and manipulate. A `Mesh`, represented as a [half-edge
graph](https://en.wikipedia.org/wiki/doubly_connected_edge_list), supports
arbitrary geometry for vertices, edges, and faces. The graph can also be
queried and manipulated in ways that generators and iterator expressions
cannot.

```rust
use nalgebra::Point3;
use plexus::generate::{sphere, LruIndexer};
use plexus::graph::{FaceKey, Mesh};
use plexus::prelude::*;

// Construct a mesh from a sphere primitive. The vertex geometry is convertible
// to `Point3<f32>` via the `FromGeometry` trait in this example.
let mut mesh = sphere::UVSphere::<f32>::with_unit_radius(8, 8)
    .polygons_with_position()
    .map_vertices(|position| position.into_hash())
    .collect::<Mesh<Point3<f32>>>();
// Extrude a face in the mesh.
let key = mesh.faces().nth(0).unwrap().key();
let face = mesh.face_mut(key).unwrap();
let face = face.extrude(1.0).unwrap();
```

## Geometric Traits

Meshes support arbitrary geometry via optional traits. Implementing these
traits allows more operations to be supported, but only two basic traits are
required: `Geometry` and `Attribute`.

Geometric traits are optionally implemented for types in the
[nalgebra](https://crates.io/crates/nalgebra) and
[cgmath](https://crates.io/crates/cgmath) crates so that common types can be
used right away for vertex geomtry.

## Hashing Floating-Point Values

When collecting an iterator expression into a graph or buffer, an indexer is
used to transform the geometry into raw buffers. `HashIndexer` is fast and
reliable, and is used by `collect` (which can be overridden via
`collect_with_indexer`). However, geometry often contains floating point
values, which do not implement `Hash`. An `LruIndexer` can also be used, but
can be slower and requires a sufficient capacity to work correctly.

The [ordered-float](https://crates.io/crates/ordered-float) crate is used by
the `ordered` module to ease this problem. Common geometric types implement
traits that provide conversions to and from a conjugate type that implements
`Hash` (via the `into_hash` and `from_hash` functions). Some geometric types
can be constructed from these conjugate types, as seen in the `Mesh` example.

The `ordered` module also exposes some hashing functions for floating point
primitives, which can be used to directly implement `Hash`. With the
[derivative](https://crates.io/crates/derivative) crate, floating point fields
can be hashed using one of these functions while deriving `Hash`. The `Vertex`
type used in the above example could be defined as follows:

```rust
use plexus::ordered;

#[derive(Derivative)]
#[derivative(Hash)]
pub struct Vertex {
    #[derivative(Hash(hash_with="ordered::hash_float_array"))]
    pub position: [f32; 3],
    ...
}
```
