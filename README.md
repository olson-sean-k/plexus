![Plexus](https://raw.githubusercontent.com/olson-sean-k/plexus/master/doc/plexus.png)

**Plexus** is a Rust library for generating and manipulating 3D meshes.

[![Build Status](https://travis-ci.org/olson-sean-k/plexus.svg?branch=master)](https://travis-ci.org/olson-sean-k/plexus)
[![Build Status](https://ci.appveyor.com/api/projects/status/0uy6rcg3tvbu6cms?svg=true)](https://ci.appveyor.com/project/olson-sean-k/plexus)
[![Documentation](https://docs.rs/plexus/badge.svg)](https://docs.rs/plexus)
[![Crate](https://img.shields.io/crates/v/plexus.svg)](https://crates.io/crates/plexus)

## Generation and Iterator Expressions

Streams of topological and geometric data can be generated from primitives like
cubes and spheres using iterator expressions. Primitives emit topological
structures like `Triangle`s or `Quad`s, which contain arbitrary geometric data
in their vertices. These can be transformed and decomposed into other
topologies and geometric data via triangulation, tesselation, and other
operations.

```rust
use nalgebra::Point3;
use plexus::buffer::MeshBuffer;
use plexus::prelude::*;
use plexus::primitive::sphere::UvSphere;

// Example module in the local crate that provides basic rendering.
use render::{self, Color, Vertex};

// Construct a buffer of index and vertex data from a sphere primitive.
let buffer = UvSphere::new(16, 16)
    .polygons_with_position()
    .map_vertices(|position| -> Point3<f32> { position.into() })
    .map_vertices(|position| position * 10.0)
    .map_vertices(|position| Vertex::new(position, Color::white()))
    .collect::<MeshBuffer<u32, Vertex>>();
render::draw(buffer.as_index_slice(), buffer.as_vertex_slice());
```

For an example of rendering, see the [viewer
example](https://github.com/olson-sean-k/plexus/tree/master/examples/viewer).

## Half-Edge Graph Meshes

Generators produce an ephemeral stream of topology and vertex geometry. A
`Mesh`, represented as a [half-edge
graph](https://en.wikipedia.org/wiki/doubly_connected_edge_list), supports
arbitrary geometry for vertices, edges, and faces. The graph can also be
traversed and manipulated in ways that generators and iterator expressions
cannot, such as circulation, extrusion, merging, and joining.

```rust
use nalgebra::Point3;
use plexus::graph::Mesh;
use plexus::prelude::*;
use plexus::primitive::sphere::{Bounds, UvSphere};

// Construct a mesh from a sphere primitive. The vertex geometry is convertible
// to `Point3` via the `FromGeometry` trait in this example.
let mut mesh = sphere::UvSphere::new(8, 8)
    .polygons_with_position_from(Bounds::unit_width())
    .collect::<Mesh<Point3<f32>>>();
// Extrude a face in the mesh.
let key = mesh.faces().nth(0).unwrap().key();
if let Ok(face) = mesh.face_mut(key).unwrap().extrude(1.0) {
    // ...
}
```

Plexus avoids exposing very basic topological operations like inserting
individual vertices, because they can easily be done incorrectly and lead to
invalid topologies. Instead, meshes are manipulated with higher-level
operations like extrusion and joining.

## Geometric Traits

Meshes support arbitrary geometry for vertices, edges, and faces (including no
geometry at all) via optional traits. Implementing these traits enables more
operations and features, but only two basic traits are required: `Geometry` and
`Attribute`.

```rust
use nalgebra::{Point3, Vector3};
use plexus::geometry::{Attribute, Geometry};
use plexus::geometry::convert::AsPosition;

#[derive(Clone, Copy)]
pub struct VertexGeometry {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
}

impl Attribute for VertexGeometry {}

impl Geometry for VertexGeometry {
    type Vertex = Self;
    type Edge = ();
    type Face = ();
}

impl AsPosition for VertexGeometry {
    type Target = Point3<f32>;

    fn as_position(&self) -> &Self::Target {
        &self.position
    }

    fn as_position_mut(&mut self) -> &mut Self::Target {
        &mut self.position
    }
}
```

Geometric traits are optionally implemented for types in the
[nalgebra](https://crates.io/crates/nalgebra) and
[cgmath](https://crates.io/crates/cgmath) crates so that common types can be
used right away for vertex geometry. See the `geometry-cgmath` and
`geometry-nalgebra` (enabled by default) crate features. Both 2D and 3D
geometry are supported by mesh operations.

## Hashing Floating-Point Values

When collecting an iterator expression into a graph or buffer, an indexer is
used to transform the geometry into raw buffers. `HashIndexer` is fast and
reliable, and is used by `collect` (which can be overridden via
`collect_with_indexer`). However, geometry often contains floating point
values, which do not implement `Hash`. An `LruIndexer` can also be used, but
may be slower and requires a sufficient capacity to work correctly.

The [decorum](https://crates.io/crates/decorum) crate is used to ease this
problem. Hashable types like `NotNan`, `Finite`, `R32`, etc. can be used as
geometric data and are emitted by primitive generators like `UvSphere` and
`Cube`. Decorum can also be used to more easily make custom geometric data
hashable.
