![Plexus](https://raw.githubusercontent.com/olson-sean-k/plexus/master/doc/plexus.png)

**Plexus** is a Rust library for 2D and 3D mesh processing.

[![Build Status](https://travis-ci.org/olson-sean-k/plexus.svg?branch=master)](https://travis-ci.org/olson-sean-k/plexus)
[![Documentation](https://docs.rs/plexus/badge.svg)](https://docs.rs/plexus)
[![Crate](https://img.shields.io/crates/v/plexus.svg)](https://crates.io/crates/plexus)

## Primitives and Iterator Expressions

Streams of geometric data can be generated from primitives like cubes and
spheres using iterator expressions. Primitives emit topological structures like
`Triangle`s or `Quad`s, which contain arbitrary data in their vertices. These
can be transformed and decomposed via tessellation and other operations.

```rust
use nalgebra::Point3;
use plexus::buffer::MeshBuffer3;
use plexus::prelude::*;
use plexus::primitive::sphere::UvSphere;

// Example module in the local crate that provides basic rendering.
use render::{self, Color, Vertex};

// Construct a linear buffer of index and vertex data from a sphere primitive.
let buffer = UvSphere::new(16, 16)
    .polygons_with_position()
    .map_vertices(|position| -> Point3<f64> { position.into() })
    .map_vertices(|position| Vertex::new(position, Color::white()))
    .triangulate()
    .collect::<MeshBuffer3<u32, Vertex>>();
render::draw(buffer.as_index_slice(), buffer.as_vertex_slice());
```

For an example of rendering, see the [viewer
example](https://github.com/olson-sean-k/plexus/tree/master/examples/viewer).

## Half-Edge Graph Meshes

`MeshGraph`, represented as a [half-edge
graph](https://en.wikipedia.org/wiki/doubly_connected_edge_list), supports
arbitrary geometry for vertices, arcs (half-edges), edges, and faces. Graphs
are persistent and can be traversed and manipulated in ways that iterator
expressions and linear buffers cannot, such as circulation, extrusion, merging,
and joining.

```rust
use nalgebra::Point3;
use plexus::graph::MeshGraph;
use plexus::prelude::*;
use plexus::primitive::sphere::{Bounds, UvSphere};

// Construct a mesh from a sphere primitive. The vertex geometry is convertible
// to `Point3` via the `FromGeometry` trait in this example.
let mut graph = UvSphere::new(8, 8)
    .polygons_with_position_from(Bounds::unit_width())
    .collect::<MeshGraph<Point3<f64>>>();
// Extrude a face in the mesh.
let abc = graph.faces().nth(0).unwrap().key();
if let Ok(face) = graph.face_mut(abc).unwrap().extrude(1.0) {
    // ...
}
```

Plexus avoids exposing very basic topological operations like inserting
individual vertices, because they can easily be done incorrectly. Instead,
meshes are typically manipulated with higher-level operations like splitting
and joining.

## Geometric Traits

Graphs support arbitrary geometry for vertices, arcs, edges, and faces
(including no geometry at all) via optional traits. Implementing these traits
enables geometric features, but only one trait is required: `Geometry`.

```rust
use decorum::R64;
use nalgebra::{Point3, Vector3};
use plexus::geometry::convert::AsPosition;
use plexus::geometry::Geometry;

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct VertexGeometry {
    pub position: Point3<R64>,
    pub normal: Vector3<R64>,
}

impl Geometry for VertexGeometry {
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl AsPosition for VertexGeometry {
    type Target = Point3<R64>;

    fn as_position(&self) -> &Self::Target {
        &self.position
    }

    fn as_position_mut(&mut self) -> &mut Self::Target {
        &mut self.position
    }
}
```

Geometric operations are vertex-based. By implementing `AsPosition` to expose
positional data from vertices and implementing geometric traits for that
positional data, operations like extrusion are exposed.

Geometric traits are optionally implemented for types in the
[cgmath](https://crates.io/crates/cgmath),
[mint](https://crates.io/crates/mint), and
[nalgebra](https://crates.io/crates/nalgebra) crates so that common types can be
used out-of-the-box for vertex geometry. See the `geometry-cgmath`,
`geometry-mint`, and `geometry-nalgebra` crate features. By default, the
`geometry-nalgebra` feature is enabled. Both 2D and 3D geometry are supported
by mesh operations.
