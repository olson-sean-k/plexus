![Plexus](https://raw.githubusercontent.com/olson-sean-k/plexus/master/doc/plexus.svg?sanitize=true)

**Plexus** is a Rust library for polygonal mesh processing.

[![Build Status](https://travis-ci.org/olson-sean-k/plexus.svg?branch=master)](https://travis-ci.org/olson-sean-k/plexus)
[![Documentation](https://docs.rs/plexus/badge.svg)](https://docs.rs/plexus)
[![Crate](https://img.shields.io/crates/v/plexus.svg)](https://crates.io/crates/plexus)
[![Chat](https://badges.gitter.im/plexus-rs/community.svg)](https://gitter.im/plexus-rs/community#)

## Primitives and Iterator Expressions

Plexus provides a rich set of primitive topological structures that can be
composed using generators and iterator expressions. Iterator expressions operate
over a sequence of polygons like `Trigon`s or `Tetragon`s with arbitrary data in
their vertices. These can be transformed, decomposed (tessellated), indexed, and
collected into mesh data structures.

```rust
use decorum::N64;
use nalgebra::Point3;
use plexus::buffer::MeshBuffer3;
use plexus::prelude::*;
use plexus::primitive::generate::Position;
use plexus::primitive::sphere::UvSphere;

// Example rendering module.
use render::{self, Color4, Vertex};

// Construct a linear buffer of index and vertex data from a sphere.
let buffer = UvSphere::new(16, 16)
    .polygons::<Position<Point3<N64>>>()
    .map_vertices(|position| Vertex::new(position, Color4::white()))
    .triangulate()
    .collect::<MeshBuffer3<u64, Vertex>>();
render::draw(buffer.as_index_slice(), buffer.as_vertex_slice());
```

The [`decorum`](https://crates.io/crates/decorum) crate is used for
floating-point values that can be hashed for fast indexing. See the
[sphere](https://github.com/olson-sean-k/plexus/tree/master/examples/sphere.rs)
and
[teapot](https://github.com/olson-sean-k/plexus/tree/master/examples/teapot.rs)
examples for rendering.

## Half-Edge Graphs

The `MeshGraph` type represents meshes as a [half-edge
graph](https://en.wikipedia.org/wiki/doubly_connected_edge_list) and supports
arbitrary geometry for vertices, arcs (half-edges), edges, and faces. Graphs
are persistent and can be traversed and manipulated in ways that iterator
expressions and linear buffers cannot, such as circulation, extrusion, merging,
and splitting.

```rust
use decorum::N64;
use nalgebra::Point3;
use plexus::graph::MeshGraph;
use plexus::prelude::*;
use plexus::primitive::generate::Position;
use plexus::primitive::sphere::{Bounds, UvSphere};

// Construct a mesh from a sphere.
let mut graph = UvSphere::new(8, 8)
    .polygons_from::<Position<Point3<N64>>>(Bounds::unit_width())
    .collect::<MeshGraph<Point3<f64>>>();
// Extrude a face in the mesh.
let key = graph.faces().nth(0).unwrap().key();
let face = graph.face_mut(key).unwrap().extrude(1.0);
```

Plexus avoids exposing very basic topological operations like inserting
individual vertices into a graph, because they can easily be done incorrectly.
Instead, graphs are typically manipulated with higher-level operations like
merging and splitting.

## Geometric Traits

Plexus provides optional traits to support vertex-based spatial operations by
exposing positional data in vertices. The
[`theon`](https://crates.io/crates/theon) crate is used to abstract Euclidean
spaces and if positional data supports these traits, then geometric operations
become available.

```rust
use decorum::N64;
use nalgebra::{Point3, Vector3};
use plexus::graph::GraphGeometry;
use plexus::AsPosition;

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct Vertex {
    pub position: Point3<N64>,
    pub normal: Vector3<N64>,
}

impl GraphGeometry for Vertex {
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl AsPosition for Vertex {
    type Position = Point3<N64>;

    fn as_position(&self) -> &Self::Position {
        &self.position
    }

    fn as_position_mut(&mut self) -> &mut Self::Position {
        &mut self.position
    }
}
```

By implementing `AsPosition` to expose positional data from vertices in a
graph, spatial operations like computation of normals, smoothing, and
topological mutations like poking faces become available. `MeshGraph` also
provides purely topological operations that allow user code to specify
arbitrary geometries without requiring these traits.

Geometric traits are optionally implemented for types in the
[`cgmath`](https://crates.io/crates/cgmath),
[`mint`](https://crates.io/crates/mint), and
[`nalgebra`](https://crates.io/crates/nalgebra) crates by enabling Cargo
features.

| Feature             | Default | Crate      |
|---------------------|---------|------------|
| `geometry-cgmath`   | No      | `cgmath`   |
| `geometry-mint`     | No      | `mint`     |
| `geometry-nalgebra` | No      | `nalgebra` |

If using one of the supported crates, it is highly recommended to enable the
corresponding feature.

## Encodings

Plexus provides support for polygonal mesh encodings. This allows `MeshGraph`s
and `MeshBuffer`s to be serialized and deserialized to and from various formats.

```rust
use nalgebra::Point3;
use plexus::encoding::ply::{FromPly, PositionEncoding};
use plexus::graph::MeshGraph;
use std::fs::File;

let ply = File::open("cube.ply").unwrap();
let encoding = PositionEncoding::<Point3<f64>>::default();
let (graph, _) = MeshGraph::<Point3<f64>>::from_ply(encoding, ply).unwrap();
```

Encoding support is optional and enabled via Cargo features.

| Feature        | Default | Encoding | Read | Write |
|----------------|---------|----------|------|-------|
| `encoding-ply` | No      | PLY      | Yes  | No    |
