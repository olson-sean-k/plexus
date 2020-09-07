<div align="center">
    <img alt="Plexus" src="https://raw.githubusercontent.com/olson-sean-k/plexus/master/doc/plexus.svg?sanitize=true" width="320"/>
</div>
<br/>

**Plexus** is a highly composable Rust library for polygonal mesh processing.
See [the website][website] for the most recent [API documentation][rustdoc] and
the [user guide][guide].

[![GitHub](https://img.shields.io/badge/GitHub-olson--sean--k/plexus-8da0cb?logo=github&style=for-the-badge)](https://github.com/olson-sean-k/plexus)
[![docs.rs](https://img.shields.io/badge/docs.rs-plexus-66c2a5?logo=rust&style=for-the-badge)](https://docs.rs/plexus)
[![crates.io](https://img.shields.io/crates/v/plexus.svg?logo=rust&style=for-the-badge)](https://crates.io/crates/plexus)
[![Gitter](https://img.shields.io/badge/Gitter-plexus--rs-c266a5?logo=gitter&style=for-the-badge)](https://gitter.im/plexus-rs/community)

## Primitives

Plexus provides a rich set of primitive topological structures that can be
composed using generators and iterator expressions. Iterator expressions operate
over a sequence of polygons with arbitrary vertex data. These polygons can be
decomposed, tessellated, indexed, and collected into mesh data structures.

```rust
use decorum::R64; // See "Integrations".
use nalgebra::Point3;
use plexus::buffer::MeshBuffer;
use plexus::index::Flat3;
use plexus::prelude::*;
use plexus::primitive::generate::Position;
use plexus::primitive::sphere::UvSphere;

use crate::render::pipeline::{Color4, Vertex};

type E3 = Point3<R64>;

// Create a buffer of index and vertex data from a uv-sphere.
let buffer: MeshBuffer<Flat3, Vertex> = UvSphere::new(16, 8)
    .polygons::<Position<E3>>()
    .map_vertices(|position| Vertex::new(position, Color4::white()))
    .triangulate()
    .collect();
```

The above example uses a generator and iterator expression to transform the
positional data of a sphere into a linear buffer for indexed drawing. See [the
sphere example][example-sphere] for a rendered demonstration.

## Half-Edge Graphs

The `MeshGraph` type represents polygonal meshes as an ergonomic [half-edge
graph][dcel] that supports arbitrary data in vertices, arcs (half-edges), edges,
and faces. Graphs can be traversed and manipulated in many ways that iterator
expressions and linear buffers cannot.

```rust
use ultraviolet::vec::Vec3;
use plexus::graph::MeshGraph;
use plexus::prelude::*;
use plexus::primitive::Tetragon;

type E3 = Vec3;

// Create a graph of a tetragon.
let mut graph = MeshGraph::<E3>::from(Tetragon::from([
    (1.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (-1.0, -1.0, 0.0),
    (1.0, -1.0, 0.0),
]));
// Extrude the face forming the tetragon to form a cube.
let key = graph.faces().nth(0).unwrap().key();
face = graph.face_mut(key).unwrap().extrude_with_offset(1.0).unwrap();
```

Plexus avoids exposing very basic topological operations like inserting
individual vertices into a graph, because they can easily be done incorrectly.
Instead, graphs are typically manipulated with more abstract operations like
merging and splitting.

See [the user guide][guide-graphs] for more details about graphs.

## Geometric Traits

Plexus provides optional traits to support spatial operations by exposing
positional data in vertices. If the data exposed by the `AsPosition` trait
supports these geometric traits, then geometric operations become available in
primitive and mesh data structure APIs.

```rust
use glam::Vec3A;
use plexus::geometry::{AsPosition, Vector};
use plexus::graph::GraphData;
use plexus::prelude::*;

type E3 = Vec3A;

#[derive(Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: E3,
    pub normal: Vector<E3>,
}

impl GraphData for Vertex {
    type Vertex = Self;
    type Arc = ();
    type Edge = ();
    type Face = ();
}

impl AsPosition for Vertex {
    type Position = E3;

    fn as_position(&self) -> &Self::Position {
        &self.position
    }
}
```

These traits enable APIs for offset extrusion, poking, smoothing, etc. Data
structures like `MeshGraph` also provide functions that allow user code to
specify arbitrary geometry without requiring any of these traits; the data in
these structures may be arbitrary, including no data at all.

## Integrations

Plexus integrates with the [`theon`] crate to provide geometric traits and
support various mathematics crates in the Rust ecosystem. Any mathematics crate
can be used and, if it is supported by Theon, Plexus provides geometric APIs by
enabling Cargo features.

| Feature                | Default | Crate           |
|------------------------|---------|-----------------|
| `geometry-cgmath`      | No      | [`cgmath`]      |
| `geometry-glam`        | No      | [`glam`]        |
| `geometry-mint`        | No      | [`mint`]        |
| `geometry-nalgebra`    | No      | [`nalgebra`]    |
| `geometry-ultraviolet` | No      | [`ultraviolet`] |

If using one of these supported crates, then enabling the corresponding feature
is highly recommended.

Plexus also integrates with the [`decorum`] crate for floating-point
representations that can be hashed for fast indexing. The `R64` type is a
(totally ordered) real number with an `f64` representation that cannot be `NaN`
nor infinity, for example. Geometric conversion traits are implemented for
supported types to allow for implicit conversions of scalar types.

## Encodings

Plexus provides support for polygonal mesh encodings. This allows mesh data
structures like `MeshGraph` and `MeshBuffer` to be serialized and deserialized
to and from various formats.

```rust
use nalgebra::Point3;
use plexus::encoding::ply::{FromPly, PositionEncoding};
use plexus::graph::MeshGraph;
use plexus::prelude::*;
use std::fs::File;

let ply = File::open("cube.ply").unwrap();
let encoding = PositionEncoding::<Point3<f64>>::default();
let (graph, _) = MeshGraph::<Point3<f64>>::from_ply(encoding, ply).unwrap();
```

Encoding support is optional and enabled via Cargo features.

| Feature        | Default | Encoding | Read | Write |
|----------------|---------|----------|------|-------|
| `encoding-ply` | No      | PLY      | Yes  | No    |

See [the teapot example][example-teapot] for a rendered demonstration of reading
a mesh from the file system.

[dcel]: https://en.wikipedia.org/wiki/doubly_connected_edge_list

[guide]: https://plexus.rs/user-guide/getting-started
[guide-graphs]: https://plexus.rs/user-guide/graphs
[rustdoc]: https://plexus.rs/rustdoc/plexus
[website]: https://plexus.rs

[example-sphere]: https://github.com/olson-sean-k/plexus/tree/master/examples/sphere/src/main.rs
[example-teapot]: https://github.com/olson-sean-k/plexus/tree/master/examples/teapot/src/main.rs

[`cgmath`]: https://crates.io/crates/cgmath
[`decorum`]: https://crates.io/crates/decorum
[`glam`]: https://crates.io/crates/glam
[`mint`]: https://crates.io/crates/mint
[`nalgebra`]: https://crates.io/crates/nalgebra
[`theon`]: https://crates.io/crates/theon
[`ultraviolet`]: https://crates.io/crates/ultraviolet
