![Plexus](https://raw.githubusercontent.com/olson-sean-k/plexus/master/doc/plexus.png)

**Plexus** is a Rust library for generating and manipulating 3D meshes.

[![Build Status](https://travis-ci.org/olson-sean-k/plexus.svg?branch=master)](https://travis-ci.org/olson-sean-k/plexus)
[![Documentation](https://docs.rs/plexus/badge.svg)](https://docs.rs/plexus)
[![Crate](https://img.shields.io/crates/v/plexus.svg)](https://crates.io/crate/plexus)

## Generation and Iterator Expressions

Meshes can be generated from primitives like cubes and spheres using iterator
expressions. Primitives emit topological structures like `Triangle`s or
`Quads`, which contain arbitrary geometric data in their vertices. These can be
transformed and decomposed into other topologies and geometric data, including
triangulation, tesselation, and conversion into rendering pipeline data.

```rust
use ordered_float::OrderedFloat;
use plexus::buffer::conjoint::ConjointBuffer;
use plexus::generate::{sphere, MapVertices, SpatialPolygons};

use render::{self, Vertex}; // Module in the local crate providing rendering.

type r32 = OrderedFloat<f32>; // `f32` with `Eq` and `Hash` implementations.

// Construct a buffer of index and vertex data from a sphere primitive and
// render it. Note that `(r32, r32, r32)` is convertible to `Vertex` via the
// `From` trait in this example.
let buffer = sphere::UVSphere::<f32>::with_unit_radius(16, 16)
    .spatial_polygons()
    .map_verticies(|(x, y, z)| (r32(x), r32(y), r32(z)))
    .map_verticies(|(x, y, z)| (x * 10.0, y * 10.0, z * 10.0))
    .collect::<ConjointBuffer<_, Vertex>>();
render::draw(buffer.as_index_slice(), buffer.as_vertex_slice());
```
