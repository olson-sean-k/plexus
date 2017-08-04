![Plexus](https://raw.githubusercontent.com/olson-sean-k/plexus/master/doc/plexus.png)

**Plexus** is a Rust library for generating, manipulating, and buffering 3D
meshes.

[![Build Status](https://travis-ci.org/olson-sean-k/plexus.svg?branch=master)](https://travis-ci.org/olson-sean-k/plexus)
[![Documentation](https://docs.rs/plexus/badge.svg)](https://docs.rs/plexus)

## Generation

Meshes can be generated from primitives using iterator expressions.

```rust
use nalgebra::Point3;
use plexus::generate::{sphere, MapVertices, SpatialPolygons, Subdivide, Triangulate};

let triangles = sphere::UVSphere::<f32>::with_unit_radius(16, 16)
    .spatial_polygons()
    .map_verticies(|(x, y, z)| Point3::new(x, y, z))
    .subdivide()
    .triangulate();
```

## Buffering

To render meshes, mesh data can be collecting into buffers that expose vertex
and index data.

```rust
use ordered_float::OrderedFloat;
use plexus::buffer::conjoint::ConjointBuffer;
use plexus::generate::{sphere, MapVertices, SpatialPolygons, Triangulate};

type OrdF<T> = OrderedFloat<T>;
type Point<T> = (OrdF<T>, OrdF<T>, OrdF<T>);

let buffer = sphere::UVSphere::<f32>::with_unit_radius(16, 16)
    .spatial_polygons()
    .map_verticies(|(x, y, z)| (OrdF(x), OrdF(y), OrdF(z)))
    .triangulate()
    .collect::<ConjointBuffer<_, Point<f32>>>();
```
