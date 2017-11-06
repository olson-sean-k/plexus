# Viewer Example

This example renders a mesh with simple, custom geometry, using a generator,
graph, and buffer.

Run this example with `$ cargo run --example viewer`.

![Viewer](https://raw.githubusercontent.com/olson-sean-k/plexus/master/doc/viewer.png)

## Constructing the Mesh

The mesh is constructed in `main.rs`, which attempts to hide details of the
renderer. It uses a primitive generator (`UVSphere`) to generate a sphere and
convert its vertex geometry into a format compatible with the custom graph
geometry. The topology is then collected into a graph, which is used to
manipulate the sphere (extrusion, joining, etc.). The mesh is then converted
into a buffer that can be used for rendering.

## Rendering the Mesh

This example uses the `gfx` and `glutin` crates to render with OpenGL using a
very simple rendering pipeline. The mesh buffer is used as a vertex buffer and
index buffer that `gfx` can easily render.
