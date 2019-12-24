//! Polytope generation.
//!
//! This module provides a generic iterator and traits for generating streams
//! of geometric and topological data for polytopes like cubes and spheres.
//!
//! The primary API of this module is exposed by the `Generator` trait.

use std::marker::PhantomData;
use std::ops::Range;

use crate::primitive::Polygonal;

/// Geometric attribute.
///
/// Types implementing this trait can be used with `Generator` to query
/// geometric attributes. For example, the `Position` type can be used to get
/// positional data for cubes or spheres via `Cube` and `UvSphere`.
pub trait Attribute {}

/// Meta-attribute for surface normals.
///
/// Describes the surface normals of a polytope. The generated data is derived
/// from the type parameter `S`, which typically requires `EuclideanSpace`.
///
/// # Examples
///
/// Generating raw buffers with normal data for a sphere:
///
/// ```rust
/// # extern crate decorum;
/// # extern crate nalgebra;
/// # extern crate plexus;
/// #
/// use decorum::N64;
/// use nalgebra::Point3;
/// use plexus::index::{Flat3, HashIndexer};
/// use plexus::prelude::*;
/// use plexus::primitive::generate::Normal;
/// use plexus::primitive::sphere::UvSphere;
///
/// let (indices, normals) = UvSphere::new(8, 8)
///     .polygons::<Normal<Point3<N64>>>()
///     .map_vertices(|normal| normal.into_inner())
///     .triangulate()
///     .index_vertices::<Flat3, _>(HashIndexer::default());
/// ```
pub struct Normal<S = ()> {
    phantom: PhantomData<S>,
}

impl<S> Attribute for Normal<S> {}

/// Meta-attribute for positions.
///
/// Describes the position of vertices in a polytope. The generated data is
/// derived from the type parameter `S`, which typically requires
/// `EuclideanSpace`.
///
/// # Examples
///
/// Generating raw buffers with positional data for a cube:
///
/// ```rust
/// # extern crate decorum;
/// # extern crate nalgebra;
/// # extern crate plexus;
/// #
/// use decorum::N64;
/// use nalgebra::Point3;
/// use plexus::index::{Flat3, HashIndexer};
/// use plexus::prelude::*;
/// use plexus::primitive::cube::Cube;
/// use plexus::primitive::generate::Position;
/// use plexus::primitive::Polygon;
///
/// let (indices, positions) = Cube::new()
///     .polygons::<Position<Point3<N64>>>()
///     .triangulate()
///     .index_vertices::<Polygon<usize>, _>(HashIndexer::default());
/// ```
pub struct Position<S = ()> {
    phantom: PhantomData<S>,
}

impl<S> Attribute for Position<S> {}

pub struct Generate<'a, G, S, P>
where
    G: 'a,
{
    generator: &'a G,
    state: S,
    range: Range<usize>,
    f: fn(&'a G, &S, usize) -> P,
}

impl<'a, G, S, P> Generate<'a, G, S, P>
where
    G: 'a,
{
    fn new(generator: &'a G, state: S, n: usize, f: fn(&'a G, &S, usize) -> P) -> Self {
        Generate {
            generator,
            state,
            range: 0..n,
            f,
        }
    }
}

impl<'a, G, S, P> Iterator for Generate<'a, G, S, P>
where
    G: 'a,
{
    type Item = P;

    fn next(&mut self) -> Option<Self::Item> {
        self.range
            .next()
            .map(|index| (self.f)(self.generator, &self.state, index))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

pub trait PolygonGenerator {
    fn polygon_count(&self) -> usize;
}

pub trait AttributeGenerator<A>
where
    A: Attribute,
{
    type State: Default;
}

pub trait AttributePolygonGenerator<A>: AttributeGenerator<A> + PolygonGenerator
where
    A: Attribute,
{
    type Output: Polygonal;

    fn polygon_from(&self, state: &Self::State, index: usize) -> Self::Output;
}

pub trait AttributeVertexGenerator<A>: AttributeGenerator<A>
where
    A: Attribute,
{
    type Output;

    fn vertex_count(&self) -> usize;

    fn vertex_from(&self, state: &Self::State, index: usize) -> Self::Output;
}

pub trait IndexingPolygonGenerator<A>: PolygonGenerator
where
    A: Attribute,
{
    type Output: Polygonal<Vertex = usize>;

    fn indexing_polygon(&self, index: usize) -> Self::Output;
}

/// Functions for iterating over the topological structures of generators.
pub trait Generator: Sized {
    /// Provides an iterator over the set of **unique** vertices with the given
    /// attribute data.
    ///
    /// Each geometric attribute has an independent set of unique values. For
    /// example, `Cube` generates six unique surface normals and eight unique
    /// positions.
    ///
    /// This can be paired with the `indexing_polygons` function to index the
    /// set of vertices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point3;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    ///
    /// type E3 = Point3<f64>;
    ///
    /// let cube = Cube::new();
    ///
    /// let positions = cube.vertices::<Position<E3>>().collect::<Vec<_>>();
    /// let indices = cube
    ///     .indexing_polygons::<Position>()
    ///     .triangulate()
    ///     .vertices()
    ///     .collect::<Vec<_>>();
    /// ```
    fn vertices<A>(
        &self,
    ) -> Generate<Self, Self::State, <Self as AttributeVertexGenerator<A>>::Output>
    where
        Self: AttributeVertexGenerator<A>,
        A: Attribute,
    {
        self.vertices_from(Default::default())
    }

    fn vertices_from<A>(
        &self,
        state: Self::State,
    ) -> Generate<Self, Self::State, <Self as AttributeVertexGenerator<A>>::Output>
    where
        Self: AttributeVertexGenerator<A>,
        A: Attribute,
    {
        Generate::new(self, state, self.vertex_count(), Self::vertex_from)
    }

    /// Provides an iterator over the set of polygons with the given
    /// attribute data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::N64;
    /// use nalgebra::Point3;
    /// use plexus::index::HashIndexer;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    /// use plexus::primitive::Tetragon;
    ///
    /// let (indices, positions) = Cube::new()
    ///     .polygons::<Position<Point3<N64>>>()
    ///     .index_vertices::<Tetragon<usize>, _>(HashIndexer::default());
    /// ```
    fn polygons<A>(
        &self,
    ) -> Generate<Self, Self::State, <Self as AttributePolygonGenerator<A>>::Output>
    where
        Self: AttributePolygonGenerator<A>,
        A: Attribute,
    {
        self.polygons_from(Default::default())
    }

    fn polygons_from<A>(
        &self,
        state: Self::State,
    ) -> Generate<Self, Self::State, <Self as AttributePolygonGenerator<A>>::Output>
    where
        Self: AttributePolygonGenerator<A>,
        A: Attribute,
    {
        Generate::new(self, state, self.polygon_count(), Self::polygon_from)
    }

    /// Provides an iterator over a set of polygons that index the unique set
    /// of vertices with the given attribute.
    ///
    /// Indexing differs per geometric attribute, because each attribute has an
    /// independent set of unique values. For example, `Cube` generates six
    /// unique surface normals and eight unique positions.
    ///
    /// When used with meta-attribute types like `Position`, input types are
    /// not needed and default type parameters can be used instead. For
    /// example, if `Position<Point3<f64>>` is used to generate positional
    /// data, then `Position<()>` or `Position` can be used to generate
    /// indexing polygons.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer3;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    ///
    /// let cube = Cube::new();
    /// let buffer = MeshBuffer3::<usize, _>::from_raw_buffers(
    ///     cube.indexing_polygons::<Position>().vertices(),
    ///     cube.vertices::<Position<Point3<f64>>>(),
    /// );
    /// ```
    fn indexing_polygons<A>(&self) -> Generate<Self, (), Self::Output>
    where
        Self: IndexingPolygonGenerator<A>,
        A: Attribute,
    {
        Generate::new(self, (), self.polygon_count(), |generator, _, index| {
            generator.indexing_polygon(index)
        })
    }
}
