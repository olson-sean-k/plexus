//! Polytope generation.
//!
//! This module provides a generic iterator and traits for generating polygons
//! and vertices containing geometric attributes of polytopes like cubes and
//! spheres. The [`Generate`] iterator can be used in iterator expressions.
//!
//! The primary API of this module is provided by the [`Generator`] trait, which
//! is implemented by polytope types like [`Cube`] and [`UvSphere`].
//!
//! # Examples
//!
//! Generating [raw buffers][`buffer`] from the positional data of a
//! [$uv$-sphere][`UvSphere`]:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use nalgebra::Point3;
//! use plexus::prelude::*;
//! use plexus::primitive::generate::Position;
//! use plexus::primitive::sphere::UvSphere;
//!
//! let sphere = UvSphere::new(16, 16);
//!
//! // Generate the unique set of positional vertices.
//! let positions = sphere
//!     .vertices::<Position<Point3<f64>>>()
//!     .collect::<Vec<_>>();
//!
//! // Generate polygons that index the unique set of positional vertices. The
//! // polygons are decomposed into triangles and then into vertices, where each
//! // vertex is an index into the position data.
//! let indices = sphere
//!     .indexing_polygons::<Position>()
//!     .triangulate()
//!     .vertices()
//!     .collect::<Vec<_>>();
//! ```
//!
//! [`buffer`]: crate::buffer
//! [`Cube`]: crate::primitive::cube::Cube
//! [`Generate`]: crate::primitive::generate::Generate
//! [`Generator`]: crate::primitive::generate::Generator
//! [`UvSphere`]: crate::primitive::sphere::UvSphere

use std::marker::PhantomData;
use std::ops::Range;

use crate::primitive::Polygonal;

/// Geometric attribute.
///
/// Types implementing this trait can be used with [`Generator`] to query
/// geometric attributes. For example, the [`Position`] type can be used to get
/// positional data for cubes or spheres via [`Cube`] and [`UvSphere`].
///
/// [`Cube`]: crate::primitive::cube::Cube
/// [`Generator`]: crate::primitive::generate::Generator
/// [`Position`]: crate::primitive::generate::Position
/// [`UvSphere`]: crate::primitive::sphere::UvSphere
pub trait Attribute {}

/// Meta-attribute for surface normals.
///
/// Describes the surface normals of a polytope. The generated data is derived
/// from the type parameter `S`, which typically requires [`EuclideanSpace`].
///
/// # Examples
///
/// Generating raw buffers with normal data of a [$uv$-sphere][`UvSphere`]:
///
/// ```rust
/// # extern crate decorum;
/// # extern crate nalgebra;
/// # extern crate plexus;
/// #
/// use decorum::R64;
/// use nalgebra::Point3;
/// use plexus::index::{Flat3, HashIndexer};
/// use plexus::prelude::*;
/// use plexus::primitive::generate::Normal;
/// use plexus::primitive::sphere::UvSphere;
///
/// let (indices, normals) = UvSphere::new(8, 8)
///     .polygons::<Normal<Point3<R64>>>()
///     .map_vertices(|normal| normal.into_inner())
///     .triangulate()
///     .index_vertices::<Flat3, _>(HashIndexer::default());
/// ```
///
/// [`EuclideanSpace`]: theon::space::EuclideanSpace
/// [`UvSphere`]: crate::primitive::sphere::UvSphere
pub struct Normal<S = ()> {
    phantom: PhantomData<fn() -> S>,
}

impl<S> Attribute for Normal<S> {}

/// Meta-attribute for positions.
///
/// Describes the position of vertices in a polytope. The generated data is
/// derived from the type parameter `S`, which typically requires
/// [`EuclideanSpace`].
///
/// # Examples
///
/// Generating raw buffers with positional data of a [cube][`Cube`]:
///
/// ```rust
/// # extern crate decorum;
/// # extern crate nalgebra;
/// # extern crate plexus;
/// #
/// use decorum::R64;
/// use nalgebra::Point3;
/// use plexus::index::{Flat3, HashIndexer};
/// use plexus::prelude::*;
/// use plexus::primitive::cube::Cube;
/// use plexus::primitive::generate::Position;
/// use plexus::primitive::UnboundedPolygon;
///
/// let (indices, positions) = Cube::new()
///     .polygons::<Position<Point3<R64>>>()
///     .triangulate()
///     .index_vertices::<UnboundedPolygon<usize>, _>(HashIndexer::default());
/// ```
///
/// [`EuclideanSpace`]: theon::space::EuclideanSpace
/// [`Cube`]: crate::primitive::cube::Cube
/// [`UvSphere`]: crate::primitive::sphere::UvSphere
pub struct Position<S = ()> {
    phantom: PhantomData<fn() -> S>,
}

impl<S> Attribute for Position<S> {}

/// Iterator that generates topology and geometric attributes.
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

/// Functions for iterating over the topology and geometry of polytopes.
pub trait Generator: Sized {
    /// Gets an iterator over the set of **unique** vertices with the given
    /// attribute data.
    ///
    /// Each geometric attribute has an independent set of unique values. For
    /// example, [`Cube`] generates six unique surface normals and eight unique
    /// positions.
    ///
    /// This can be paired with the
    /// [`indexing_polygons`][`Generator::indexing_polygons`] function to index
    /// the set of vertices.
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
    ///
    /// [`Cube`]: crate::primitive::cube::Cube
    /// [`Generator::indexing_polygons`]: crate::primitive::generate::Generator::indexing_polygons
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

    /// Gets an iterator over the set of polygons with the given attribute data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::R64;
    /// use nalgebra::Point3;
    /// use plexus::index::HashIndexer;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    /// use plexus::primitive::Tetragon;
    ///
    /// let (indices, positions) = Cube::new()
    ///     .polygons::<Position<Point3<R64>>>()
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

    /// Gets an iterator over a set of polygons that index the unique set of
    /// vertices with the given attribute.
    ///
    /// Indexing differs per geometric attribute, because each attribute has an
    /// independent set of unique values. For example, [`Cube`] generates six
    /// unique surface normals and eight unique positions.
    ///
    /// When used with meta-attribute types like [`Position`], input types are
    /// not needed and default type parameters can be used instead. For example,
    /// if `Position<Point3<f64>>` is used to generate positional data, then
    /// `Position<()>` (or `Position`) can be used to generate indexing
    /// polygons.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point3;
    /// use plexus::buffer::MeshBuffer4;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    ///
    /// type E3 = Point3<f64>;
    ///
    /// let cube = Cube::new();
    /// let buffer = MeshBuffer4::<usize, E3>::from_raw_buffers(
    ///     cube.indexing_polygons::<Position>(),
    ///     cube.vertices::<Position<E3>>(),
    /// );
    /// ```
    ///
    /// [`Cube`]: crate::primitive::cube::Cube
    /// [`Position`]: crate::primitive::generate::Position
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
