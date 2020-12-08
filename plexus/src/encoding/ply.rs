//! [PLY] encoding.
//!
//! This module provides support for the [PLY] format via the [`FromPly`] and
//! [`ToPly`] traits. These traits can be used with a decoder and encoder to
//! read and write mesh data structures to and from the [PLY] format.
//!
//! [PLY] support is implemented using the [`ply-rs`] crate and some of its
//! types are re-exported here.
//!
//! # Examples
//!
//! Reading a [PLY] file into a [`MeshGraph`]:
//!
//! ```rust
//! # extern crate nalgebra;
//! # extern crate plexus;
//! #
//! use nalgebra::Point3;
//! use plexus::encoding::ply::{FromPly, PositionEncoding};
//! use plexus::graph::MeshGraph;
//! use std::io::Read;
//!
//! type E3 = Point3<f64>;
//!
//! // Read from a file, network, etc.
//! fn read() -> impl Read {
//!     // ...
//!     # let ply: &[u8] = include_bytes!("../../../data/cube.ply");
//!     # ply
//! }
//!
//! let encoding = PositionEncoding::<E3>::default();
//! let (graph, _) = MeshGraph::<E3>::from_ply(encoding, read()).unwrap();
//! ```
//!
//! [PLY]: https://en.wikipedia.org/wiki/PLY_(file_format)
//!
//! [`ply-rs`]: https://crates.io/crates/ply-rs
//!
//! [`FromPly`]: crate::encoding::ply::FromPly
//! [`ToPly`]: crate::encoding::ply::ToPly
//! [`MeshGraph`]: crate::graph::MeshGraph

#![cfg(feature = "encoding-ply")]

use num::cast;
use num::NumCast;
use ply_rs::parser::Parser;
use ply_rs::ply::KeyMap;
use smallvec::SmallVec;
use std::io::{self, Read, Write};
use std::iter::FromIterator;
use std::marker::PhantomData;
use theon::space::{EuclideanSpace, FiniteDimensional};
use thiserror::Error;
use typenum::{NonZero, Unsigned, U2, U3};

use crate::encoding::{FaceDecoder, FromEncoding, VertexDecoder};

pub use ply_rs::ply::{
    ElementDef as ElementDefinition, Property, PropertyDef as PropertyDefinition, PropertyType,
};

// TODO: These traits only allow a single element to be read for each topology
//       (vertices, faces, etc.). It may be useful to allow code to aggregate
//       various elements to produce an output for a single topology.
// TODO: Consider using the newtype pattern to hide underlying types and expose
//       a smaller and more tailored API surface.

pub type Header = KeyMap<ElementDefinition>;
pub type Payload = KeyMap<Vec<Element>>;
pub type Element = KeyMap<Property>;

pub struct Ply {
    pub header: Header,
    pub payload: Payload,
}

impl Ply {
    pub fn parse(mut read: impl Read) -> io::Result<Self> {
        Parser::<Element>::new().read_ply(&mut read).map(|ply| Ply {
            header: ply.header.elements,
            payload: ply.payload,
        })
    }
}

/// Errors concerning the [PLY] encoding.
///
/// [PLY]: https://en.wikipedia.org/wiki/PLY_(file_format)
#[derive(Debug, Error)]
pub enum PlyError {
    #[error("required element not found")]
    ElementNotFound,
    #[error("required property not found")]
    PropertyNotFound,
    /// The type of a property conflicts with a decoding.
    #[error("conflicting property type found")]
    PropertyTypeConflict,
    /// A polygonal mesh data structure is not compatible with encoded PLY data.
    #[error("encoding operation failed")]
    EncodingIncompatible,
    /// An I/O operation (read or write via the `Read` and `Write` traits)
    /// failed.
    #[error("I/O operation failed")]
    Io(io::Error),
}

impl From<io::Error> for PlyError {
    fn from(error: io::Error) -> Self {
        PlyError::Io(error)
    }
}

pub trait ElementExt {
    fn scalar<K, T>(&self, key: K) -> Result<T, PlyError>
    where
        K: AsRef<str>,
        T: NumCast;

    fn list<K, T, I>(&self, key: K) -> Result<I, PlyError>
    where
        K: AsRef<str>,
        T: NumCast,
        I: FromIterator<T>;
}

impl ElementExt for Element {
    fn scalar<K, T>(&self, key: K) -> Result<T, PlyError>
    where
        K: AsRef<str>,
        T: NumCast,
    {
        self.get(key.as_ref())
            .ok_or(PlyError::PropertyNotFound)?
            .clone()
            .into_scalar()
    }

    fn list<K, T, I>(&self, key: K) -> Result<I, PlyError>
    where
        K: AsRef<str>,
        T: NumCast,
        I: FromIterator<T>,
    {
        self.get(key.as_ref())
            .ok_or(PlyError::PropertyNotFound)?
            .clone()
            .into_list()
    }
}

pub trait PropertyExt {
    fn into_scalar<T>(self) -> Result<T, PlyError>
    where
        T: NumCast;

    fn into_list<T, I>(self) -> Result<I, PlyError>
    where
        T: NumCast,
        I: FromIterator<T>;
}

impl PropertyExt for Property {
    fn into_scalar<T>(self) -> Result<T, PlyError>
    where
        T: NumCast,
    {
        match self {
            Property::Char(value) => num_cast_scalar(value),
            Property::UChar(value) => num_cast_scalar(value),
            Property::Short(value) => num_cast_scalar(value),
            Property::UShort(value) => num_cast_scalar(value),
            Property::Int(value) => num_cast_scalar(value),
            Property::UInt(value) => num_cast_scalar(value),
            Property::Float(value) => num_cast_scalar(value),
            Property::Double(value) => num_cast_scalar(value),
            _ => Err(PlyError::PropertyTypeConflict),
        }
    }

    fn into_list<T, I>(self) -> Result<I, PlyError>
    where
        T: NumCast,
        I: FromIterator<T>,
    {
        match self {
            Property::ListChar(values) => num_cast_list(values),
            Property::ListUChar(values) => num_cast_list(values),
            Property::ListShort(values) => num_cast_list(values),
            Property::ListUShort(values) => num_cast_list(values),
            Property::ListInt(values) => num_cast_list(values),
            Property::ListUInt(values) => num_cast_list(values),
            Property::ListFloat(values) => num_cast_list(values),
            Property::ListDouble(values) => num_cast_list(values),
            _ => Err(PlyError::PropertyTypeConflict),
        }
    }
}

pub trait VertexElementDecoder {
    fn decode_vertex_elements<'a>(
        &self,
        definitions: &'a Header,
        elements: &'a Payload,
    ) -> Result<(&'a ElementDefinition, &'a Vec<Element>), PlyError> {
        decode_elements(definitions, elements, "vertex")
    }
}

pub trait VertexPropertyDecoder: VertexDecoder {
    fn decode_vertex_properties<'a, I>(
        &self,
        definition: &'a ElementDefinition,
        elements: I,
    ) -> Result<Self::Output, PlyError>
    where
        I: IntoIterator<Item = &'a Element>;
}

pub trait FaceElementDecoder {
    fn decode_face_elements<'a>(
        &self,
        definitions: &'a Header,
        elements: &'a Payload,
    ) -> Result<(&'a ElementDefinition, &'a Vec<Element>), PlyError> {
        decode_elements(definitions, elements, "face")
    }
}

pub trait FacePropertyDecoder: FaceDecoder {
    fn decode_face_properties<'a, I>(
        &self,
        definition: &'a ElementDefinition,
        elements: I,
    ) -> Result<Self::Output, PlyError>
    where
        I: IntoIterator<Item = &'a Element>;
}

pub trait FromPly<E>: Sized {
    fn from_ply(decoder: E, read: impl Read) -> Result<(Self, Ply), PlyError>;
}

impl<T, E> FromPly<E> for T
where
    T: FromEncoding<E>,
    E: FaceElementDecoder + FacePropertyDecoder + VertexPropertyDecoder + VertexElementDecoder,
{
    fn from_ply(decoder: E, mut read: impl Read) -> Result<(Self, Ply), PlyError> {
        let ply = Ply::parse(&mut read)?;
        let mesh = T::from_encoding(
            decode_vertex_properties(&decoder, &ply.header, &ply.payload)?,
            decode_face_properties(&decoder, &ply.header, &ply.payload)?,
        )
        .map_err(|_| PlyError::EncodingIncompatible)?;
        Ok((mesh, ply))
    }
}

pub trait ToPly<E> {
    fn to_ply(
        &self,
        definitions: &Header,
        encoder: E,
        write: impl Write,
    ) -> Result<usize, PlyError>;
}

pub trait DecodePosition<N>: FiniteDimensional<N = N> + Sized
where
    N: NonZero + Unsigned,
{
    fn decode_position(element: &Element) -> Result<Self, PlyError>;
}

impl<T> DecodePosition<U2> for T
where
    T: EuclideanSpace + FiniteDimensional<N = U2>,
{
    fn decode_position(element: &Element) -> Result<Self, PlyError> {
        let position = EuclideanSpace::from_xy(element.scalar("x")?, element.scalar("y")?);
        Ok(position)
    }
}

impl<T> DecodePosition<U3> for T
where
    T: EuclideanSpace + FiniteDimensional<N = U3>,
{
    fn decode_position(element: &Element) -> Result<Self, PlyError> {
        let position = EuclideanSpace::from_xyz(
            element.scalar("x")?,
            element.scalar("y")?,
            element.scalar("z")?,
        );
        Ok(position)
    }
}

pub struct PositionEncoding<T> {
    phantom: PhantomData<T>,
}

impl<T> Default for PositionEncoding<T> {
    fn default() -> Self {
        PositionEncoding {
            phantom: PhantomData,
        }
    }
}

impl<T> FaceDecoder for PositionEncoding<T> {
    type Output = Vec<(Self::Index, Self::Face)>;
    type Index = SmallVec<[usize; 4]>;
    type Face = ();
}

impl<T> FaceElementDecoder for PositionEncoding<T> {}

impl<T> FacePropertyDecoder for PositionEncoding<T> {
    fn decode_face_properties<'a, I>(
        &self,
        _: &'a ElementDefinition,
        elements: I,
    ) -> Result<<Self as FaceDecoder>::Output, PlyError>
    where
        I: IntoIterator<Item = &'a Element>,
    {
        elements
            .into_iter()
            .map(|element| {
                let indices = element.list("vertex_index")?;
                Ok((indices, ()))
            })
            .collect()
    }
}

impl<T> VertexDecoder for PositionEncoding<T> {
    type Output = Vec<Self::Vertex>;
    type Vertex = T;
}

impl<T> VertexElementDecoder for PositionEncoding<T> {}

impl<T, N> VertexPropertyDecoder for PositionEncoding<T>
where
    T: DecodePosition<N> + FiniteDimensional<N = N>,
    N: NonZero + Unsigned,
{
    fn decode_vertex_properties<'a, I>(
        &self,
        _: &'a ElementDefinition,
        elements: I,
    ) -> Result<<Self as VertexDecoder>::Output, PlyError>
    where
        I: IntoIterator<Item = &'a Element>,
    {
        elements
            .into_iter()
            .map(|element| T::decode_position(element))
            .collect()
    }
}

pub fn decode_elements<'a, K>(
    definitions: &'a Header,
    elements: &'a Payload,
    key: K,
) -> Result<(&'a ElementDefinition, &'a Vec<Element>), PlyError>
where
    K: AsRef<str>,
{
    definitions
        .get(key.as_ref())
        .ok_or(PlyError::ElementNotFound)
        .and_then(|definition| {
            elements
                .get(&definition.name)
                .ok_or(PlyError::ElementNotFound)
                .map(|elements| (definition, elements))
        })
}

pub fn decode_vertex_properties<E>(
    decoder: &E,
    definitions: &Header,
    elements: &Payload,
) -> Result<E::Output, PlyError>
where
    E: VertexElementDecoder + VertexPropertyDecoder,
{
    decoder
        .decode_vertex_elements(definitions, elements)
        .and_then(|(definition, elements)| decoder.decode_vertex_properties(definition, elements))
}

pub fn decode_face_properties<E>(
    decoder: &E,
    definitions: &Header,
    elements: &Payload,
) -> Result<E::Output, PlyError>
where
    E: FaceElementDecoder + FacePropertyDecoder,
{
    decoder
        .decode_face_elements(definitions, elements)
        .and_then(|(definition, elements)| decoder.decode_face_properties(definition, elements))
}

fn num_cast_scalar<T, U>(value: T) -> Result<U, PlyError>
where
    T: NumCast,
    U: NumCast,
{
    cast::cast(value).ok_or(PlyError::PropertyTypeConflict)
}

fn num_cast_list<T, U, I>(values: Vec<T>) -> Result<I, PlyError>
where
    T: NumCast,
    U: NumCast,
    I: FromIterator<U>,
{
    values
        .into_iter()
        .map(num_cast_scalar)
        .collect::<Result<_, _>>()
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use crate::buffer::MeshBuffer;
    use crate::encoding::ply::{FromPly, PositionEncoding};
    use crate::graph::MeshGraph;
    use crate::primitive::Tetragon;

    type E3 = Point3<f64>;

    #[test]
    fn decode_into_buffer() {
        let (buffer, _) = {
            let ply: &[u8] = include_bytes!("../../../data/cube.ply");
            MeshBuffer::<Tetragon<usize>, E3>::from_ply(PositionEncoding::<E3>::default(), ply)
                .unwrap()
        };
        assert_eq!(8, buffer.as_vertex_slice().len());
        assert_eq!(6, buffer.as_index_slice().len());
    }

    #[test]
    fn decode_into_graph() {
        let (graph, _) = {
            let ply: &[u8] = include_bytes!("../../../data/cube.ply");
            MeshGraph::<E3>::from_ply(PositionEncoding::<E3>::default(), ply).unwrap()
        };
        assert_eq!(8, graph.vertex_count());
        assert_eq!(12, graph.edge_count());
        assert_eq!(6, graph.face_count());
    }
}
