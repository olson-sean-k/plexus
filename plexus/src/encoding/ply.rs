//! [PLY](https://en.wikipedia.org/wiki/PLY_(file_format)) encoding.
//!
//! This module provides support for the PLY format via the `FromPly` and
//! `ToPly` traits. These traits can be used with a decoder and encoder to read
//! and write mesh data structures to and from the PLY format.
//!
//! PLY support is implemented using the
//! [`ply-rs`](https://crates.io/crates/ply-rs) crate.
//!
//! # Examples
//!
//! Reading a PLY file into a `MeshGraph`:
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
//! # fn main() {
//! let encoding = PositionEncoding::<E3>::default();
//! let (graph, _) = MeshGraph::<E3>::from_ply(encoding, read()).unwrap();
//! # }
//! ```

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
use typenum::U3;

use crate::buffer::BufferError;
use crate::encoding::{FaceDecoder, FromEncoding, VertexDecoder};
use crate::graph::GraphError;

pub use ply_rs::ply::{
    ElementDef as ElementDefinition, Property, PropertyDef as PropertyDefinition, PropertyType,
};

// TODO: These traits only allow a single element to be read for each topology
//       (vertices, faces, etc.). It may be useful to allow code to aggregate
//       various elements to produce an output for a single topology.
// TODO: Consider using the new-type pattern to hide underlying types and
//       expose a smaller and more tailored API surface.

pub type Header = KeyMap<ElementDefinition>;
pub type Payload = KeyMap<Vec<Element>>;
pub type Element = KeyMap<Property>;

#[derive(Debug)]
pub enum PlyError {
    ElementNotFound,
    PropertyNotFound,
    Encoding,
    Io(io::Error),
}

impl From<BufferError> for PlyError {
    fn from(_: BufferError) -> Self {
        PlyError::Encoding
    }
}

impl From<io::Error> for PlyError {
    fn from(error: io::Error) -> Self {
        PlyError::Io(error)
    }
}

impl From<GraphError> for PlyError {
    fn from(_: GraphError) -> Self {
        PlyError::Encoding
    }
}

pub trait ElementExt {
    fn read_scalar<T>(&self, key: &str) -> Result<T, PlyError>
    where
        T: NumCast;

    fn read_list<T, I>(&self, key: &str) -> Result<I, PlyError>
    where
        T: NumCast,
        I: FromIterator<T>;
}

impl ElementExt for Element {
    fn read_scalar<T>(&self, key: &str) -> Result<T, PlyError>
    where
        T: NumCast,
    {
        self.get(key)
            .ok_or_else(|| PlyError::PropertyNotFound)?
            .clone()
            .into_scalar()
    }

    fn read_list<T, I>(&self, key: &str) -> Result<I, PlyError>
    where
        T: NumCast,
        I: FromIterator<T>,
    {
        self.get(key)
            .ok_or_else(|| PlyError::PropertyNotFound)?
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
            _ => Err(PlyError::Encoding),
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
            _ => Err(PlyError::Encoding),
        }
    }
}

pub trait VertexElementDecoder {
    fn decode_vertex_elements<'a>(
        &self,
        definitions: &'a Header,
        elements: &'a Payload,
    ) -> Result<(&'a ElementDefinition, &'a Vec<Element>), PlyError> {
        definitions
            .get("vertex")
            .ok_or_else(|| PlyError::ElementNotFound)
            .map(|definition| (definition, elements.get(&definition.name).unwrap()))
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
        definitions
            .get("face")
            .ok_or_else(|| PlyError::ElementNotFound)
            .map(|definition| (definition, elements.get(&definition.name).unwrap()))
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
    fn from_ply<R>(decoder: E, read: R) -> Result<(Self, Header), PlyError>
    where
        R: Read;
}

impl<T, E> FromPly<E> for T
where
    T: FromEncoding<E>,
    PlyError: From<<T as FromEncoding<E>>::Error>,
    E: FaceElementDecoder + FacePropertyDecoder + VertexPropertyDecoder + VertexElementDecoder,
{
    fn from_ply<R>(decoder: E, mut read: R) -> Result<(Self, Header), PlyError>
    where
        R: Read,
    {
        let ply = Parser::<Element>::new().read_ply(&mut read)?;
        let vertices = decoder
            .decode_vertex_elements(&ply.header.elements, &ply.payload)
            .and_then(|(definition, elements)| {
                decoder.decode_vertex_properties(definition, elements)
            })?;
        let faces = decoder
            .decode_face_elements(&ply.header.elements, &ply.payload)
            .and_then(|(definition, elements)| {
                decoder.decode_face_properties(definition, elements)
            })?;
        let mesh = T::from_encoding(vertices, faces)?;
        Ok((mesh, ply.header.elements))
    }
}

pub trait ToPly<E> {
    fn to_ply<W>(&self, definitions: Header, encoder: E, write: W) -> Result<usize, PlyError>
    where
        W: Write;
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
                let indices = element.read_list("index")?;
                Ok((indices, ()))
            })
            .collect::<Result<_, _>>()
    }
}

impl<T> VertexDecoder for PositionEncoding<T> {
    type Output = Vec<Self::Vertex>;
    type Vertex = T;
}

impl<T> VertexElementDecoder for PositionEncoding<T> {}

// TODO: Support two-dimensional spaces.
impl<T> VertexPropertyDecoder for PositionEncoding<T>
where
    T: EuclideanSpace + FiniteDimensional<N = U3>,
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
            .map(|element| {
                let vertex = EuclideanSpace::from_xyz(
                    element.read_scalar("x")?,
                    element.read_scalar("y")?,
                    element.read_scalar("z")?,
                );
                Ok(vertex)
            })
            .collect::<Result<_, _>>()
    }
}

fn num_cast_scalar<T, U>(value: T) -> Result<U, PlyError>
where
    T: NumCast,
    U: NumCast,
{
    cast::cast(value).ok_or_else(|| PlyError::Encoding)
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

    use crate::encoding::ply::{FromPly, PositionEncoding};
    use crate::graph::MeshGraph;

    type E3 = Point3<f64>;

    #[test]
    fn decode() {
        let graph = {
            let ply: &[u8] = include_bytes!("../../../data/cube.ply");
            MeshGraph::<E3>::from_ply(PositionEncoding::<E3>::default(), ply)
                .unwrap()
                .0
        };
        assert_eq!(8, graph.vertex_count());
        assert_eq!(12, graph.edge_count());
        assert_eq!(6, graph.face_count());
    }
}
