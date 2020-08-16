//! Serialization and encodings.
//!
//! This module provides encoding support enabled via Cargo features. Each
//! enabled encoding has a corresponding sub-module. For example, when [PLY]
//! support is enabled, the `ply` module is exposed. The following table
//! summarizes the encodings supported by Plexus:
//!
//! | Feature        | Default | Encoding | Read | Write |
//! |----------------|---------|----------|------|-------|
//! | `encoding-ply` | No      | [PLY]    | Yes  | No    |
//!
//! This module provides traits used by all encodings. These traits describe the
//! outputs and inputs of decoders and encoders, respectively. Generally, these
//! traits should **not** be used directly. Instead, prefer the conversion
//! traits exposed for specific encodings, such as `FromPly` when using [PLY].
//!
//! [PLY]: https://en.wikipedia.org/wiki/ply_(file_format)

pub mod ply;

use std::fmt::Debug;

pub trait VertexDecoder {
    type Output: IntoIterator<Item = Self::Vertex>;
    type Vertex;
}

pub trait FaceDecoder {
    type Output: IntoIterator<Item = (Self::Index, Self::Face)>;
    type Index: IntoIterator<Item = usize>;
    type Face;
}

// TODO: This trait is a bit limiting. Consider implementing more specific
//       traits like `FromPly` directly. This could allow more specific
//       features to be supported, such as edge geometry for `MeshGraph`s.
pub trait FromEncoding<E>: Sized
where
    E: FaceDecoder + VertexDecoder,
{
    type Error: Debug;

    fn from_encoding(
        vertices: <E as VertexDecoder>::Output,
        faces: <E as FaceDecoder>::Output,
    ) -> Result<Self, Self::Error>;
}
