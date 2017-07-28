//! Linear buffers that can be used for rendering.
//!
//! This module provides buffers that can be read by graphics pipelines to
//! render meshes. There are two basic types of buffers: _conjoint_ and
//! _disjoint_. Conjoint buffers store structured vertex data in a single
//! buffer, with each index in the index buffer referring to an element that
//! completely describes that vertex. Disjoint buffers store vertex data
//! components (position, normal, etc.) in separate buffers, using a structured
//! set of indeces for each vertex.

pub mod conjoint;
pub mod disjoint;
