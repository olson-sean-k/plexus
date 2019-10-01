pub mod edge;
pub mod face;
pub mod vertex;

use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};

use crate::graph::core::OwnedCore;
use crate::graph::geometry::GraphGeometry;
use crate::graph::mutation::face::FaceMutation;
use crate::graph::storage::alias::*;
use crate::graph::storage::payload::{ArcPayload, FacePayload, VertexPayload};
use crate::graph::storage::{AsStorage, StorageProxy};
use crate::graph::GraphError;

/// Marker trait for graph representations that promise to be in a consistent
/// state.
///
/// This trait is only implemented by representations that ensure that their
/// storage is only ever mutated via the mutation API (and therefore is
/// consistent). Note that `Core` does not implement this trait and instead
/// acts as a raw container for topological storage that can be freely
/// manipulated.
///
/// This trait allows code to make assumptions about the data it operates
/// against. For example, views expose an API to user code that assumes that
/// topologies are present and therefore unwraps values.
pub trait Consistent {}

impl<'a, T> Consistent for &'a T where T: Consistent {}

impl<'a, T> Consistent for &'a mut T where T: Consistent {}

pub trait Mutate: Sized {
    type Mutant;
    type Error: Debug;

    fn mutate(mutant: Self::Mutant) -> Self;

    fn replace(target: &mut Self::Mutant, replacement: Self::Mutant) -> Replace<Self>
    where
        Self::Mutant: Default,
    {
        Replace::replace(target, replacement)
    }

    fn commit(self) -> Result<Self::Mutant, Self::Error>;

    fn commit_with<F, T, E>(mut self, f: F) -> Result<(Self::Mutant, T), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<T, E>,
        E: Into<Self::Error>,
    {
        let output = f(&mut self);
        match output {
            Ok(value) => self.commit().map(|mutant| (mutant, value)),
            Err(error) => {
                self.abort();
                Err(error.into())
            }
        }
    }

    fn abort(self) {}
}

pub struct Replace<'a, M>
where
    M: Mutate,
    M::Mutant: Default,
{
    inner: Option<(&'a mut <M as Mutate>::Mutant, M)>,
}

impl<'a, M> Replace<'a, M>
where
    M: Mutate,
    M::Mutant: Default,
{
    pub fn replace(
        target: &'a mut <M as Mutate>::Mutant,
        replacement: <M as Mutate>::Mutant,
    ) -> Self {
        let mutant = mem::replace(target, replacement);
        Replace {
            inner: Some((target, M::mutate(mutant))),
        }
    }

    fn drain(&mut self) -> (&'a mut <M as Mutate>::Mutant, M) {
        self.inner.take().unwrap()
    }

    fn drain_and_commit(&mut self) -> Result<<Self as Mutate>::Mutant, <Self as Mutate>::Error> {
        let (target, inner) = self.drain();
        let mutant = inner.commit()?;
        mem::replace(target, mutant);
        Ok(target)
    }

    fn drain_and_abort(&mut self) {
        let (_, inner) = self.drain();
        inner.abort();
    }
}

impl<'a, M> AsRef<M> for Replace<'a, M>
where
    M: Mutate,
    M::Mutant: Default,
{
    fn as_ref(&self) -> &M {
        &self.inner.as_ref().unwrap().1
    }
}

impl<'a, M> AsMut<M> for Replace<'a, M>
where
    M: Mutate,
    M::Mutant: Default,
{
    fn as_mut(&mut self) -> &mut M {
        &mut self.inner.as_mut().unwrap().1
    }
}

impl<'a, M> Drop for Replace<'a, M>
where
    M: Mutate,
    M::Mutant: Default,
{
    fn drop(&mut self) {
        self.drain_and_abort()
    }
}

impl<'a, M> Mutate for Replace<'a, M>
where
    M: Mutate,
    M::Mutant: Default,
{
    type Mutant = &'a mut <M as Mutate>::Mutant;
    type Error = <M as Mutate>::Error;

    fn mutate(mutant: Self::Mutant) -> Self {
        Self::replace(mutant, Default::default())
    }

    fn commit(mut self) -> Result<Self::Mutant, Self::Error> {
        let mutant = self.drain_and_commit();
        mem::forget(self);
        mutant
    }

    fn abort(mut self) {
        self.drain_and_abort();
        mem::forget(self);
    }
}

/// Graph mutation.
pub struct Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    inner: FaceMutation<G>,
    phantom: PhantomData<M>,
}

impl<M, G> AsRef<Self> for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<M, G> AsMut<Self> for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

impl<M, G> AsStorage<ArcPayload<G>> for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    fn as_storage(&self) -> &StorageProxy<ArcPayload<G>> {
        self.inner.as_arc_storage()
    }
}

impl<M, G> AsStorage<FacePayload<G>> for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    fn as_storage(&self) -> &StorageProxy<FacePayload<G>> {
        self.inner.as_face_storage()
    }
}

impl<M, G> AsStorage<VertexPayload<G>> for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    fn as_storage(&self) -> &StorageProxy<VertexPayload<G>> {
        self.inner.as_vertex_storage()
    }
}

// TODO: This is a hack. Replace this with delegation.
impl<M, G> Deref for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    type Target = FaceMutation<G>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<M, G> DerefMut for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<M, G> Mutate for Mutation<M, G>
where
    M: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
    type Mutant = M;
    type Error = GraphError;

    fn mutate(mutant: Self::Mutant) -> Self {
        Mutation {
            inner: FaceMutation::mutate(mutant.into()),
            phantom: PhantomData,
        }
    }

    fn commit(self) -> Result<Self::Mutant, Self::Error> {
        self.inner.commit().map(|core| core.into())
    }
}

pub trait Mutable<G>: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>
where
    G: GraphGeometry,
{
}

impl<T, G> Mutable<G> for T
where
    T: Consistent + From<OwnedCore<G>> + Into<OwnedCore<G>>,
    G: GraphGeometry,
{
}
