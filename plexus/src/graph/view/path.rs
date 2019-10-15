use fool::BoolExt;
use indexmap::{indexset, IndexSet};
use itertools::Itertools;
use std::borrow::Borrow;
use std::marker::PhantomData;

use crate::graph::borrow::Reborrow;
use crate::graph::geometry::GraphGeometry;
use crate::graph::mutation::Consistent;
use crate::graph::storage::alias::*;
use crate::graph::storage::key::{ArcKey, VertexKey};
use crate::graph::storage::payload::{ArcPayload, VertexPayload};
use crate::graph::storage::{AsStorage, AsStorageMut};
use crate::graph::view::edge::ArcView;
use crate::graph::view::vertex::VertexView;
use crate::graph::view::{FromKeyedSource, IntoView};
use crate::graph::{GraphError, OptionExt, Selector};

// TODO: It would probably better for `PathView` to behave as a double-ended
//       queue. That would avoid the need for `pop_swap` and allow paths to be
//       re-rooted (the initiating vertex could be changed). This would not
//       work well with `IndexSet`.

/// View of a path.
///
/// Provides a representation of **non-intersecting** paths in a graph. A path
/// is conceptually an ordered set of vertices that are joined by arcs. A path
/// over vertices $A$, $B$, and $C$ is notated $\overrightarrow{\\{A, B,
/// C\\}}$.
#[derive(Clone)]
pub struct PathView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<VertexPayload<G>> + Consistent,
    G: GraphGeometry,
{
    // Paths are represented using a head and tail, where the head is the
    // initiating vertex key and the tail is an ordered set of vertex keys. It
    // is possible for the first and last (that is, the head and terminating
    // element of the tail) to be the same vertex key, in which case the path
    // is closed.
    head: VertexKey,
    tail: IndexSet<VertexKey>,
    storage: M,
    phantom: PhantomData<G>,
}

impl<M, G> PathView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<VertexPayload<G>> + Consistent,
    G: GraphGeometry,
{
    /// Pushes a vertex onto the path.
    pub fn push(&mut self, destination: Selector<VertexKey>) -> Result<ArcKey, GraphError> {
        self.is_open()
            .ok_or_else(|| GraphError::TopologyMalformed)?;
        let a = self.last();
        let b = match destination {
            Selector::ByKey(b) => {
                self.storage
                    .reborrow()
                    .as_vertex_storage()
                    .contains_key(&b)
                    .ok_or_else(|| GraphError::TopologyNotFound)?;
                self.storage
                    .reborrow()
                    .as_arc_storage()
                    .contains_key(&(a, b).into())
                    .ok_or_else(|| GraphError::TopologyMalformed)?;
                b
            }
            Selector::ByIndex(index) => {
                let storage = self.storage.reborrow();
                let vertex = VertexView::from_keyed_source((a, storage)).expect_consistent();
                let vertex = vertex
                    .neighboring_vertices()
                    .nth(index)
                    .ok_or_else(|| GraphError::TopologyNotFound)?;
                vertex.key()
            }
        };
        // Do not allow intersections unless they form a loop with the first
        // vertex in the path.
        //
        // Note that resolving the key `b` above already implicitly prohibits
        // `a == b`, so there is no need to test that here.
        (!self.tail.contains(&b)).ok_or_else(|| GraphError::TopologyMalformed)?;
        Ok(self.push_unchecked(b))
    }

    /// Pops the terminating vertex from the path.
    pub fn pop(&mut self) -> Option<ArcKey> {
        if self.tail.len() > 1 {
            Some(self.pop_unchecked())
        }
        else {
            None
        }
    }

    /// Exchanges the terminating vertex of the path.
    ///
    /// Unlike `pop`, this function allows an existing path's originating arc
    /// to be changed by exchanging the destination vertex.
    pub fn pop_swap(&mut self, destination: Selector<VertexKey>) -> Option<ArcKey> {
        if !self.tail.is_empty() {
            // This pop operation may exhaust the tail and must be followed by
            // a push.
            let ab = self.pop_unchecked();
            let (_, b) = ab.into();
            match self.push(destination) {
                Ok(ab) => Some(ab),
                _ => {
                    // Restore the original path.
                    self.push_unchecked(b);
                    None
                }
            }
        }
        else {
            None
        }
    }

    /// Gets an iterator over the keys of the vertices in the path.
    pub fn keys(&self) -> impl Iterator<Item = &VertexKey> {
        let head = Some(&self.head);
        head.into_iter().chain(self.tail.iter())
    }

    /// Gets the initiating vertex of the path.
    pub fn first(&self) -> VertexKey {
        self.head
    }

    /// Gets the terminating vertex of the path.
    pub fn last(&self) -> VertexKey {
        self.tail.iter().cloned().last().expect("path tail empty")
    }

    /// Gets an iterator over the vertices in the path.
    pub fn vertices(&self) -> impl Iterator<Item = VertexView<&M::Target, G>> {
        let storage = self.storage.reborrow();
        self.keys()
            .cloned()
            .map(move |key| (key, storage).into_view().expect_consistent())
    }

    /// Gets an iterator over the arcs in the path.
    pub fn arcs(&self) -> impl Iterator<Item = ArcView<&M::Target, G>> {
        // Get the outgoing arc of each vertex, but skip the terminating
        // vertex.
        self.vertices()
            .tuple_windows()
            .map(|(vertex, _)| vertex.into_outgoing_arc())
    }

    /// Returns `true` if the path is open.
    ///
    /// An _open path_ is a path that does **not** form a loop.
    pub fn is_open(&self) -> bool {
        !self.is_closed()
    }

    /// Returns `true` if the path is closed.
    ///
    /// A _closed path_ is a path that forms a loop by starting and ending at
    /// the same vertex. Note that this exludes paths that self-intersect and
    /// include arcs that do not participate in loops. `PathView` disallows
    /// such paths.
    pub fn is_closed(&self) -> bool {
        self.first() == self.last()
    }

    fn push_unchecked(&mut self, b: VertexKey) -> ArcKey {
        let a = self.last();
        self.tail.insert(b);
        (a, b).into()
    }

    fn pop_unchecked(&mut self) -> ArcKey {
        let a = self.head;
        let b = self.tail.pop().expect("path tail empty");
        (a, b).into()
    }
}

impl<'a, M, G> PathView<&'a mut M, G>
where
    M: 'a
        + AsStorage<ArcPayload<G>>
        + AsStorage<VertexPayload<G>>
        + AsStorageMut<ArcPayload<G>>
        + AsStorageMut<VertexPayload<G>>
        + Consistent,
    G: 'a + GraphGeometry,
{
    /// Converts a mutable view into an immutable view.
    ///
    /// This is useful when mutations are not (or no longer) needed and mutual
    /// access is desired.
    pub fn into_ref(self) -> PathView<&'a M, G> {
        let PathView {
            head,
            tail,
            storage,
            ..
        } = self;
        PathView {
            head,
            tail,
            storage: &*storage,
            phantom: PhantomData,
        }
    }
}

// TODO: `FromKeyedSource` does not provide error information.

impl<I, M, G> FromKeyedSource<(I, M)> for PathView<M, G>
where
    I: IntoIterator,
    I::Item: Borrow<VertexKey>,
    M: Reborrow,
    M::Target: AsStorage<ArcPayload<G>> + AsStorage<VertexPayload<G>> + Consistent,
    G: GraphGeometry,
{
    fn from_keyed_source(source: (I, M)) -> Option<Self> {
        // TODO: `Result`s are mapped into `Option`s to comply with
        //       `FromKeyedSource`.
        let (keys, storage) = source;
        let mut keys = keys.into_iter().map(|key| *key.borrow());
        let head = keys
            .next()
            .ok_or_else(|| GraphError::TopologyNotFound)
            .ok()?;
        let tail = keys
            .next()
            .ok_or_else(|| GraphError::TopologyNotFound)
            .ok()?;
        let mut path = PathView {
            head,
            tail: indexset![tail],
            storage,
            phantom: PhantomData,
        };
        for key in keys {
            path.push(Selector::ByKey(key)).ok()?;
        }
        Some(path)
    }
}

#[cfg(test)]
mod tests {}
