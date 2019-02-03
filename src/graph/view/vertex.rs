use either::Either;
use fool::prelude::*;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};

use crate::geometry::Geometry;
use crate::graph::container::alias::OwnedCore;
use crate::graph::container::{Bind, Consistent, Reborrow, ReborrowMut};
use crate::graph::mutation::vertex::{self, VertexRemoveCache};
use crate::graph::mutation::{Mutate, Mutation};
use crate::graph::storage::convert::alias::*;
use crate::graph::storage::convert::{AsStorage, AsStorageMut};
use crate::graph::storage::{ArcKey, FaceKey, Storage, VertexKey};
use crate::graph::topology::{Arc, Face, Topological, Vertex};
use crate::graph::view::convert::{FromKeyedSource, IntoView};
use crate::graph::view::{ArcView, FaceView, OrphanArcView, OrphanFaceView};
use crate::graph::{GraphError, OptionExt};

/// Reference to a vertex.
///
/// Provides traversals, queries, and mutations related to vertices in a mesh.
/// See the module documentation for more information about topological views.
pub struct VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: Geometry,
{
    key: VertexKey,
    storage: M,
    phantom: PhantomData<G>,
}

/// Storage.
impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: Geometry,
{
    // TODO: This may become useful as the `mutation` module is developed. It
    //       may also be necessary to expose this API to user code.
    #[allow(dead_code)]
    pub(in crate::graph) fn bind<T, N>(self, storage: N) -> VertexView<<M as Bind<T, N>>::Output, G>
    where
        T: Topological,
        M: Bind<T, N>,
        M::Output: Reborrow,
        <M::Output as Reborrow>::Target: AsStorage<Vertex<G>>,
        N: AsStorage<T>,
    {
        let (key, origin) = self.into_keyed_storage();
        VertexView::from_keyed_storage_unchecked(key, origin.bind(storage))
    }
}

impl<'a, M, G> VertexView<&'a mut M, G>
where
    M: 'a + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: 'a + Geometry,
{
    /// Converts a mutable view into an orphan view.
    pub fn into_orphan(self) -> OrphanVertexView<'a, G> {
        let (key, storage) = self.into_keyed_storage();
        (key, storage).into_view().unwrap()
    }

    /// Converts a mutable view into an immutable view.
    ///
    /// This is useful when mutations are not (or no longer) needed and mutual
    /// access is desired.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// use nalgebra::Point3;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    ///
    /// # fn main() {
    /// let mut graph = Cube::new()
    ///     .polygons_with_position()
    ///     .collect::<MeshGraph<Point3<f32>>>();
    /// let key = graph.arcs().nth(0).unwrap().key();
    /// let vertex = graph.arc_mut(key).unwrap().split().unwrap().into_ref();
    ///
    /// // This would not be possible without conversion into an immutable view.
    /// let _ = vertex.into_outgoing_arc().into_face().unwrap();
    /// let _ = vertex
    ///     .into_outgoing_arc()
    ///     .into_opposite_arc()
    ///     .into_face()
    ///     .unwrap();
    /// # }
    /// ```
    pub fn into_ref(self) -> VertexView<&'a M, G> {
        let (key, storage) = self.into_keyed_storage();
        (key, &*storage).into_view().unwrap()
    }

    pub fn with_ref<T, K, F>(self, f: F) -> Either<Result<T, GraphError>, Self>
    where
        T: FromKeyedSource<(K, &'a mut M)>,
        F: FnOnce(VertexView<&M, G>) -> Option<K>,
    {
        if let Some(key) = f(self.interior_reborrow()) {
            let (_, storage) = self.into_keyed_storage();
            Either::Left(
                T::from_keyed_source((key, storage)).ok_or_else(|| GraphError::TopologyNotFound),
            )
        }
        else {
            Either::Right(self)
        }
    }
}

impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: Geometry,
{
    /// Gets the key for this vertex.
    pub fn key(&self) -> VertexKey {
        self.key
    }

    fn from_keyed_storage(key: VertexKey, storage: M) -> Option<Self> {
        storage
            .reborrow()
            .as_storage()
            .contains_key(&key)
            .some(VertexView::from_keyed_storage_unchecked(key, storage))
    }

    fn from_keyed_storage_unchecked(key: VertexKey, storage: M) -> Self {
        VertexView {
            key,
            storage,
            phantom: PhantomData,
        }
    }

    fn into_keyed_storage(self) -> (VertexKey, M) {
        let VertexView { key, storage, .. } = self;
        (key, storage)
    }

    fn interior_reborrow(&self) -> VertexView<&M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow();
        VertexView::from_keyed_storage_unchecked(key, storage)
    }
}

impl<M, G> VertexView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Vertex<G>>,
    G: Geometry,
{
    fn interior_reborrow_mut(&mut self) -> VertexView<&mut M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow_mut();
        VertexView::from_keyed_storage_unchecked(key, storage)
    }
}

/// Reachable API.
impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn into_reachable_outgoing_arc(self) -> Option<ArcView<M, G>> {
        let key = self.arc;
        key.and_then(move |key| {
            let (_, storage) = self.into_keyed_storage();
            (key, storage).into_view()
        })
    }

    pub(in crate::graph) fn reachable_outgoing_arc(&self) -> Option<ArcView<&M::Target, G>> {
        self.arc.and_then(|key| {
            let storage = self.storage.reborrow();
            (key, storage).into_view()
        })
    }

    pub(in crate::graph) fn reachable_incoming_arcs(
        &self,
    ) -> impl Clone + Iterator<Item = ArcView<&M::Target, G>> {
        ArcCirculator::from(self.interior_reborrow())
    }
}

impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent,
    G: Geometry,
{
    pub fn into_outgoing_arc(self) -> ArcView<M, G> {
        self.into_reachable_outgoing_arc().expect_consistent()
    }

    pub fn outgoing_arc(&self) -> ArcView<&M::Target, G> {
        self.reachable_outgoing_arc().expect_consistent()
    }

    pub fn incoming_arcs(&self) -> impl Clone + Iterator<Item = ArcView<&M::Target, G>> {
        self.reachable_incoming_arcs()
    }
}

/// Reachable API.
impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_neighboring_faces(
        &self,
    ) -> impl Clone + Iterator<Item = FaceView<&M::Target, G>> {
        FaceCirculator::from(ArcCirculator::from(self.interior_reborrow()))
    }
}

impl<M, G> VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    G: Geometry,
{
    pub fn neighboring_faces(&self) -> impl Clone + Iterator<Item = FaceView<&M::Target, G>> {
        self.reachable_neighboring_faces()
    }
}

/// Reachable API.
impl<M, G> VertexView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Arc<G>> + AsStorageMut<Arc<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_outgoing_orphan_arc(&mut self) -> Option<OrphanArcView<G>> {
        if let Some(key) = self.arc {
            (key, self.storage.reborrow_mut()).into_view()
        }
        else {
            None
        }
    }

    pub(in crate::graph) fn reachable_incoming_orphan_arcs(
        &mut self,
    ) -> impl Iterator<Item = OrphanArcView<G>> {
        ArcCirculator::from(self.interior_reborrow_mut())
    }
}

impl<M, G> VertexView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Arc<G>> + AsStorageMut<Arc<G>> + AsStorage<Vertex<G>> + Consistent,
    G: Geometry,
{
    pub fn outgoing_orphan_arc(&mut self) -> OrphanArcView<G> {
        self.reachable_outgoing_orphan_arc().expect_consistent()
    }

    pub fn incoming_orphan_arcs(&mut self) -> impl Iterator<Item = OrphanArcView<G>> {
        self.reachable_incoming_orphan_arcs()
    }
}

/// Reachable API.
impl<M, G> VertexView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target:
        AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_neighboring_orphan_faces(
        &mut self,
    ) -> impl Iterator<Item = OrphanFaceView<G>> {
        FaceCirculator::from(ArcCirculator::from(self.interior_reborrow_mut()))
    }
}

impl<M, G> VertexView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Arc<G>>
        + AsStorage<Face<G>>
        + AsStorageMut<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent,
    G: Geometry,
{
    pub fn neighboring_orphan_faces(&mut self) -> impl Iterator<Item = OrphanFaceView<G>> {
        self.reachable_neighboring_orphan_faces()
    }
}

impl<'a, M, G> VertexView<&'a mut M, G>
where
    M: AsStorage<Arc<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Default
        + From<OwnedCore<G>>
        + Into<OwnedCore<G>>,
    G: 'a + Geometry,
{
    pub fn remove(self) -> Result<(), GraphError> {
        let (a, storage) = self.into_keyed_storage();
        let cache = VertexRemoveCache::snapshot(&storage, a)?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| vertex::remove_with_cache(mutation, cache))
            .map(|_| ())
    }
}

impl<M, G> Clone for VertexView<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: Geometry,
{
    fn clone(&self) -> Self {
        VertexView {
            key: self.key,
            storage: self.storage.clone(),
            phantom: PhantomData,
        }
    }
}

impl<M, G> Copy for VertexView<M, G>
where
    M: Copy + Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: Geometry,
{
}

impl<M, G> Deref for VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: Geometry,
{
    type Target = Vertex<G>;

    fn deref(&self) -> &Self::Target {
        self.storage.reborrow().as_storage().get(&self.key).unwrap()
    }
}

impl<M, G> DerefMut for VertexView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.storage
            .reborrow_mut()
            .as_storage_mut()
            .get_mut(&self.key)
            .unwrap()
    }
}

impl<M, G> FromKeyedSource<(VertexKey, M)> for VertexView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Vertex<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (VertexKey, M)) -> Option<Self> {
        let (key, storage) = source;
        VertexView::from_keyed_storage(key, storage)
    }
}

/// Orphan reference to a vertex.
///
/// Consider using `OrphanVertex` instead. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
pub struct OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    key: VertexKey,
    vertex: &'a mut Vertex<G>,
}

impl<'a, G> OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    pub fn key(&self) -> VertexKey {
        self.key
    }
}

impl<'a, G> Deref for OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    type Target = Vertex<G>;

    fn deref(&self) -> &Self::Target {
        &*self.vertex
    }
}

impl<'a, G> DerefMut for OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.vertex
    }
}

impl<'a, M, G> FromKeyedSource<(VertexKey, &'a mut M)> for OrphanVertexView<'a, G>
where
    M: AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (VertexKey, &'a mut M)) -> Option<Self> {
        let (key, storage) = source;
        storage
            .as_storage_mut()
            .get_mut(&key)
            .map(|vertex| OrphanVertexView { key, vertex })
    }
}

impl<'a, G> FromKeyedSource<(VertexKey, &'a mut Vertex<G>)> for OrphanVertexView<'a, G>
where
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (VertexKey, &'a mut Vertex<G>)) -> Option<Self> {
        let (key, vertex) = source;
        Some(OrphanVertexView { key, vertex })
    }
}

struct ArcCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>>,
    G: Geometry,
{
    storage: M,
    outgoing: Option<ArcKey>,
    breadcrumb: Option<ArcKey>,
    phantom: PhantomData<G>,
}

impl<M, G> ArcCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>>,
    G: Geometry,
{
    fn next(&mut self) -> Option<ArcKey> {
        self.outgoing
            .map(|outgoing| outgoing.opposite())
            .and_then(|incoming| {
                self.storage
                    .reborrow()
                    .as_storage()
                    .get(&incoming)
                    .map(|incoming| incoming.next)
                    .map(|outgoing| (incoming, outgoing))
            })
            .and_then(|(incoming, outgoing)| {
                self.breadcrumb.map(|_| {
                    if self.breadcrumb == outgoing {
                        self.breadcrumb = None;
                    }
                    else {
                        self.outgoing = outgoing;
                    }
                    incoming
                })
            })
    }
}

impl<M, G> Clone for ArcCirculator<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<Arc<G>>,
    G: Geometry,
{
    fn clone(&self) -> Self {
        ArcCirculator {
            storage: self.storage.clone(),
            outgoing: self.outgoing,
            breadcrumb: self.breadcrumb,
            phantom: PhantomData,
        }
    }
}

impl<M, G> From<VertexView<M, G>> for ArcCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    fn from(vertex: VertexView<M, G>) -> Self {
        let key = vertex.arc;
        let (_, storage) = vertex.into_keyed_storage();
        ArcCirculator {
            storage,
            outgoing: key,
            breadcrumb: key,
            phantom: PhantomData,
        }
    }
}

impl<'a, M, G> Iterator for ArcCirculator<&'a M, G>
where
    M: 'a + AsStorage<Arc<G>>,
    G: 'a + Geometry,
{
    type Item = ArcView<&'a M, G>;

    fn next(&mut self) -> Option<Self::Item> {
        ArcCirculator::next(self).and_then(|key| (key, self.storage).into_view())
    }
}

impl<'a, M, G> Iterator for ArcCirculator<&'a mut M, G>
where
    M: 'a + AsStorage<Arc<G>> + AsStorageMut<Arc<G>>,
    G: 'a + Geometry,
{
    type Item = OrphanArcView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        ArcCirculator::next(self).and_then(|key| {
            (key, unsafe {
                // Apply `'a` to the autoref from `reborrow_mut`,
                // `as_storage_mut`, and `get_mut`.
                mem::transmute::<&'_ mut Storage<Arc<G>>, &'a mut Storage<Arc<G>>>(
                    self.storage.as_storage_mut(),
                )
            })
                .into_view()
        })
    }
}

struct FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    input: ArcCirculator<M, G>,
}

impl<M, G> FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    fn next(&mut self) -> Option<FaceKey> {
        while let Some(arc) = self.input.next() {
            if let Some(face) = self
                .input
                .storage
                .reborrow()
                .as_arc_storage()
                .get(&arc)
                .and_then(|arc| arc.face)
            {
                return Some(face);
            }
            else {
                // Skip arcs with no face. This can occur within non
                // -enclosed meshes.
                continue;
            }
        }
        None
    }
}

impl<M, G> Clone for FaceCirculator<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    fn clone(&self) -> Self {
        FaceCirculator {
            input: self.input.clone(),
        }
    }
}

impl<M, G> From<ArcCirculator<M, G>> for FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    fn from(input: ArcCirculator<M, G>) -> Self {
        FaceCirculator { input }
    }
}

impl<'a, M, G> Iterator for FaceCirculator<&'a M, G>
where
    M: 'a + AsStorage<Arc<G>> + AsStorage<Face<G>>,
    G: 'a + Geometry,
{
    type Item = FaceView<&'a M, G>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).and_then(|key| (key, self.input.storage).into_view())
    }
}

impl<'a, M, G> Iterator for FaceCirculator<&'a mut M, G>
where
    M: 'a + AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: 'a + Geometry,
{
    type Item = OrphanFaceView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).and_then(|key| {
            (key, unsafe {
                // Apply `'a` to the autoref from `reborrow_mut`,
                // `as_storage_mut`, and `get_mut`.
                mem::transmute::<&'_ mut Storage<Face<G>>, &'a mut Storage<Face<G>>>(
                    self.input.storage.as_storage_mut(),
                )
            })
                .into_view()
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use crate::graph::*;
    use crate::primitive::generate::*;
    use crate::primitive::sphere::UvSphere;

    #[test]
    fn circulate_over_arcs() {
        let graph = UvSphere::new(4, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f32>>>();

        // All faces should be triangles and all vertices should have 4
        // (incoming) arcs.
        for vertex in graph.vertices() {
            assert_eq!(4, vertex.incoming_arcs().count());
        }
    }
}
