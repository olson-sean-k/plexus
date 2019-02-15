use either::Either;
use fool::prelude::*;
use std::cmp;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Add, Deref, DerefMut, Mul};

use crate::geometry::alias::{ScaledFaceNormal, VertexPosition};
use crate::geometry::convert::AsPosition;
use crate::geometry::Geometry;
use crate::graph::container::alias::OwnedCore;
use crate::graph::container::{Bind, Consistent, Reborrow, ReborrowMut};
use crate::graph::geometry::{FaceCentroid, FaceNormal};
use crate::graph::mutation::face::{
    self, FaceBisectCache, FaceBridgeCache, FaceDivergeCache, FaceExtrudeCache, FaceInsertCache,
    FaceRemoveCache,
};
use crate::graph::mutation::{Mutate, Mutation};
use crate::graph::storage::convert::{AsStorage, AsStorageMut};
use crate::graph::storage::{ArcKey, FaceKey, Storage, VertexKey};
use crate::graph::topology::{Arc, Edge, Face, Topological, Vertex};
use crate::graph::view::convert::{FromKeyedSource, IntoKeyedSource, IntoView};
use crate::graph::view::{
    ArcNeighborhood, ArcView, OrphanArcView, OrphanVertexView, Selector, VertexView,
};
use crate::graph::{GraphError, OptionExt};

use Selector::ByIndex;

/// Reference to a face.
///
/// Provides traversals, queries, and mutations related to faces in a mesh. See
/// the module documentation for more information about topological views.
pub struct FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    key: FaceKey,
    storage: M,
    phantom: PhantomData<G>,
}

/// Storage.
impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    // TODO: This may become useful as the `mutation` module is developed. It
    //       may also be necessary to expose this API to user code.
    #[allow(dead_code)]
    pub(in crate::graph) fn bind<T, N>(self, storage: N) -> FaceView<<M as Bind<T, N>>::Output, G>
    where
        T: Topological,
        M: Bind<T, N>,
        M::Output: Reborrow,
        <M::Output as Reborrow>::Target: AsStorage<Face<G>>,
        N: AsStorage<T>,
    {
        let (key, origin) = self.into_keyed_source();
        FaceView::from_keyed_source_unchecked((key, origin.bind(storage)))
    }
}

impl<'a, M, G> FaceView<&'a mut M, G>
where
    M: 'a + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: 'a + Geometry,
{
    /// Converts a mutable view into an orphan view.
    pub fn into_orphan(self) -> OrphanFaceView<'a, G> {
        let (key, storage) = self.into_keyed_source();
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
    /// let key = graph.faces().nth(0).unwrap().key();
    /// let face = graph
    ///     .face_mut(key)
    ///     .unwrap()
    ///     .extrude(1.0)
    ///     .unwrap()
    ///     .into_ref();
    ///
    /// // This would not be possible without conversion into an immutable view.
    /// let _ = face.into_arc();
    /// let _ = face.into_arc().into_next_arc();
    /// # }
    /// ```
    pub fn into_ref(self) -> FaceView<&'a M, G> {
        let (key, storage) = self.into_keyed_source();
        (key, &*storage).into_view().unwrap()
    }

    pub fn with_ref<T, K, F>(self, f: F) -> Either<Result<T, GraphError>, Self>
    where
        T: FromKeyedSource<(K, &'a mut M)>,
        F: FnOnce(FaceView<&M, G>) -> Option<K>,
    {
        if let Some(key) = f(self.interior_reborrow()) {
            let (_, storage) = self.into_keyed_source();
            Either::Left(
                T::from_keyed_source((key, storage)).ok_or_else(|| GraphError::TopologyNotFound),
            )
        }
        else {
            Either::Right(self)
        }
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    /// Gets the key for this face.
    pub fn key(&self) -> FaceKey {
        self.key
    }

    fn from_keyed_source_unchecked(source: (FaceKey, M)) -> Self {
        let (key, storage) = source;
        FaceView {
            key,
            storage,
            phantom: PhantomData,
        }
    }

    fn interior_reborrow(&self) -> FaceView<&M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow();
        FaceView::from_keyed_source_unchecked((key, storage))
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    fn interior_reborrow_mut(&mut self) -> FaceView<&mut M::Target, G> {
        let key = self.key;
        let storage = self.storage.reborrow_mut();
        FaceView::from_keyed_source_unchecked((key, storage))
    }
}

/// Reachable API.
impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_arc(&self) -> Option<ArcView<&M::Target, G>> {
        let key = self.arc;
        let storage = self.storage.reborrow();
        (key, storage).into_view()
    }

    pub(in crate::graph) fn into_reachable_arc(self) -> Option<ArcView<M, G>> {
        let key = self.arc;
        let (_, storage) = self.into_keyed_source();
        (key, storage).into_view()
    }

    pub(in crate::graph) fn reachable_interior_arcs(
        &self,
    ) -> impl Clone + Iterator<Item = ArcView<&M::Target, G>> {
        ArcCirculator::from(self.interior_reborrow())
    }

    pub(in crate::graph) fn reachable_neighboring_faces(
        &self,
    ) -> impl Clone + Iterator<Item = FaceView<&M::Target, G>> {
        FaceCirculator::from(ArcCirculator::from(self.interior_reborrow()))
    }

    pub(in crate::graph) fn reachable_arity(&self) -> usize {
        self.reachable_interior_arcs().count()
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + Consistent,
    G: Geometry,
{
    pub fn into_interior_path(self) -> InteriorPathView<M, G> {
        let key = self.arc().key();
        let (_, storage) = self.into_keyed_source();
        (key, storage).into_view().expect_consistent()
    }

    pub fn into_arc(self) -> ArcView<M, G> {
        self.into_reachable_arc().expect_consistent()
    }

    pub fn interior_path(&self) -> InteriorPathView<&M::Target, G> {
        let key = self.arc().key();
        let storage = self.storage.reborrow();
        (key, storage).into_view().expect_consistent()
    }

    pub fn arc(&self) -> ArcView<&M::Target, G> {
        self.reachable_arc().expect_consistent()
    }

    pub fn interior_arcs(&self) -> impl Clone + Iterator<Item = ArcView<&M::Target, G>> {
        self.reachable_interior_arcs()
    }

    pub fn neighboring_faces(&self) -> impl Clone + Iterator<Item = FaceView<&M::Target, G>> {
        self.reachable_neighboring_faces()
    }

    pub fn arity(&self) -> usize {
        self.reachable_arity()
    }
}

/// Reachable API.
impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_vertices(
        &self,
    ) -> impl Clone + Iterator<Item = VertexView<&M::Target, G>> {
        VertexCirculator::from(ArcCirculator::from(self.interior_reborrow()))
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    G: Geometry,
{
    pub fn neighborhood(&self) -> FaceNeighborhood {
        FaceNeighborhood::from(self.interior_reborrow())
    }

    pub fn vertices(&self) -> impl Clone + Iterator<Item = VertexView<&M::Target, G>> {
        self.reachable_vertices()
    }
}

/// Reachable API.
impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Arc<G>> + AsStorageMut<Arc<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_interior_orphan_arcs(
        &mut self,
    ) -> impl Iterator<Item = OrphanArcView<G>> {
        ArcCirculator::from(self.interior_reborrow_mut())
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Arc<G>> + AsStorageMut<Arc<G>> + AsStorage<Face<G>> + Consistent,
    G: Geometry,
{
    pub fn interior_orphan_arcs(&mut self) -> impl Iterator<Item = OrphanArcView<G>> {
        self.reachable_interior_orphan_arcs()
    }
}

/// Reachable API.
impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_neighboring_orphan_faces(
        &mut self,
    ) -> impl Iterator<Item = OrphanFaceView<G>> {
        FaceCirculator::from(ArcCirculator::from(self.interior_reborrow_mut()))
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>> + Consistent,
    G: Geometry,
{
    pub fn neighboring_orphan_faces(&mut self) -> impl Iterator<Item = OrphanFaceView<G>> {
        self.reachable_neighboring_orphan_faces()
    }
}

/// Reachable API.
impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target:
        AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: Geometry,
{
    pub(in crate::graph) fn reachable_orphan_vertices(
        &mut self,
    ) -> impl Iterator<Item = OrphanVertexView<G>> {
        VertexCirculator::from(ArcCirculator::from(self.interior_reborrow_mut()))
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Arc<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + AsStorageMut<Vertex<G>>
        + Consistent,
    G: Geometry,
{
    pub fn orphan_vertices(&mut self) -> impl Iterator<Item = OrphanVertexView<G>> {
        self.reachable_orphan_vertices()
    }
}

impl<'a, M, G> FaceView<&'a mut M, G>
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
    pub fn remove(self) -> Result<InteriorPathView<&'a mut M, G>, GraphError> {
        let (abc, storage) = self.into_keyed_source();
        let cache = FaceRemoveCache::snapshot(&storage, abc)?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| face::remove_with_cache(mutation, cache))
            .map(|(storage, face)| (face.arc, storage).into_view().expect_consistent())
    }
}

impl<'a, M, G> FaceView<&'a mut M, G>
where
    M: AsStorage<Arc<G>>
        + AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Default
        + From<OwnedCore<G>>
        + Into<OwnedCore<G>>,
    G: 'a + Geometry,
{
    pub fn bisect(
        self,
        source: Selector<VertexKey>,
        destination: Selector<VertexKey>,
    ) -> Result<ArcView<&'a mut M, G>, GraphError> {
        let source = source.key_or_else(|index| {
            self.vertices()
                .nth(index)
                .ok_or_else(|| GraphError::TopologyNotFound)
                .map(|vertex| vertex.key())
        })?;
        let destination = destination.key_or_else(|index| {
            self.vertices()
                .nth(index)
                .ok_or_else(|| GraphError::TopologyNotFound)
                .map(|vertex| vertex.key())
        })?;
        let (abc, storage) = self.into_keyed_source();
        let cache = FaceBisectCache::snapshot(&storage, abc, source, destination)?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| face::bisect_with_cache(mutation, cache))
            .map(|(storage, arc)| (arc, storage).into_view().expect_consistent())
    }

    pub fn bridge(self, destination: FaceKey) -> Result<(), GraphError> {
        let (source, storage) = self.into_keyed_source();
        let cache = FaceBridgeCache::snapshot(&storage, source, destination)?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| face::bridge_with_cache(mutation, cache))
            .map(|_| ())
    }

    pub fn merge(self, destination: Selector<FaceKey>) -> Result<Self, GraphError> {
        let destination = destination.key_or_else(|index| {
            self.neighboring_faces()
                .nth(index)
                .ok_or_else(|| GraphError::TopologyNotFound)
                .map(|face| face.key())
        })?;
        let ab = self
            .interior_arcs()
            .find(|arc| match arc.opposite_arc().face() {
                Some(face) => face.key() == destination,
                _ => false,
            })
            .map(|arc| arc.key())
            .ok_or_else(|| GraphError::TopologyNotFound)?;
        let geometry = self.geometry.clone();
        let (_, storage) = self.into_keyed_source();
        ArcView::from_keyed_source((ab, storage))
            .expect_consistent()
            .remove()?
            .into_outgoing_arc()
            .into_interior_path()
            .get_or_insert_face_with(|| geometry)
    }

    pub fn triangulate(self) -> Result<Self, GraphError> {
        let mut face = self;
        while face.arity() > 3 {
            face = face
                .bisect(ByIndex(0), ByIndex(2))?
                .into_face()
                .expect("bisection resulted in no face");
        }
        Ok(face)
    }

    pub fn diverge_with<F>(self, f: F) -> Result<VertexView<&'a mut M, G>, GraphError>
    where
        F: FnOnce() -> G::Vertex,
    {
        let (abc, storage) = self.into_keyed_source();
        let cache = FaceDivergeCache::snapshot(&storage, abc, f())?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| face::diverge_with_cache(mutation, cache))
            .map(|(storage, vertex)| (vertex, storage).into_view().expect_consistent())
    }

    pub fn diverge(self) -> Result<VertexView<&'a mut M, G>, GraphError>
    where
        G::Vertex: Default,
    {
        self.diverge_with(|| Default::default())
    }

    pub fn diverge_at_centroid(self) -> Result<VertexView<&'a mut M, G>, GraphError>
    where
        G: FaceCentroid<Centroid = <G as Geometry>::Vertex>,
    {
        let centroid = self.centroid()?;
        self.diverge_with(move || centroid)
    }

    pub fn extrude<T>(self, distance: T) -> Result<FaceView<&'a mut M, G>, GraphError>
    where
        G: FaceNormal,
        G::Normal: Mul<T>,
        G::Vertex: AsPosition,
        ScaledFaceNormal<G, T>: Clone,
        VertexPosition<G>: Add<ScaledFaceNormal<G, T>, Output = VertexPosition<G>> + Clone,
    {
        let (abc, storage) = self.into_keyed_source();
        let cache = FaceExtrudeCache::snapshot(&storage, abc, distance)?;
        Mutation::replace(storage, Default::default())
            .commit_with(move |mutation| face::extrude_with_cache(mutation, cache))
            .map(|(storage, face)| (face, storage).into_view().expect_consistent())
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    G: FaceCentroid + Geometry,
{
    pub fn centroid(&self) -> Result<G::Centroid, GraphError> {
        G::centroid(self.interior_reborrow())
    }
}

impl<M, G> FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    G: FaceNormal + Geometry,
{
    pub fn normal(&self) -> Result<G::Normal, GraphError> {
        G::normal(self.interior_reborrow())
    }
}

impl<M, G> Clone for FaceView<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    fn clone(&self) -> Self {
        FaceView {
            storage: self.storage.clone(),
            key: self.key,
            phantom: PhantomData,
        }
    }
}

impl<M, G> Copy for FaceView<M, G>
where
    M: Copy + Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
}

impl<M, G> Deref for FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    type Target = Face<G>;

    fn deref(&self) -> &Self::Target {
        self.storage.reborrow().as_storage().get(&self.key).unwrap()
    }
}

impl<M, G> DerefMut for FaceView<M, G>
where
    M: Reborrow + ReborrowMut,
    M::Target: AsStorage<Face<G>> + AsStorageMut<Face<G>>,
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

impl<M, G> FromKeyedSource<(FaceKey, M)> for FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (FaceKey, M)) -> Option<Self> {
        let (key, storage) = source;
        storage
            .reborrow()
            .as_storage()
            .contains_key(&key)
            .some(FaceView::from_keyed_source_unchecked((key, storage)))
    }
}

impl<M, G> IntoKeyedSource<(FaceKey, M)> for FaceView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Face<G>>,
    G: Geometry,
{
    fn into_keyed_source(self) -> (FaceKey, M) {
        let FaceView { key, storage, .. } = self;
        (key, storage)
    }
}

/// Orphan reference to a face.
pub struct OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    key: FaceKey,
    face: &'a mut Face<G>,
}

impl<'a, G> OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    pub fn key(&self) -> FaceKey {
        self.key
    }
}

impl<'a, G> Deref for OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    type Target = Face<G>;

    fn deref(&self) -> &Self::Target {
        &*self.face
    }
}

impl<'a, G> DerefMut for OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.face
    }
}

impl<'a, M, G> FromKeyedSource<(FaceKey, &'a mut M)> for OrphanFaceView<'a, G>
where
    M: AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (FaceKey, &'a mut M)) -> Option<Self> {
        let (key, storage) = source;
        storage
            .as_storage_mut()
            .get_mut(&key)
            .map(|face| OrphanFaceView { key, face })
    }
}

impl<'a, G> FromKeyedSource<(FaceKey, &'a mut Face<G>)> for OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (FaceKey, &'a mut Face<G>)) -> Option<Self> {
        let (key, face) = source;
        Some(OrphanFaceView { key, face })
    }
}

#[derive(Clone, Debug)]
pub struct FaceNeighborhood {
    key: FaceKey,
    arcs: Vec<ArcNeighborhood>,
}

impl FaceNeighborhood {
    pub fn key(&self) -> FaceKey {
        self.key
    }

    pub fn interior_arcs(&self) -> &[ArcNeighborhood] {
        self.arcs.as_slice()
    }
}

impl<M, G> From<FaceView<M, G>> for FaceNeighborhood
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + Consistent,
    G: Geometry,
{
    fn from(face: FaceView<M, G>) -> Self {
        FaceNeighborhood {
            key: face.key,
            arcs: face
                .reachable_interior_arcs()
                .map(|arc| arc.neighborhood())
                .collect(),
        }
    }
}

pub struct InteriorPathView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + Consistent,
    G: Geometry,
{
    storage: M,
    arc: ArcKey,
    face: Option<FaceKey>,
    phantom: PhantomData<G>,
}

impl<'a, M, G> InteriorPathView<&'a mut M, G>
where
    M: AsStorage<Arc<G>> + Consistent,
    G: 'a + Geometry,
{
    pub fn into_ref(self) -> InteriorPathView<&'a M, G> {
        let (arc, face, storage) = self.into_keyed_source();
        InteriorPathView::from_keyed_source_unchecked((arc, face, &*storage))
    }

    pub fn with_ref<T, K, F>(self, f: F) -> Either<Result<T, GraphError>, Self>
    where
        T: FromKeyedSource<(K, &'a mut M)>,
        F: FnOnce(InteriorPathView<&M, G>) -> Option<K>,
    {
        if let Some(key) = f(self.interior_reborrow()) {
            let (_, _, storage) = self.into_keyed_source();
            Either::Left(
                T::from_keyed_source((key, storage)).ok_or_else(|| GraphError::TopologyNotFound),
            )
        }
        else {
            Either::Right(self)
        }
    }
}

impl<M, G> InteriorPathView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + Consistent,
    G: Geometry,
{
    fn from_keyed_source_unchecked(source: (ArcKey, Option<FaceKey>, M)) -> Self {
        let (arc, face, storage) = source;
        InteriorPathView {
            storage,
            arc,
            face,
            phantom: PhantomData,
        }
    }

    pub fn arity(&self) -> usize {
        self.arcs().count()
    }

    pub fn arcs(&self) -> impl Clone + Iterator<Item = ArcView<&M::Target, G>> {
        ArcCirculator::from(self.interior_reborrow())
    }

    fn interior_reborrow(&self) -> InteriorPathView<&M::Target, G> {
        let arc = self.arc;
        let face = self.face;
        let storage = self.storage.reborrow();
        InteriorPathView::from_keyed_source_unchecked((arc, face, storage))
    }
}

impl<M, G> InteriorPathView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent,
    G: Geometry,
{
    pub fn into_arc(self) -> ArcView<M, G> {
        let (arc, _, storage) = self.into_keyed_source();
        (arc, storage).into_view().expect_consistent()
    }

    pub fn arc(&self) -> ArcView<&M::Target, G> {
        let arc = self.arc;
        let storage = self.storage.reborrow();
        (arc, storage).into_view().expect_consistent()
    }

    pub fn distance(
        &self,
        source: Selector<VertexKey>,
        destination: Selector<VertexKey>,
    ) -> Result<usize, GraphError> {
        let arity = self.arity();
        let select = |selector: Selector<_>| {
            selector
                .index_or_else(|key| {
                    self.vertices()
                        .map(|vertex| vertex.key())
                        .enumerate()
                        .find(|(_, a)| *a == key)
                        .map(|(index, _)| index)
                        .ok_or_else(|| GraphError::TopologyNotFound)
                })
                .and_then(|index| {
                    if index >= arity {
                        Err(GraphError::TopologyNotFound)
                    }
                    else {
                        Ok(index)
                    }
                })
        };
        let source = select(source)? as isize;
        let destination = select(destination)? as isize;
        let difference = (source - destination).abs() as usize;
        Ok(cmp::min(difference, arity - difference))
    }

    pub fn vertices(&self) -> impl Clone + Iterator<Item = VertexView<&M::Target, G>> {
        VertexCirculator::from(ArcCirculator::from(self.interior_reborrow()))
    }
}

impl<M, G> InteriorPathView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>> + Consistent,
    G: Geometry,
{
    pub fn into_face(self) -> Option<FaceView<M, G>> {
        let (_, face, storage) = self.into_keyed_source();
        if let Some(face) = face {
            Some((face, storage).into_view().expect_consistent())
        }
        else {
            None
        }
    }

    pub fn face(&self) -> Option<FaceView<&M::Target, G>> {
        if let Some(face) = self.face {
            let storage = self.storage.reborrow();
            Some((face, storage).into_view().expect_consistent())
        }
        else {
            None
        }
    }
}

impl<'a, M, G> InteriorPathView<&'a mut M, G>
where
    M: AsStorage<Vertex<G>>
        + AsStorage<Arc<G>>
        + AsStorage<Face<G>>
        + Consistent
        + Default
        + From<OwnedCore<G>>
        + Into<OwnedCore<G>>,
    G: 'a + Geometry,
{
    pub fn get_or_insert_face(self) -> Result<FaceView<&'a mut M, G>, GraphError> {
        self.get_or_insert_face_with(|| Default::default())
    }

    pub fn get_or_insert_face_with<F>(self, f: F) -> Result<FaceView<&'a mut M, G>, GraphError>
    where
        F: FnOnce() -> G::Face,
    {
        if let Some(face) = self.face.clone().take() {
            let (_, _, storage) = self.into_keyed_source();
            Ok((face, storage).into_view().expect_consistent())
        }
        else {
            let vertices = self
                .vertices()
                .map(|vertex| vertex.key())
                .collect::<Vec<_>>();
            let (_, _, storage) = self.into_keyed_source();
            let cache = FaceInsertCache::snapshot(&storage, &vertices, (Default::default(), f()))?;
            Mutation::replace(storage, Default::default())
                .commit_with(move |mutation| mutation.insert_face_with_cache(cache))
                .map(|(storage, face)| (face, storage).into_view().expect_consistent())
        }
    }
}

impl<M, G> FromKeyedSource<(ArcKey, M)> for InteriorPathView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + Consistent,
    G: Geometry,
{
    fn from_keyed_source(source: (ArcKey, M)) -> Option<Self> {
        // Because the storage is consistent, this code assumes that any and
        // all arcs in the graph will form a loop. Note that this allows
        // exterior arcs of non-enclosed meshes to form a region. For
        // conceptually flat meshes, this is odd, but is topologically
        // consistent.
        let (key, storage) = source;
        if let Some(arc) = storage.reborrow().as_storage().get(&key) {
            let face = arc.face.clone();
            Some(InteriorPathView {
                storage,
                arc: key,
                face,
                phantom: PhantomData,
            })
        }
        else {
            None
        }
    }
}

impl<M, G> IntoKeyedSource<(ArcKey, Option<FaceKey>, M)> for InteriorPathView<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + Consistent,
    G: Geometry,
{
    fn into_keyed_source(self) -> (ArcKey, Option<FaceKey>, M) {
        let InteriorPathView {
            storage, arc, face, ..
        } = self;
        (arc, face, storage)
    }
}

struct VertexCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>>,
    G: Geometry,
{
    input: ArcCirculator<M, G>,
}

impl<M, G> VertexCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>>,
    G: Geometry,
{
    fn next(&mut self) -> Option<VertexKey> {
        let ab = self.input.next();
        ab.map(|ab| {
            let (_, b) = ab.into();
            b
        })
    }
}

impl<M, G> Clone for VertexCirculator<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<Arc<G>>,
    G: Geometry,
{
    fn clone(&self) -> Self {
        VertexCirculator {
            input: self.input.clone(),
        }
    }
}

impl<M, G> From<ArcCirculator<M, G>> for VertexCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>>,
    G: Geometry,
{
    fn from(input: ArcCirculator<M, G>) -> Self {
        VertexCirculator { input }
    }
}

// TODO: This iterator could provide a size hint of `(3, None)`, but this is
//       only the case when the underlying mesh is consistent.
impl<'a, M, G> Iterator for VertexCirculator<&'a M, G>
where
    M: 'a + AsStorage<Arc<G>> + AsStorage<Vertex<G>>,
    G: 'a + Geometry,
{
    type Item = VertexView<&'a M, G>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).and_then(|key| (key, self.input.storage).into_view())
    }
}

// TODO: This iterator could provide a size hint of `(3, None)`, but this is
//       only the case when the underlying mesh is consistent.
impl<'a, M, G> Iterator for VertexCirculator<&'a mut M, G>
where
    M: 'a + AsStorage<Arc<G>> + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: 'a + Geometry,
{
    type Item = OrphanVertexView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).and_then(|key| {
            (key, unsafe {
                mem::transmute::<&'_ mut Storage<Vertex<G>>, &'a mut Storage<Vertex<G>>>(
                    self.input.storage.as_storage_mut(),
                )
            })
                .into_view()
        })
    }
}

struct ArcCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>>,
    G: Geometry,
{
    storage: M,
    arc: Option<ArcKey>,
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
        self.arc.and_then(|arc| {
            let next = self
                .storage
                .reborrow()
                .as_storage()
                .get(&arc)
                .and_then(|arc| arc.next);
            self.breadcrumb.map(|_| {
                if self.breadcrumb == next {
                    self.breadcrumb = None;
                }
                else {
                    self.arc = next;
                }
                arc
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
            arc: self.arc,
            breadcrumb: self.breadcrumb,
            phantom: PhantomData,
        }
    }
}

impl<M, G> From<FaceView<M, G>> for ArcCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    fn from(face: FaceView<M, G>) -> Self {
        let arc = face.arc;
        let (_, storage) = face.into_keyed_source();
        ArcCirculator {
            storage,
            arc: Some(arc),
            breadcrumb: Some(arc),
            phantom: PhantomData,
        }
    }
}

impl<M, G> From<InteriorPathView<M, G>> for ArcCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>> + Consistent,
    G: Geometry,
{
    fn from(path: InteriorPathView<M, G>) -> Self {
        let (arc, _, storage) = path.into_keyed_source();
        ArcCirculator {
            storage,
            arc: Some(arc),
            breadcrumb: Some(arc),
            phantom: PhantomData,
        }
    }
}

// TODO: This iterator could provide a size hint of `(3, None)`, but this is
//       only the case when the underlying mesh is consistent.
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

// TODO: This iterator could provide a size hint of `(3, None)`, but this is
//       only the case when the underlying mesh is consistent.
impl<'a, M, G> Iterator for ArcCirculator<&'a mut M, G>
where
    M: 'a + AsStorage<Arc<G>> + AsStorageMut<Arc<G>>,
    G: 'a + Geometry,
{
    type Item = OrphanArcView<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        ArcCirculator::next(self).and_then(|key| {
            (key, unsafe {
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
    M::Target: AsStorage<Arc<G>>,
    G: Geometry,
{
    input: ArcCirculator<M, G>,
}

impl<M, G> FaceCirculator<M, G>
where
    M: Reborrow,
    M::Target: AsStorage<Arc<G>>,
    G: Geometry,
{
    fn next(&mut self) -> Option<FaceKey> {
        while let Some(ba) = self.input.next().map(|ab| ab.opposite()) {
            if let Some(abc) = self
                .input
                .storage
                .reborrow()
                .as_storage()
                .get(&ba)
                .and_then(|opposite| opposite.face)
            {
                return Some(abc);
            }
            else {
                // Skip arcs with no opposing face. This can occur within
                // non-enclosed meshes.
                continue;
            }
        }
        None
    }
}

impl<M, G> Clone for FaceCirculator<M, G>
where
    M: Clone + Reborrow,
    M::Target: AsStorage<Arc<G>>,
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
    M::Target: AsStorage<Arc<G>>,
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
    use nalgebra::{Point2, Point3};

    use crate::graph::*;
    use crate::primitive::cube::Cube;
    use crate::primitive::generate::*;
    use crate::primitive::index::*;
    use crate::primitive::sphere::UvSphere;
    use crate::*;

    #[test]
    fn circulate_over_arcs() {
        let graph = UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f32>>>();
        let face = graph.faces().nth(0).unwrap();

        // All faces should be triangles and should have three edges.
        assert_eq!(3, face.interior_arcs().count());
    }

    #[test]
    fn circulate_over_faces() {
        let graph = UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f32>>>();
        let face = graph.faces().nth(0).unwrap();

        // No matter which face is selected, it should have three neighbors.
        assert_eq!(3, face.neighboring_faces().count());
    }

    #[test]
    fn remove_face() {
        let mut graph = UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f32>>>();

        // The graph should begin with 6 faces.
        assert_eq!(6, graph.face_count());

        // Remove a face from the graph.
        let abc = graph.faces().nth(0).unwrap().key();
        {
            let face = graph.face_mut(abc).unwrap();
            assert_eq!(3, face.arity()); // The face should be triangular.

            let path = face.remove().unwrap().into_ref();
            assert_eq!(3, path.arity()); // The path should also be triangular.
        }

        // After the removal, the graph should have only 5 faces.
        assert_eq!(5, graph.face_count());
    }

    #[test]
    fn bisect_face() {
        let mut graph = MeshGraph::<Point2<f32>>::from_raw_buffers_with_arity(
            vec![0u32, 1, 2, 3],
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            4,
        )
        .unwrap();
        let abc = graph.faces().nth(0).unwrap().key();
        let (p, q) = {
            let face = graph.face(abc).unwrap();
            let mut vertices = face.vertices().map(|vertex| vertex.key()).step_by(2);
            (vertices.next().unwrap(), vertices.next().unwrap())
        };
        let arc = graph
            .face_mut(abc)
            .unwrap()
            .bisect(ByKey(p), ByKey(q))
            .unwrap()
            .into_ref();

        assert!(arc.face().is_some());
        assert!(arc.opposite_arc().face().is_some());
        assert_eq!(4, graph.vertex_count());
        assert_eq!(10, graph.arc_count());
        assert_eq!(2, graph.face_count());
    }

    #[test]
    fn extrude_face() {
        let mut graph = UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f32>>>();
        {
            let key = graph.faces().nth(0).unwrap().key();
            let face = graph
                .face_mut(key)
                .unwrap()
                .extrude(1.0)
                .unwrap()
                .into_ref();

            // The extruded face, being a triangle, should have three
            // neighboring faces.
            assert_eq!(3, face.neighboring_faces().count());
        }

        assert_eq!(8, graph.vertex_count());
        // The mesh begins with 18 arcs. The extrusion adds three quads
        // with four interior arcs each, so there are `18 + (3 * 4)`
        // arcs.
        assert_eq!(30, graph.arc_count());
        // All faces are triangles and the mesh begins with six such faces. The
        // extruded face remains, in addition to three connective faces, each
        // of which is constructed from quads.
        assert_eq!(9, graph.face_count());
    }

    #[test]
    fn join_faces() {
        // Construct a graph with two connected quads.
        let mut graph = MeshGraph::<Point2<f32>>::from_raw_buffers_with_arity(
            vec![0u32, 1, 2, 3, 0, 3, 4, 5],
            vec![
                (0.0, 0.0),  // 0
                (1.0, 0.0),  // 1
                (1.0, 1.0),  // 2
                (0.0, 1.0),  // 3
                (-1.0, 1.0), // 4
                (-1.0, 0.0), // 5
            ],
            4,
        )
        .unwrap();

        // The graph should begin with 2 faces.
        assert_eq!(2, graph.face_count());

        // Get the keys for the two faces and join them.
        let abc = graph.faces().nth(0).unwrap().key();
        let def = graph.faces().nth(1).unwrap().key();
        graph.face_mut(abc).unwrap().merge(ByKey(def)).unwrap();

        // After the removal, the graph should have 1 face.
        assert_eq!(1, graph.face_count());
        assert_eq!(6, graph.faces().nth(0).unwrap().arity());
    }

    #[test]
    fn diverge_face() {
        let mut graph = Cube::new()
            .polygons_with_position() // 6 quads, 24 vertices.
            .collect::<MeshGraph<Point3<f32>>>();
        let key = graph.faces().nth(0).unwrap().key();
        let vertex = graph.face_mut(key).unwrap().diverge_at_centroid().unwrap();

        // Diverging a quad yields a tetrahedron.
        assert_eq!(4, vertex.neighboring_faces().count());

        // Traverse to one of the triangles in the tetrahedron.
        let face = vertex.into_outgoing_arc().into_face().unwrap();

        assert_eq!(3, face.arity());

        // Diverge the triangle.
        let vertex = face.diverge_at_centroid().unwrap();

        assert_eq!(3, vertex.neighboring_faces().count());
    }

    #[test]
    fn triangulate_mesh() {
        let (indices, vertices) = Cube::new()
            .polygons_with_position() // 6 quads, 24 vertices.
            .index_vertices(HashIndexer::default());
        let mut graph = MeshGraph::<Point3<f32>>::from_raw_buffers(indices, vertices).unwrap();
        graph.triangulate().unwrap();

        assert_eq!(8, graph.vertex_count());
        assert_eq!(36, graph.arc_count());
        assert_eq!(18, graph.edge_count());
        // Each quad becomes 2 triangles, so 6 quads become 12 triangles.
        assert_eq!(12, graph.face_count());
    }

    #[test]
    fn interior_path_distance() {
        let graph = MeshGraph::<Point2<f32>>::from_raw_buffers_with_arity(
            vec![0u32, 1, 2, 3],
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            4,
        )
        .unwrap();
        let face = graph.faces().nth(0).unwrap();
        let keys = face
            .vertices()
            .map(|vertex| vertex.key())
            .collect::<Vec<_>>();
        let path = face.into_interior_path();
        assert_eq!(2, path.distance(keys[0].into(), keys[2].into()).unwrap());
        assert_eq!(1, path.distance(keys[0].into(), keys[3].into()).unwrap());
        assert_eq!(0, path.distance(keys[0].into(), keys[0].into()).unwrap());
    }
}
