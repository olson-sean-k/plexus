use failure::Error;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Add, Deref, DerefMut, Mul};

use geometry::convert::AsPosition;
use geometry::Geometry;
use graph::geometry::alias::{ScaledFaceNormal, VertexPosition};
use graph::geometry::{FaceCentroid, FaceNormal};
use graph::mesh::{Edge, Face, Mesh, Vertex};
use graph::mutation::{ModalMutation, Mutation};
use graph::storage::convert::{AsStorage, AsStorageMut};
use graph::storage::{Bind, EdgeKey, FaceKey, Storage, Topological, VertexKey};
use graph::view::convert::{FromKeyedSource, IntoView};
use graph::view::{
    edge, Consistency, Consistent, EdgeKeyTopology, EdgeView, Inconsistent, IteratorExt,
    OrphanEdgeView, OrphanVertexView, VertexView,
};
use graph::{GraphError, Perimeter};

/// Do **not** use this type directly. Use `FaceRef` and `FaceMut` instead.
///
/// This type is only re-exported so that its members are shown in
/// documentation. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
pub struct FaceView<M, G, C>
where
    M: AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    key: FaceKey,
    storage: M,
    phantom: PhantomData<(G, C)>,
}

/// Storage.
impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn bind<T, N>(self, storage: N) -> FaceView<<M as Bind<T, N>>::Output, G, C>
    where
        T: Topological,
        M: Bind<T, N>,
        M::Output: AsStorage<Face<G>>,
        N: AsStorage<T>,
    {
        let FaceView {
            key,
            storage: origin,
            ..
        } = self;
        FaceView {
            key,
            storage: origin.bind(storage),
            phantom: PhantomData,
        }
    }

    pub(in graph) fn as_storage<T>(&self) -> &Storage<T>
    where
        T: Topological,
        M: AsStorage<T>,
    {
        AsStorage::<T>::as_storage(&self.storage)
    }

    pub(in graph) fn as_storage_mut<T>(&mut self) -> &mut Storage<T>
    where
        T: Topological,
        M: AsStorageMut<T>,
    {
        AsStorageMut::<T>::as_storage_mut(&mut self.storage)
    }
}

impl<'a, 'b, M, G, C> FaceView<&'a &'b M, G, C>
where
    M: AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn into_interior_deref(self) -> FaceView<&'b M, G, C>
    where
        (FaceKey, &'b M): IntoView<FaceView<&'b M, G, C>>,
    {
        let key = self.key;
        let storage = *self.storage;
        (key, storage).into_view()
    }

    pub fn interior_deref(&self) -> FaceView<&'b M, G, C>
    where
        (FaceKey, &'b M): IntoView<FaceView<&'b M, G, C>>,
    {
        let key = self.key;
        let storage = *self.storage;
        (key, storage).into_view()
    }
}

impl<'a, M, G, C> FaceView<&'a mut M, G, C>
where
    M: 'a + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: 'a + Geometry,
    C: Consistency,
{
    pub fn into_orphan(self) -> OrphanFaceView<'a, G> {
        let FaceView { key, storage, .. } = self;
        (key, storage.as_storage_mut().get_mut(&key).unwrap()).into_view()
    }
}

impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn to_orphan(&mut self) -> OrphanFaceView<G> {
        let key = self.key;
        (key, self.storage.as_storage_mut().get_mut(&key).unwrap()).into_view()
    }
}

impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn key(&self) -> FaceKey {
        self.key
    }

    pub(in graph) fn from_keyed_storage(key: FaceKey, storage: M) -> Self {
        FaceView {
            key,
            storage,
            phantom: PhantomData,
        }
    }

    fn to_ref(&self) -> FaceView<&M, G, C> {
        let key = self.key;
        let storage = &self.storage;
        FaceView::from_keyed_storage(key, storage)
    }

    fn to_mut(&mut self) -> FaceView<&mut M, G, C> {
        let key = self.key;
        let storage = &mut self.storage;
        FaceView::from_keyed_storage(key, storage)
    }
}

impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    // TODO: Does this require consistency? Is it needed in mutations?
    pub fn to_key_topology<'a>(&'a self) -> FaceKeyTopology
    where
        (EdgeKey, &'a M): IntoView<EdgeView<&'a M, G, C>>,
    {
        FaceKeyTopology::new(
            self.key,
            self.reachable_edges().map(|edge| edge.to_key_topology()),
        )
    }

    // TODO: Does this require consistency? Is it needed in mutations?
    pub fn arity(&self) -> usize {
        self.reachable_edges().count()
    }

    pub(in graph) fn reachable_edges(&self) -> impl Iterator<Item = EdgeView<&M, G, C>> {
        EdgeCirculator::from(self.to_ref())
            .map_with_ref(|circulator, key| EdgeView::from_keyed_storage(key, circulator.storage))
    }

    pub(in graph) fn reachable_faces(&self) -> impl Iterator<Item = FaceView<&M, G, C>> {
        FaceCirculator::from(EdgeCirculator::from(self.to_ref())).map_with_ref(|circulator, key| {
            FaceView::from_keyed_storage(key, circulator.input.storage)
        })
    }
}

impl<M, G> FaceView<M, G, Consistent>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    pub fn edges(&self) -> impl Iterator<Item = EdgeView<&M, G, Consistent>> {
        self.reachable_edges()
    }

    pub fn faces(&self) -> impl Iterator<Item = FaceView<&M, G, Consistent>> {
        self.reachable_faces()
    }
}

impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_vertices(&self) -> impl Iterator<Item = VertexView<&M, G, C>> {
        VertexCirculator::from(EdgeCirculator::from(self.to_ref())).map_with_ref(
            |circulator, key| VertexView::from_keyed_storage(key, circulator.input.storage),
        )
    }
}

impl<M, G> FaceView<M, G, Consistent>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: Geometry,
{
    pub fn vertices(&self) -> impl Iterator<Item = VertexView<&M, G, Consistent>> {
        self.reachable_vertices()
    }
}

impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorageMut<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_orphan_edges<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = OrphanEdgeView<'a, G>> {
        EdgeCirculator::from(self.to_mut()).map_with_mut(|circulator, key| {
            (key, unsafe {
                // Apply `'a` to the autoref from `as_storage_mut` and
                // `get_mut`.
                mem::transmute::<&'_ mut Edge<G>, &'a mut Edge<G>>(
                    circulator.storage.as_storage_mut().get_mut(&key).unwrap(),
                )
            }).into_view()
        })
    }
}

impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_orphan_faces<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = OrphanFaceView<'a, G>> {
        FaceCirculator::from(EdgeCirculator::from(self.to_mut())).map_with_mut(|circulator, key| {
            (key, unsafe {
                // Apply `'a` to the autoref from `as_storage_mut` and
                // `get_mut`.
                mem::transmute::<&'_ mut Face<G>, &'a mut Face<G>>(
                    circulator
                        .input
                        .storage
                        .as_storage_mut()
                        .get_mut(&key)
                        .unwrap(),
                )
            }).into_view()
        })
    }
}

impl<M, G> FaceView<M, G, Consistent>
where
    M: AsStorage<Edge<G>> + AsStorageMut<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    pub fn orphan_edges(&mut self) -> impl Iterator<Item = OrphanEdgeView<G>> {
        self.reachable_orphan_edges()
    }
}

impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    pub fn orphan_faces(&mut self) -> impl Iterator<Item = OrphanFaceView<G>> {
        self.reachable_orphan_faces()
    }
}

impl<M, G, C> FaceView<M, G, C>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: Geometry,
    C: Consistency,
{
    pub(in graph) fn reachable_orphan_vertices<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = OrphanVertexView<'a, G>> {
        VertexCirculator::from(EdgeCirculator::from(self.to_mut())).map_with_mut(
            |circulator, key| {
                (key, unsafe {
                    // Apply `'a` to the autoref from `as_storage_mut` and
                    // `get_mut`.
                    mem::transmute::<&'_ mut Vertex<G>, &'a mut Vertex<G>>(
                        circulator
                            .input
                            .storage
                            .as_storage_mut()
                            .get_mut(&key)
                            .unwrap(),
                    )
                }).into_view()
            },
        )
    }
}

impl<M, G> FaceView<M, G, Consistent>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>> + AsStorageMut<Vertex<G>>,
    G: Geometry,
{
    pub fn orphan_vertices(&mut self) -> impl Iterator<Item = OrphanVertexView<G>> {
        self.reachable_orphan_vertices()
    }
}

impl<M, G> FaceView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsMut<Mesh<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    pub fn join(self, destination: FaceKey) -> Result<(), Error> {
        let FaceView {
            mut storage,
            key: source,
            ..
        } = self;
        let mut mutation = Mutation::replace(storage.as_mut(), Mesh::empty());
        join(&mut *mutation, source, destination)?;
        mutation.commit()?;
        Ok(())
    }
}

impl<M, G> FaceView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: FaceCentroid + Geometry,
{
    pub fn centroid(&self) -> Result<G::Centroid, Error> {
        G::centroid(self)
    }
}

impl<'a, G> FaceView<&'a mut Mesh<G>, G, Consistent>
where
    G: FaceCentroid<Centroid = <G as Geometry>::Vertex> + Geometry,
{
    pub fn triangulate(self) -> Result<Option<VertexView<&'a mut Mesh<G>, G>>, Error> {
        let FaceView {
            storage, key: abc, ..
        } = self;
        let mut mutation = Mutation::immediate(storage);
        Ok(match triangulate(&mut mutation, abc)? {
            Some(vertex) => Some((vertex, mutation.commit()).into_view()),
            _ => None,
        })
    }
}

impl<M, G> FaceView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Edge<G>> + AsStorage<Face<G>> + AsStorage<Vertex<G>>,
    G: FaceNormal + Geometry,
{
    pub fn normal(&self) -> Result<G::Normal, Error> {
        G::normal(self)
    }
}

impl<'a, G> FaceView<&'a mut Mesh<G>, G, Consistent>
where
    G: FaceNormal + Geometry,
    G::Vertex: AsPosition,
{
    pub fn extrude<T>(self, distance: T) -> Result<FaceView<&'a mut Mesh<G>, G, Consistent>, Error>
    where
        G::Normal: Mul<T>,
        ScaledFaceNormal<G, T>: Clone,
        VertexPosition<G>: Add<ScaledFaceNormal<G, T>, Output = VertexPosition<G>> + Clone,
    {
        let FaceView {
            storage, key: abc, ..
        } = self;
        let mut mutation = Mutation::replace(storage, Mesh::empty());
        let face = extrude(&mut *mutation, abc, distance)?;
        Ok((face, mutation.commit().unwrap()).into_view())
    }
}

impl<M, G, C> Clone for FaceView<M, G, C>
where
    M: AsStorage<Face<G>> + Clone,
    G: Geometry,
    C: Consistency,
{
    fn clone(&self) -> Self {
        FaceView {
            storage: self.storage.clone(),
            key: self.key,
            phantom: PhantomData,
        }
    }
}

impl<M, G, C> Copy for FaceView<M, G, C>
where
    M: AsStorage<Face<G>> + Copy,
    G: Geometry,
    C: Consistency,
{
}

impl<M, G, C> Deref for FaceView<M, G, C>
where
    M: AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    type Target = Face<G>;

    fn deref(&self) -> &Self::Target {
        self.storage.as_storage().get(&self.key).unwrap()
    }
}

impl<M, G, C> DerefMut for FaceView<M, G, C>
where
    M: AsStorage<Face<G>> + AsStorageMut<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.storage.as_storage_mut().get_mut(&self.key).unwrap()
    }
}

impl<M, G> FromKeyedSource<(FaceKey, M)> for FaceView<M, G, Inconsistent>
where
    M: AsStorage<Face<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (FaceKey, M)) -> Self {
        let (key, storage) = source;
        FaceView {
            key,
            storage,
            phantom: PhantomData,
        }
    }
}

impl<M, G> FromKeyedSource<(FaceKey, M)> for FaceView<M, G, Consistent>
where
    M: AsRef<Mesh<G>> + AsStorage<Face<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (FaceKey, M)) -> Self {
        let (key, storage) = source;
        FaceView {
            key,
            storage,
            phantom: PhantomData,
        }
    }
}

/// Do **not** use this type directly. Use `OrphanFace` instead.
///
/// This type is only re-exported so that its members are shown in
/// documentation. See this issue:
/// <https://github.com/rust-lang/rust/issues/39437>
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

impl<'a, G> FromKeyedSource<(FaceKey, &'a mut Face<G>)> for OrphanFaceView<'a, G>
where
    G: 'a + Geometry,
{
    fn from_keyed_source(source: (FaceKey, &'a mut Face<G>)) -> Self {
        let (key, face) = source;
        OrphanFaceView { key, face }
    }
}

#[derive(Clone, Debug)]
pub struct FaceKeyTopology {
    key: FaceKey,
    edges: Vec<EdgeKeyTopology>,
}

impl FaceKeyTopology {
    fn new<I>(face: FaceKey, edges: I) -> Self
    where
        I: IntoIterator<Item = EdgeKeyTopology>,
    {
        FaceKeyTopology {
            key: face,
            edges: edges.into_iter().collect(),
        }
    }

    pub fn key(&self) -> FaceKey {
        self.key
    }

    pub fn edges(&self) -> &[EdgeKeyTopology] {
        self.edges.as_slice()
    }
}

pub struct VertexCirculator<M, G>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
{
    input: EdgeCirculator<M, G>,
}

impl<M, G> From<EdgeCirculator<M, G>> for VertexCirculator<M, G>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
{
    fn from(input: EdgeCirculator<M, G>) -> Self {
        VertexCirculator { input }
    }
}

impl<M, G> Iterator for VertexCirculator<M, G>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
{
    type Item = VertexKey;

    fn next(&mut self) -> Option<Self::Item> {
        let edge = self.input.next();
        edge.and_then(|edge| self.input.storage.as_storage().get(&edge))
            .map(|edge| edge.vertex)
    }
}

pub struct EdgeCirculator<M, G>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
{
    storage: M,
    edge: Option<EdgeKey>,
    breadcrumb: Option<EdgeKey>,
    phantom: PhantomData<G>,
}

impl<M, G, C> From<FaceView<M, G, C>> for EdgeCirculator<M, G>
where
    M: AsStorage<Edge<G>> + AsStorage<Face<G>>,
    G: Geometry,
    C: Consistency,
{
    fn from(face: FaceView<M, G, C>) -> Self {
        let edge = face.edge;
        let FaceView { storage, .. } = face;
        EdgeCirculator {
            storage,
            edge: Some(edge),
            breadcrumb: Some(edge),
            phantom: PhantomData,
        }
    }
}

impl<M, G> FromKeyedSource<(EdgeKey, M)> for EdgeCirculator<M, G>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
{
    fn from_keyed_source(source: (EdgeKey, M)) -> Self {
        let (key, storage) = source;
        EdgeCirculator {
            storage,
            edge: Some(key),
            breadcrumb: Some(key),
            phantom: PhantomData,
        }
    }
}

impl<M, G> Iterator for EdgeCirculator<M, G>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
{
    type Item = EdgeKey;

    fn next(&mut self) -> Option<Self::Item> {
        self.edge.and_then(|edge| {
            let next = self
                .storage
                .as_storage()
                .get(&edge)
                .and_then(|edge| edge.next);
            self.breadcrumb.map(|_| {
                if self.breadcrumb == next {
                    self.breadcrumb = None;
                }
                else {
                    self.edge = next;
                }
                edge
            })
        })
    }
}

pub struct FaceCirculator<M, G>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
{
    input: EdgeCirculator<M, G>,
}

impl<M, G> From<EdgeCirculator<M, G>> for FaceCirculator<M, G>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
{
    fn from(input: EdgeCirculator<M, G>) -> Self {
        FaceCirculator { input }
    }
}

impl<M, G> Iterator for FaceCirculator<M, G>
where
    M: AsStorage<Edge<G>>,
    G: Geometry,
{
    type Item = FaceKey;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(edge) = self
            .input
            .next()
            .and_then(|edge| self.input.storage.as_storage().get(&edge))
        {
            if let Some(face) = edge
                .opposite
                .and_then(|opposite| self.input.storage.as_storage().get(&opposite))
                .and_then(|opposite| opposite.face)
            {
                return Some(face);
            }
            else {
                // Skip edges with no opposing face. This can occur within
                // non-enclosed meshes.
                continue;
            }
        }
        None
    }
}

pub(in graph) fn triangulate<'a, M, G>(
    mutation: &mut M,
    abc: FaceKey,
) -> Result<Option<VertexKey>, Error>
where
    M: ModalMutation<'a, G>,
    G: 'a + FaceCentroid<Centroid = <G as Geometry>::Vertex> + Geometry,
{
    let (perimeter, centroid, face) = {
        let face = match mutation.as_mesh().face(abc) {
            Some(face) => face,
            _ => return Err(GraphError::TopologyNotFound.into()),
        };
        let perimeter = face
            .edges()
            .map(|edge| (edge.vertex, edge.next_edge().vertex))
            .collect::<Vec<_>>();
        if perimeter.len() <= 3 {
            return Ok(None);
        }
        (perimeter, face.centroid()?, face.geometry.clone())
    };
    // This is the point of no return; the mesh has mutated. Unwrap
    // results.
    mutation.remove_face(abc).unwrap();
    let c = mutation.insert_vertex(centroid);
    for (a, b) in perimeter {
        mutation
            .insert_face(&[a, b, c], (Default::default(), face.clone()))
            .unwrap();
    }
    Ok(Some(c))
}

pub(in graph) fn join<'a, M, G>(
    mutation: &mut M,
    source: FaceKey,
    destination: FaceKey,
) -> Result<(), Error>
where
    M: ModalMutation<'a, G>,
    G: 'a + Geometry,
{
    let (sources, destination) = {
        let source = match mutation.as_mesh().face(source) {
            Some(face) => face,
            _ => return Err(GraphError::TopologyNotFound.into()),
        };
        // Ensure that the opposite face exists and has the same arity.
        let arity = source.arity();
        let destination = match mutation.as_mesh().face(destination) {
            Some(destination) => {
                if destination.arity() != arity {
                    return Err(GraphError::ArityNonConstant.into());
                }
                destination
            }
            _ => {
                return Err(GraphError::TopologyNotFound.into());
            }
        };
        // Decompose the faces into their key topologies.
        (
            source
                .to_key_topology()
                .edges()
                .iter()
                .map(|topology| {
                    (
                        topology.clone(),
                        mutation
                            .as_mesh()
                            .edge(topology.key())
                            .unwrap()
                            .geometry
                            .clone(),
                    )
                })
                .collect::<Vec<_>>(),
            destination.to_key_topology(),
        )
    };
    // Remove the source and destination faces. Pair the topology with edge
    // geometry for the source face. At this point, we can assume that the
    // faces exist and no failures should occur; unwrap results.
    mutation.remove_face(source)?;
    mutation.remove_face(destination.key())?;
    // TODO: Is it always correct to reverse the order of the opposite
    //       face's edges?
    // Re-insert the edges of the faces and join the mutual edges.
    for (source, destination) in sources.into_iter().zip(destination.edges().iter().rev()) {
        let (a, b) = source.0.vertices();
        let (c, d) = destination.vertices();
        let ab = mutation.insert_edge((a, b), source.1.clone()).unwrap();
        let cd = mutation.insert_edge((c, d), source.1).unwrap();
        edge::join(mutation, ab, cd).unwrap();
    }
    // TODO: Is there any reasonable topology this can return?
    Ok(())
}

pub(in graph) fn extrude<'a, M, G, T>(
    mutation: &mut M,
    abc: FaceKey,
    distance: T,
) -> Result<FaceKey, Error>
where
    M: ModalMutation<'a, G>,
    G: 'a + FaceNormal + Geometry,
    G::Normal: Mul<T>,
    G::Vertex: AsPosition,
    ScaledFaceNormal<G, T>: Clone,
    VertexPosition<G>: Add<ScaledFaceNormal<G, T>, Output = VertexPosition<G>> + Clone,
{
    // Collect all the vertex keys of the face along with their translated
    // geometries.
    let (sources, destinations, face) = {
        let face = match mutation.as_mesh().face(abc) {
            Some(face) => face,
            _ => return Err(GraphError::TopologyNotFound.into()),
        };
        let translation = face.normal()? * distance;
        let sources = face
            .vertices()
            .map(|vertex| vertex.key())
            .collect::<Vec<_>>();
        let destinations = face
            .vertices()
            .map(|vertex| {
                let mut geometry = vertex.geometry.clone();
                *geometry.as_position_mut() = geometry.as_position().clone() + translation.clone();
                geometry
            })
            .collect::<Vec<_>>();
        (sources, destinations, face.geometry.clone())
    };
    // This is the point of no return; the mesh has been mutated. Unwrap
    // results.
    mutation.remove_face(abc).unwrap();
    let destinations = destinations
        .into_iter()
        .map(|vertex| mutation.insert_vertex(vertex))
        .collect::<Vec<_>>();
    // Use the keys for the existing vertices and the translated geometries
    // to construct the extruded face and its connective faces.
    let extrusion = mutation
        .insert_face(&destinations, (Default::default(), face))
        .unwrap();
    for ((a, c), (b, d)) in sources
        .into_iter()
        .zip(destinations.into_iter())
        .collect::<Vec<_>>()
        .perimeter()
    {
        // TODO: Split these faces to form triangles.
        mutation
            .insert_face(&[a, b, d, c], Default::default())
            .unwrap();
    }
    Ok(extrusion)
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use generate::*;
    use graph::*;

    #[test]
    fn circulate_over_edges() {
        let mesh = sphere::UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();
        let face = mesh.faces().nth(0).unwrap();

        // All faces should be triangles and should have three edges.
        assert_eq!(3, face.edges().count());
    }

    #[test]
    fn circulate_over_faces() {
        let mesh = sphere::UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();
        let face = mesh.faces().nth(0).unwrap();

        // No matter which face is selected, it should have three neighbors.
        assert_eq!(3, face.faces().count());
    }

    #[test]
    fn extrude_face() {
        let mut mesh = sphere::UvSphere::new(3, 2)
            .polygons_with_position() // 6 triangles, 18 vertices.
            .collect::<Mesh<Point3<f32>>>();
        {
            let key = mesh.faces().nth(0).unwrap().key();
            let face = mesh.face_mut(key).unwrap().extrude(1.0).unwrap();

            // The extruded face, being a triangle, should have three
            // neighboring faces.
            assert_eq!(3, face.faces().count());
        }

        assert_eq!(8, mesh.vertex_count());
        // The mesh begins with 18 edges. The extrusion adds three quads with
        // four interior edges each, so there are `18 + (3 * 4)` edges.
        assert_eq!(30, mesh.edge_count());
        // All faces are triangles and the mesh begins with six such faces. The
        // extruded face remains, in addition to three connective faces, each
        // of which is constructed from quads.
        assert_eq!(9, mesh.face_count());
    }

    #[test]
    fn triangulate_mesh() {
        let (indeces, vertices) = cube::Cube::new()
            .polygons_with_position() // 6 quads, 24 vertices.
            .flat_index_vertices(HashIndexer::default());
        let mut mesh = Mesh::<Point3<f32>>::from_raw_buffers(indeces, vertices, 4).unwrap();
        mesh.triangulate().unwrap();

        // There are 8 unique vertices and a vertex is added for each quad,
        // yielding `8 + 6` vertices.
        assert_eq!(14, mesh.vertex_count());
        assert_eq!(72, mesh.edge_count());
        // Each quad becomes a tetrahedron, so 6 quads become 24 triangles.
        assert_eq!(24, mesh.face_count());
    }
}
