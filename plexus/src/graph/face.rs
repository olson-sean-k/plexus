// TODO: The `derivative` attribute macro causes this lint failure.
#![allow(clippy::match_single_binding)]

use derivative::Derivative;
use fool::BoolExt;
use slotmap::DefaultKey;
use smallvec::SmallVec;
use std::cmp;
use std::collections::HashSet;
use std::mem;
use std::ops::{Deref, DerefMut};
use theon::query::{Intersection, Line, Plane};
use theon::space::{EuclideanSpace, FiniteDimensional, Scalar, Vector};
use theon::AsPosition;
use typenum::U3;

use crate::entity::borrow::{Reborrow, ReborrowMut};
use crate::entity::storage::{AsStorage, AsStorageMut, OpaqueKey, SlotStorage};
use crate::entity::traverse::{Adjacency, Breadth, Depth, Traversal};
use crate::entity::view::{Bind, ClosedView, Orphan, Rebind, Unbind, View};
use crate::entity::Entity;
use crate::graph::edge::{Arc, ArcKey, ArcOrphan, ArcView, Edge};
use crate::graph::geometry::{
    FaceCentroid, FaceNormal, FacePlane, Geometric, Geometry, GraphGeometry, VertexPosition,
};
use crate::graph::mutation::face::{
    self, FaceBridgeCache, FaceExtrudeCache, FaceInsertCache, FacePokeCache, FaceRemoveCache,
    FaceSplitCache,
};
use crate::graph::mutation::{Consistent, Mutable, Mutation};
use crate::graph::trace::{Trace, TraceFirst};
use crate::graph::vertex::{Vertex, VertexKey, VertexOrphan, VertexView};
use crate::graph::{GraphError, MeshGraph, OptionExt as _, ResultExt as _, Selector};
use crate::transact::{Mutate, Transact};
use crate::{DynamicArity, IteratorExt as _, StaticArity};

use Selector::ByIndex;

// TODO: The API for faces and rings presents fuzzy distinctions; many
//       operations supported by `FaceView` could be supported by `Ring` as
//       well (specifically, all topological operations where a `Face` is
//       unnecessary). In essence, a face is simply a ring with an associated
//       payload that describes its path and geometry. The geometry is the most
//       notable difference, keeping in mind that in a consistent graph all arcs
//       are part of a ring.

/// Structure that participates in a ring.
///
/// Types that implement this trait expose the ring in which they participate
/// and support ring-like operations.
pub trait Ringoid<B>: DynamicArity<Dynamic = usize> + Sized
where
    B: Reborrow,
    B::Target: AsStorage<Arc<Geometry<B>>> + Consistent + Geometric,
{
    fn into_ring(self) -> Ring<B>;

    fn ring(&self) -> Ring<&B::Target>;

    /// Gets the distance (number of arcs) between two vertices within the ring.
    fn distance(
        &self,
        source: Selector<VertexKey>,
        destination: Selector<VertexKey>,
    ) -> Result<usize, GraphError>
    where
        B::Target: AsStorage<Vertex<Geometry<B>>>,
    {
        let arity = self.arity();
        let index_of_selector = |selector: Selector<_>| match selector {
            Selector::ByKey(key) => self
                .ring()
                .vertices()
                .keys()
                .enumerate()
                .find(|(_, a)| *a == key)
                .map(|(index, _)| index)
                .ok_or_else(|| GraphError::TopologyNotFound),
            Selector::ByIndex(index) => {
                if index >= arity {
                    Err(GraphError::TopologyNotFound)
                }
                else {
                    Ok(index)
                }
            }
        };
        let source = index_of_selector(source)? as isize;
        let destination = index_of_selector(destination)? as isize;
        let difference = (source - destination).abs() as usize;
        Ok(cmp::min(difference, arity - difference))
    }
}

/// Face entity.
#[derivative(Clone, Copy, Debug, Hash)]
#[derive(Derivative)]
pub struct Face<G>
where
    G: GraphGeometry,
{
    /// Geometry.
    ///
    /// The type of this field is derived from `GraphGeometry`.
    #[derivative(Debug = "ignore", Hash = "ignore")]
    pub geometry: G::Face,
    /// Required key into the leading arc.
    pub(in crate::graph) arc: ArcKey,
}

impl<G> Face<G>
where
    G: GraphGeometry,
{
    pub(in crate::graph) fn new(arc: ArcKey, geometry: G::Face) -> Self {
        Face { geometry, arc }
    }
}

impl<G> Entity for Face<G>
where
    G: GraphGeometry,
{
    type Key = FaceKey;
    type Storage = SlotStorage<Self>;
}

/// Face key.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct FaceKey(DefaultKey);

impl OpaqueKey for FaceKey {
    type Inner = DefaultKey;

    fn from_inner(key: Self::Inner) -> Self {
        FaceKey(key)
    }

    fn into_inner(self) -> Self::Inner {
        self.0
    }
}

/// View of a face entity.
///
/// Exposes references to a face entity and provides the primary face API. Faces
/// are notated by the path of their associated ring. A triangular face with a
/// perimeter formed by vertices $A$, $B$, and $C$ is notated
/// $\overrightarrow{\\{A,B,C\\}}$. While the precise ordering of vertices is
/// determined by a face's leading arc, the same face may be notated using
/// rotations of this set, such as $\overrightarrow{\\{B,C,A\\}}$.
///
/// See the module documentation for more information about views.
pub struct FaceView<B>
where
    B: Reborrow,
    B::Target: AsStorage<Face<Geometry<B>>> + Geometric,
{
    inner: View<B, Face<Geometry<B>>>,
}

impl<B, M> FaceView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Face<Geometry<B>>> + Geometric,
{
    pub fn to_ref(&self) -> FaceView<&M> {
        self.inner.to_ref().into()
    }
}

impl<B, M> FaceView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorage<Face<Geometry<B>>> + Geometric,
{
    pub fn to_mut(&mut self) -> FaceView<&mut M> {
        self.inner.to_mut().into()
    }
}

impl<'a, M> FaceView<&'a mut M>
where
    M: AsStorageMut<Face<Geometry<M>>> + Geometric,
{
    // TODO: Relocate this documentation of `into_ref`.
    /// # Examples
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::N64;
    /// use nalgebra::Point3;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::cube::Cube;
    /// use plexus::primitive::generate::Position;
    ///
    /// let mut graph = Cube::new()
    ///     .polygons::<Position<Point3<N64>>>()
    ///     .collect::<MeshGraph<Point3<f64>>>();
    /// let key = graph.faces().nth(0).unwrap().key();
    /// let face = graph
    ///     .face_mut(key)
    ///     .unwrap()
    ///     .extrude_with_offset(1.0)
    ///     .unwrap()
    ///     .into_ref();
    ///
    /// // This would not be possible without conversion into an immutable view.
    /// let _ = face.into_arc();
    /// let _ = face.into_arc().into_next_arc();
    /// ```
    pub fn into_ref(self) -> FaceView<&'a M> {
        self.inner.into_ref().into()
    }
}

/// Reachable API.
impl<B, M> FaceView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Face<Geometry<B>>> + Geometric,
{
    pub(in crate::graph) fn into_reachable_arc(self) -> Option<ArcView<B>> {
        let key = self.arc;
        self.rebind(key)
    }
}

impl<B, M> FaceView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Face<Geometry<B>>> + Consistent + Geometric,
{
    /// Converts the face into its ring.
    pub fn into_ring(self) -> Ring<B> {
        Ringoid::into_ring(self)
    }

    /// Converts the face into its leading arc.
    pub fn into_arc(self) -> ArcView<B> {
        self.into_reachable_arc().expect_consistent()
    }

    /// Gets the ring of the face.
    pub fn ring(&self) -> Ring<&M> {
        Ringoid::ring(self)
    }

    /// Gets the leading arc of the face.
    pub fn arc(&self) -> ArcView<&M> {
        self.to_ref().into_arc()
    }

    /// Gets an iterator that traverses adjacent faces by breadth.
    ///
    /// The traversal moves from the face to its adjacent faces and so on. If
    /// there are disjoint subgraphs in the graph, then a traversal will not
    /// reach every face in the graph.
    pub fn traverse_by_breadth<'a>(&'a self) -> impl Clone + Iterator<Item = FaceView<&'a M>>
    where
        M: 'a,
    {
        Traversal::<_, _, Breadth>::from(self.to_ref())
    }

    /// Gets an iterator that traverses adjacent faces by depth.
    ///
    /// The traversal moves from the face to its adjacent faces and so on. If
    /// there are disjoint subgraphs in the graph, then a traversal will not
    /// reach every face in the graph.
    pub fn traverse_by_depth<'a>(&'a self) -> impl Clone + Iterator<Item = FaceView<&'a M>>
    where
        M: 'a,
    {
        Traversal::<_, _, Depth>::from(self.to_ref())
    }

    /// Gets an iterator of views over the arcs in the face's ring.
    pub fn interior_arcs<'a>(&'a self) -> impl Clone + Iterator<Item = ArcView<&'a M>>
    where
        M: 'a,
    {
        ArcCirculator::from(self.ring())
    }

    /// Gets an iterator of views over adjacent faces.
    pub fn adjacent_faces<'a>(&'a self) -> impl Clone + Iterator<Item = FaceView<&'a M>>
    where
        M: 'a,
    {
        FaceCirculator::from(ArcCirculator::from(self.ring()))
    }
}

impl<B, M, G> FaceView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Consistent
        + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    /// Gets an iterator of views over the vertices that form the face.
    pub fn vertices<'a>(&'a self) -> impl Clone + Iterator<Item = VertexView<&'a M>>
    where
        M: 'a,
        G: 'a,
    {
        ArcCirculator::from(self.ring()).map(|arc| arc.into_source_vertex())
    }

    pub fn centroid(&self) -> VertexPosition<G>
    where
        G: FaceCentroid,
        G::Vertex: AsPosition,
    {
        G::centroid(self.to_ref()).expect_consistent()
    }

    pub fn normal(&self) -> Result<Vector<VertexPosition<G>>, GraphError>
    where
        G: FaceNormal,
        G::Vertex: AsPosition,
    {
        G::normal(self.to_ref())
    }

    pub fn plane(&self) -> Result<Plane<VertexPosition<G>>, GraphError>
    where
        G: FacePlane,
        G::Vertex: AsPosition,
        VertexPosition<G>: FiniteDimensional<N = U3>,
    {
        G::plane(self.to_ref())
    }
}

impl<B, M> FaceView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorageMut<Arc<Geometry<B>>> + AsStorage<Face<Geometry<B>>> + Consistent + Geometric,
{
    /// Gets an iterator of orphan views over the arcs in the face's ring.
    pub fn interior_arc_orphans<'a>(&'a mut self) -> impl Iterator<Item = ArcOrphan<Geometry<B>>>
    where
        M: 'a,
    {
        ArcCirculator::from(self.to_mut().into_ring())
    }
}

impl<B, M> FaceView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorageMut<Face<Geometry<B>>> + Consistent + Geometric,
{
    /// Gets an iterator of orphan views over adjacent faces.
    pub fn adjacent_face_orphans<'a>(&'a mut self) -> impl Iterator<Item = FaceOrphan<Geometry<B>>>
    where
        M: 'a,
    {
        FaceCirculator::from(ArcCirculator::from(self.to_mut().into_ring()))
    }
}

impl<B, M> FaceView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorage<Arc<Geometry<B>>>
        + AsStorage<Face<Geometry<B>>>
        + AsStorageMut<Vertex<Geometry<B>>>
        + Consistent
        + Geometric,
{
    /// Gets an iterator of orphan views over the vertices that form the face.
    pub fn vertex_orphans<'a>(&'a mut self) -> impl Iterator<Item = VertexOrphan<Geometry<B>>>
    where
        M: 'a,
    {
        VertexCirculator::from(ArcCirculator::from(self.to_mut().into_ring()))
    }
}

impl<B, M, G> FaceView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorage<Arc<G>>
        + AsStorage<Face<G>>
        + AsStorageMut<Vertex<G>>
        + Consistent
        + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    /// Flattens the face by translating the positions of all vertices into a
    /// best-fit plane.
    ///
    /// Returns an error if a best-fit plane could not be computed or positions
    /// could not be translated into the plane.
    pub fn flatten(&mut self) -> Result<(), GraphError>
    where
        G: FacePlane,
        G::Vertex: AsPosition,
        VertexPosition<G>: EuclideanSpace + FiniteDimensional<N = U3>,
    {
        if self.arity() == 3 {
            return Ok(());
        }
        let plane = self.plane()?;
        for mut vertex in self.vertex_orphans() {
            let position = *vertex.position();
            let line = Line::<VertexPosition<G>> {
                origin: position,
                direction: plane.normal,
            };
            // TODO: If the intersection yields no result, then this may fail
            //       after mutating positions in the graph. Consider using
            //       read/write stages to avoid partial completion.
            let distance = plane
                .intersection(&line)
                .ok_or_else(|| GraphError::Geometry)?;
            let translation = *line.direction.get() * distance;
            *vertex.geometry.as_position_mut() = position + translation;
        }
        Ok(())
    }
}

impl<'a, M, G> FaceView<&'a mut M>
where
    M: AsStorage<Arc<G>>
        + AsStorage<Edge<G>>
        + AsStorage<Face<G>>
        + AsStorage<Vertex<G>>
        + Default
        + Mutable<Geometry = G>,
    G: GraphGeometry,
{
    /// Splits the face by bisecting it with a composite edge inserted between
    /// two non-adjacent vertices within the face's perimeter.
    ///
    /// The vertices can be chosen by key or index, where index selects the
    /// $n^\text{th}$ vertex within the face's ring.
    ///
    /// This can be thought of as the opposite of `merge`.
    ///
    /// Returns the arc inserted from the source vertex to the destination
    /// vertex if successful. If a face $\overrightarrow{\\{A,B, C,D\\}}$ is
    /// split from $A$ to $C$, then it will be decomposed into faces in the
    /// rings $\overrightarrow{\\{A,B,C\\}}$ and $\overrightarrow{\\{C,D,A\\}}$
    /// and the arc $\overrightarrow{AC}$ will be returned.
    ///
    /// # Errors
    ///
    /// Returns an error if either of the given vertices cannot be found, are
    /// not within the face's perimeter, or the distance between the vertices
    /// along the ring is less than two.
    ///
    /// # Examples
    ///
    /// Splitting a quadrilateral face:
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point2;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::Tetragon;
    ///
    /// let mut graph = MeshGraph::<Point2<f64>>::from_raw_buffers(
    ///     vec![Tetragon::new(0usize, 1, 2, 3)],
    ///     vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    /// )
    /// .unwrap();
    /// let key = graph.faces().nth(0).unwrap().key();
    /// let arc = graph
    ///     .face_mut(key)
    ///     .unwrap()
    ///     .split(ByIndex(0), ByIndex(2))
    ///     .unwrap()
    ///     .into_ref();
    /// ```
    pub fn split(
        self,
        source: Selector<VertexKey>,
        destination: Selector<VertexKey>,
    ) -> Result<ArcView<&'a mut M>, GraphError> {
        let key_at_index = |index| {
            self.vertices()
                .nth(index)
                .ok_or_else(|| GraphError::TopologyNotFound)
                .map(|vertex| vertex.key())
        };
        let source = source.key_or_else(key_at_index)?;
        let destination = destination.key_or_else(key_at_index)?;
        let (storage, abc) = self.unbind();
        let cache = FaceSplitCache::snapshot(&storage, abc, source, destination)?;
        Ok(Mutation::replace(storage, Default::default())
            .commit_with(|mutation| face::split(mutation, cache))
            .map(|(storage, arc)| Bind::bind(storage, arc).expect_consistent())
            .expect_consistent())
    }

    /// Merges the face into an adjacent face over their shared composite edge.
    ///
    /// The adjacent face can be chosen by key or index, where index selects
    /// the $n^\text{th}$ adjacent face.
    ///
    /// This can be thought of as the opposite of `split`.
    ///
    /// Returns the merged face if successful.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination face cannot be found or is not
    /// adjacent to the initiating face.
    ///
    /// # Examples
    ///
    /// Merging two adjacent quadrilateral faces:
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point2;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::Tetragon;
    ///
    /// let mut graph = MeshGraph::<Point2<f64>>::from_raw_buffers(
    ///     vec![Tetragon::new(0usize, 1, 2, 3), Tetragon::new(0, 3, 4, 5)],
    ///     vec![
    ///         (0.0, 0.0),  // 0
    ///         (1.0, 0.0),  // 1
    ///         (1.0, 1.0),  // 2
    ///         (0.0, 1.0),  // 3
    ///         (-1.0, 1.0), // 4
    ///         (-1.0, 0.0), // 5
    ///     ],
    /// )
    /// .unwrap();
    ///
    /// let key = graph.faces().nth(0).unwrap().key();
    /// let face = graph
    ///     .face_mut(key)
    ///     .unwrap()
    ///     .merge(ByIndex(0))
    ///     .unwrap()
    ///     .into_ref();
    /// ```
    pub fn merge(self, destination: Selector<FaceKey>) -> Result<Self, GraphError> {
        let destination = destination.key_or_else(|index| {
            self.adjacent_faces()
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
        let geometry = self.geometry;
        // TODO: Batch this operation by using the mutation API instead.
        let arc: ArcView<_> = self.rebind(ab).expect_consistent();
        Ok(arc
            .remove()
            // Removing an edge between faces must yield a vertex.
            .expect_consistent()
            .into_outgoing_arc()
            .into_ring()
            .get_or_insert_face_with(|| geometry))
    }

    /// Connects faces with equal arity with faces inserted along their
    /// perimeters.
    ///
    /// The inserted faces are always quadrilateral. Both the initiating face
    /// and destination face are removed.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination face cannot be found or the arity of
    /// the face and its destination are not the same.
    pub fn bridge(self, destination: FaceKey) -> Result<(), GraphError> {
        let (storage, source) = self.unbind();
        let cache = FaceBridgeCache::snapshot(&storage, source, destination)?;
        Mutation::replace(storage, Default::default())
            .commit_with(|mutation| face::bridge(mutation, cache))
            .expect_consistent();
        Ok(())
    }

    /// Decomposes the face into triangles. Does nothing if the face is
    /// triangular.
    ///
    /// Returns the terminating face of the decomposition.
    pub fn triangulate(self) -> Self {
        let mut face = self;
        while face.arity() > 3 {
            face = face
                .split(ByIndex(0), ByIndex(2))
                .expect_consistent()
                .into_face()
                .expect_consistent();
        }
        face
    }

    /// Subdivides the face about a vertex. A triangle fan is formed from each
    /// arc in the face's perimeter and the vertex.
    ///
    /// Poking inserts a new vertex with geometry provided by the given
    /// function.
    ///
    /// Returns the inserted vertex.
    ///
    /// # Examples
    ///
    /// Forming a pyramid from a triangular face:
    ///
    /// ```rust
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use nalgebra::Point3;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::Trigon;
    /// use plexus::AsPosition;
    ///
    /// let mut graph = MeshGraph::<Point3<f64>>::from_raw_buffers(
    ///     vec![Trigon::new(0usize, 1, 2)],
    ///     vec![(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
    /// )
    /// .unwrap();
    /// let key = graph.faces().nth(0).unwrap().key();
    /// let mut face = graph.face_mut(key).unwrap();
    ///
    /// // See `poke_with_offset`, which provides this functionality.
    /// let mut geometry = face.centroid();
    /// let position = geometry.as_position().clone() + face.normal().unwrap();
    /// face.poke_with(move || {
    ///     *geometry.as_position_mut() = position;
    ///     geometry
    /// });
    /// ```
    pub fn poke_with<F>(self, f: F) -> VertexView<&'a mut M>
    where
        F: FnOnce() -> G::Vertex,
    {
        let (storage, abc) = self.unbind();
        // This should never fail here.
        let cache = FacePokeCache::snapshot(&storage, abc).expect_consistent();
        Mutation::replace(storage, Default::default())
            .commit_with(|mutation| face::poke_with(mutation, cache, f))
            .map(|(storage, vertex)| Bind::bind(storage, vertex).expect_consistent())
            .expect_consistent()
    }

    /// Subdivides the face about its centroid. A triangle fan is formed from
    /// each arc in the face's perimeter and a vertex inserted at the centroid.
    ///
    /// Returns the inserted vertex.
    pub fn poke_at_centroid(self) -> VertexView<&'a mut M>
    where
        G: FaceCentroid,
        G::Vertex: AsPosition,
    {
        let mut geometry = self.arc().source_vertex().geometry;
        let centroid = self.centroid();
        self.poke_with(move || {
            *geometry.as_position_mut() = centroid;
            geometry
        })
    }

    /// Subdivides the face about its centroid. A triangle fan is formed from
    /// each arc in the face's perimeter and a vertex inserted at the centroid.
    /// The inserted vertex is then translated along the initiating face's
    /// normal by the given offset.
    ///
    /// Returns the inserted vertex if successful.
    ///
    /// # Errors
    ///
    /// Returns an error if the geometry could not be computed.
    ///
    /// # Examples
    ///
    /// Constructing a "spikey" sphere:
    ///
    /// ```rust
    /// # extern crate decorum;
    /// # extern crate nalgebra;
    /// # extern crate plexus;
    /// #
    /// use decorum::N64;
    /// use nalgebra::Point3;
    /// use plexus::graph::MeshGraph;
    /// use plexus::prelude::*;
    /// use plexus::primitive::generate::Position;
    /// use plexus::primitive::sphere::UvSphere;
    ///
    /// let mut graph = UvSphere::new(16, 8)
    ///     .polygons::<Position<Point3<N64>>>()
    ///     .collect::<MeshGraph<Point3<f64>>>();
    /// let keys = graph.faces().map(|face| face.key()).collect::<Vec<_>>();
    /// for key in keys {
    ///     graph.face_mut(key).unwrap().poke_with_offset(0.5).unwrap();
    /// }
    /// ```
    pub fn poke_with_offset<T>(self, offset: T) -> Result<VertexView<&'a mut M>, GraphError>
    where
        T: Into<Scalar<VertexPosition<G>>>,
        G: FaceCentroid + FaceNormal,
        G::Vertex: AsPosition,
        VertexPosition<G>: EuclideanSpace,
    {
        let mut geometry = self.arc().source_vertex().geometry;
        let position = self.centroid() + (self.normal()? * offset.into());
        Ok(self.poke_with(move || {
            *geometry.as_position_mut() = position;
            geometry
        }))
    }

    /// Extrudes the face along its normal.
    ///
    /// Returns the extruded face.
    ///
    /// # Errors
    ///
    /// Returns an error if the geometry could not be computed.
    pub fn extrude_with_offset<T>(self, offset: T) -> Result<FaceView<&'a mut M>, GraphError>
    where
        T: Into<Scalar<VertexPosition<G>>>,
        G: FaceNormal,
        G::Vertex: AsPosition,
        VertexPosition<G>: EuclideanSpace,
    {
        let translation = self.normal()? * offset.into();
        Ok(self.extrude_with_translation(translation))
    }

    /// Extrudes the face along a translation.
    ///
    /// Returns the extruded face.
    ///
    /// # Errors
    ///
    /// Returns an error if the geometry could not be computed.
    pub fn extrude_with_translation(
        self,
        translation: Vector<VertexPosition<G>>,
    ) -> FaceView<&'a mut M>
    where
        G::Vertex: AsPosition,
        VertexPosition<G>: EuclideanSpace,
    {
        self.extrude_with(|geometry| geometry.map_position(|position| *position + translation))
    }

    /// Extrudes a face using the given vertex geometry.
    ///
    /// Returns the extruded face.
    ///
    /// # Errors
    ///
    /// Returns an error if the geometry could not be computed.
    pub fn extrude_with<F>(self, f: F) -> FaceView<&'a mut M>
    where
        F: Fn(G::Vertex) -> G::Vertex,
    {
        let (storage, abc) = self.unbind();
        // This should never fail here.
        let cache = FaceExtrudeCache::snapshot(&storage, abc).expect_consistent();
        Mutation::replace(storage, Default::default())
            .commit_with(|mutation| face::extrude_with(mutation, cache, f))
            .map(|(storage, face)| Bind::bind(storage, face).expect_consistent())
            .expect_consistent()
    }

    /// Removes the face.
    ///
    /// Returns the remaining ring of the face if it is not entirely disjoint.
    pub fn remove(self) -> Option<Ring<&'a mut M>> {
        let (storage, abc) = self.unbind();
        // This should never fail here.
        let cache = FaceRemoveCache::snapshot(&storage, abc).expect_consistent();
        Mutation::replace(storage, Default::default())
            .commit_with(|mutation| face::remove(mutation, cache))
            .map(|(storage, face)| ArcView::bind(storage, face.arc))
            .expect_consistent()
            .map(|arc| arc.into_ring())
    }
}

impl<B, M, G> Adjacency for FaceView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Face<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Output = SmallVec<[Self::Key; 8]>;

    fn adjacency(&self) -> Self::Output {
        self.adjacent_faces().keys().collect()
    }
}

impl<B, M, G> ClosedView for FaceView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Key = FaceKey;
    type Entity = Face<G>;

    /// Gets the key for the face.
    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<B, M, G> Clone for FaceView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
    View<B, Face<G>>: Clone,
{
    fn clone(&self) -> Self {
        FaceView {
            inner: self.inner.clone(),
        }
    }
}

impl<B, M, G> Copy for FaceView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
    View<B, Face<G>>: Copy,
{
}

impl<B, M, G> Deref for FaceView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Target = Face<G>;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl<B, M, G> DerefMut for FaceView<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorageMut<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

impl<B, M, G> DynamicArity for FaceView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Face<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Dynamic = usize;

    /// Gets the arity of the face. This is the number of arcs that form the
    /// face's ring.
    fn arity(&self) -> Self::Dynamic {
        self.interior_arcs().count()
    }
}

impl<B, M, G> From<View<B, Face<G>>> for FaceView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(view: View<B, Face<G>>) -> Self {
        FaceView { inner: view }
    }
}

impl<B, M, G> Into<View<B, Face<G>>> for FaceView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn into(self) -> View<B, Face<G>> {
        let FaceView { inner, .. } = self;
        inner
    }
}

impl<B, M, G> PartialEq for FaceView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Face<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<B, M, G> Ringoid<B> for FaceView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + AsStorage<Face<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn into_ring(self) -> Ring<B> {
        self.into_arc().into_ring()
    }

    fn ring(&self) -> Ring<&M> {
        self.to_ref().into_ring()
    }
}

impl<B, M, G> StaticArity for FaceView<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Static = <MeshGraph<G> as StaticArity>::Static;

    const ARITY: Self::Static = MeshGraph::<G>::ARITY;
}

/// Orphan view of a face.
///
/// Provides mutable access to a face's geometry. See the module documentation
/// for more information about topological views.
pub struct FaceOrphan<'a, G>
where
    G: GraphGeometry,
{
    inner: Orphan<'a, Face<G>>,
}

impl<'a, G> ClosedView for FaceOrphan<'a, G>
where
    G: GraphGeometry,
{
    type Key = FaceKey;
    type Entity = Face<G>;

    fn key(&self) -> Self::Key {
        self.inner.key()
    }
}

impl<'a, G> Deref for FaceOrphan<'a, G>
where
    G: GraphGeometry,
{
    type Target = Face<G>;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl<'a, G> DerefMut for FaceOrphan<'a, G>
where
    G: GraphGeometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

impl<'a, M, G> From<FaceView<&'a mut M>> for FaceOrphan<'a, G>
where
    M: AsStorageMut<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(face: FaceView<&'a mut M>) -> Self {
        Orphan::from(face.inner).into()
    }
}

impl<'a, G> From<Orphan<'a, Face<G>>> for FaceOrphan<'a, G>
where
    G: GraphGeometry,
{
    fn from(inner: Orphan<'a, Face<G>>) -> Self {
        FaceOrphan { inner }
    }
}

impl<'a, M, G> From<View<&'a mut M, Face<G>>> for FaceOrphan<'a, G>
where
    M: AsStorageMut<Face<G>> + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(view: View<&'a mut M, Face<G>>) -> Self {
        FaceOrphan { inner: view.into() }
    }
}

/// Ring.
///
/// Rings are closed paths formed by arcs and their immediate adjacent arcs. In
/// a consistent graph, every arc forms such a path. Such paths may or may not
/// be occupied by faces.
///
/// Rings are notated by their path. A ring with a perimeter formed by vertices
/// $A$, $B$, and $C$ is notated $\overrightarrow{\\{A,B,C\\}}$. Note that
/// rotations of the set of vertices are equivalent, such as
/// $\overrightarrow{\\{B,C,A\\}}$.
pub struct Ring<B>
where
    B: Reborrow,
    B::Target: AsStorage<Arc<Geometry<B>>> + Consistent + Geometric,
{
    arc: ArcView<B>,
}

impl<B, M> Ring<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + Consistent + Geometric,
{
    /// Gets an iterator of views over the arcs within the ring.
    pub fn interior_arcs<'a>(&'a self) -> impl Clone + Iterator<Item = ArcView<&'a M>>
    where
        M: 'a,
    {
        ArcCirculator::from(self.to_ref())
    }

    pub fn to_ref(&self) -> Ring<&M> {
        self.arc.to_ref().into_ring()
    }
}

impl<B, M> Ring<B>
where
    B: ReborrowMut<Target = M>,
    M: AsStorageMut<Arc<Geometry<B>>> + Consistent + Geometric,
{
    pub fn to_mut(&mut self) -> Ring<&mut M> {
        self.arc.to_mut().into_ring()
    }
}

impl<'a, M> Ring<&'a mut M>
where
    M: AsStorageMut<Arc<Geometry<M>>> + Consistent + Geometric,
{
    pub fn into_ref(self) -> Ring<&'a M> {
        self.arc.into_ref().into_ring()
    }
}

impl<B, M> Ring<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + Consistent + Geometric,
{
    /// Converts the ring into its originating arc.
    pub fn into_arc(self) -> ArcView<B> {
        self.arc
    }

    /// Gets the originating arc of the ring.
    pub fn arc(&self) -> ArcView<&M> {
        self.arc.to_ref()
    }
}

impl<B, M> Ring<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Vertex<Geometry<B>>> + Consistent + Geometric,
{
    /// Gets an iterator of views over the vertices within the ring.
    pub fn vertices<'a>(&'a self) -> impl Clone + Iterator<Item = VertexView<&'a M>>
    where
        M: 'a,
    {
        ArcCirculator::from(self.to_ref()).map(|arc| arc.into_source_vertex())
    }
}

impl<B, M> Ring<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<Geometry<B>>> + AsStorage<Face<Geometry<B>>> + Consistent + Geometric,
{
    /// Converts the ring into its face.
    ///
    /// If the path has no associated face, then `None` is returned.
    pub fn into_face(self) -> Option<FaceView<B>> {
        let arc = self.into_arc();
        let key = arc.face;
        key.map(|key| arc.rebind(key).expect_consistent())
    }

    /// Gets the face of the ring.
    ///
    /// If the path has no associated face, then `None` is returned.
    pub fn face(&self) -> Option<FaceView<&M>> {
        self.to_ref().into_face()
    }
}

impl<'a, M, G> Ring<&'a mut M>
where
    M: AsStorage<Vertex<G>>
        + AsStorage<Arc<G>>
        + AsStorage<Face<G>>
        + Default
        + Mutable<Geometry = G>,
    G: GraphGeometry,
{
    /// Gets the face of the ring or inserts a face if one does not already
    /// exist.
    ///
    /// Returns the inserted face.
    pub fn get_or_insert_face(self) -> FaceView<&'a mut M> {
        self.get_or_insert_face_with(Default::default)
    }

    /// Gets the face of the ring or inserts a face if one does not already
    /// exist.
    ///
    /// If a face is inserted, then the given function is used to get the
    /// geometry for the face.
    ///
    /// Returns the inserted face.
    pub fn get_or_insert_face_with<F>(self, f: F) -> FaceView<&'a mut M>
    where
        F: FnOnce() -> G::Face,
    {
        let key = self.arc.face;
        if let Some(key) = key {
            self.arc.rebind(key).expect_consistent()
        }
        else {
            let perimeter = self.vertices().keys().collect::<Vec<_>>();
            let (storage, _) = self.arc.unbind();
            // This should never fail here.
            let cache = FaceInsertCache::snapshot(&storage, &perimeter).expect_consistent();
            Mutation::replace(storage, Default::default())
                .commit_with(|mutation| {
                    face::insert_with(mutation, cache, || (Default::default(), f()))
                })
                .map(|(storage, face)| Bind::bind(storage, face).expect_consistent())
                .expect_consistent()
        }
    }
}

impl<B, M, G> DynamicArity for Ring<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Dynamic = usize;

    /// Gets the arity of the ring. This is the number of arcs that form the
    /// path.
    fn arity(&self) -> Self::Dynamic {
        self.interior_arcs().count()
    }
}

impl<B, M, G> From<ArcView<B>> for Ring<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(arc: ArcView<B>) -> Self {
        Ring { arc }
    }
}

impl<B, M, G> PartialEq for Ring<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn eq(&self, other: &Self) -> bool {
        let keys = |ring: &Self| ring.interior_arcs().keys().collect::<HashSet<_>>();
        keys(self) == keys(other)
    }
}

impl<B, M, G> Ringoid<B> for Ring<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn into_ring(self) -> Ring<B> {
        self
    }

    fn ring(&self) -> Ring<&M> {
        self.to_ref()
    }
}

impl<B, M, G> StaticArity for Ring<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    type Static = <MeshGraph<G> as StaticArity>::Static;

    const ARITY: Self::Static = MeshGraph::<G>::ARITY;
}

pub struct VertexCirculator<B>
where
    B: Reborrow,
    B::Target: AsStorage<Arc<Geometry<B>>> + Consistent + Geometric,
{
    inner: ArcCirculator<B>,
}

impl<B, M, G> VertexCirculator<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn next(&mut self) -> Option<VertexKey> {
        let ab = self.inner.next();
        ab.map(|ab| {
            let (_, b) = ab.into();
            b
        })
    }
}

impl<B, M, G> Clone for VertexCirculator<B>
where
    B: Clone + Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn clone(&self) -> Self {
        VertexCirculator {
            inner: self.inner.clone(),
        }
    }
}

impl<B, M, G> From<ArcCirculator<B>> for VertexCirculator<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(inner: ArcCirculator<B>) -> Self {
        VertexCirculator { inner }
    }
}

impl<'a, M, G> Iterator for VertexCirculator<&'a M>
where
    M: AsStorage<Arc<G>> + AsStorage<Vertex<G>> + Consistent + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    type Item = VertexView<&'a M>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).and_then(|key| Bind::bind(self.inner.storage, key))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // This requires consistency, because an inconsistent graph may not
        // produce the expected minimum of three vertices.
        (3, None)
    }
}

impl<'a, M, G> Iterator for VertexCirculator<&'a mut M>
where
    M: AsStorage<Arc<G>> + AsStorageMut<Vertex<G>> + Consistent + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    type Item = VertexOrphan<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        VertexCirculator::next(self).map(|key| {
            let vertex = self.inner.storage.as_storage_mut().get_mut(&key).unwrap();
            let vertex = unsafe { mem::transmute::<&'_ mut Vertex<G>, &'a mut Vertex<G>>(vertex) };
            Orphan::bind_unchecked(vertex, key).into()
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // This requires consistency, because an inconsistent graph may not
        // produce the expected minimum of three vertices.
        (3, None)
    }
}

pub struct ArcCirculator<B>
where
    B: Reborrow,
    B::Target: AsStorage<Arc<Geometry<B>>> + Consistent + Geometric,
{
    storage: B,
    arc: Option<ArcKey>,
    trace: TraceFirst<ArcKey>,
}

impl<B, M, G> ArcCirculator<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    #[allow(unstable_name_collisions)]
    fn next(&mut self) -> Option<ArcKey> {
        self.arc
            .and_then(|arc| self.trace.insert(arc).then_some(arc))
            .map(|arc| {
                self.arc = self
                    .storage
                    .reborrow()
                    .as_storage()
                    .get(&arc)
                    .and_then(|arc| arc.next);
                arc
            })
    }
}

impl<B, M, G> Clone for ArcCirculator<B>
where
    B: Clone + Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn clone(&self) -> Self {
        ArcCirculator {
            storage: self.storage.clone(),
            arc: self.arc,
            trace: self.trace,
        }
    }
}

impl<B, M, G> From<Ring<B>> for ArcCirculator<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(ring: Ring<B>) -> Self {
        let (storage, key) = ring.into_arc().unbind();
        ArcCirculator {
            storage,
            arc: Some(key),
            trace: Default::default(),
        }
    }
}

impl<'a, M, G> Iterator for ArcCirculator<&'a M>
where
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    type Item = ArcView<&'a M>;

    fn next(&mut self) -> Option<Self::Item> {
        ArcCirculator::next(self).and_then(|key| Bind::bind(self.storage, key))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // This requires consistency, because an inconsistent graph may not
        // produce the expected minimum of three arcs.
        (3, None)
    }
}

impl<'a, M, G> Iterator for ArcCirculator<&'a mut M>
where
    M: AsStorageMut<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    type Item = ArcOrphan<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        ArcCirculator::next(self).map(|key| {
            let arc = self.storage.as_storage_mut().get_mut(&key).unwrap();
            let arc = unsafe { mem::transmute::<&'_ mut Arc<G>, &'a mut Arc<G>>(arc) };
            Orphan::bind_unchecked(arc, key).into()
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // This requires consistency, because an inconsistent graph may not
        // produce the expected minimum of three arcs.
        (3, None)
    }
}

pub struct FaceCirculator<B>
where
    B: Reborrow,
    B::Target: AsStorage<Arc<Geometry<B>>> + Consistent + Geometric,
{
    inner: ArcCirculator<B>,
}

impl<B, M, G> FaceCirculator<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn next(&mut self) -> Option<FaceKey> {
        while let Some(ba) = self.inner.next().map(|ab| ab.into_opposite()) {
            if let Some(abc) = self
                .inner
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

impl<B, M, G> Clone for FaceCirculator<B>
where
    B: Clone + Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn clone(&self) -> Self {
        FaceCirculator {
            inner: self.inner.clone(),
        }
    }
}

impl<B, M, G> From<ArcCirculator<B>> for FaceCirculator<B>
where
    B: Reborrow<Target = M>,
    M: AsStorage<Arc<G>> + Consistent + Geometric<Geometry = G>,
    G: GraphGeometry,
{
    fn from(inner: ArcCirculator<B>) -> Self {
        FaceCirculator { inner }
    }
}

impl<'a, M, G> Iterator for FaceCirculator<&'a M>
where
    M: AsStorage<Arc<G>> + AsStorage<Face<G>> + Consistent + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    type Item = FaceView<&'a M>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).and_then(|key| Bind::bind(self.inner.storage, key))
    }
}

impl<'a, M, G> Iterator for FaceCirculator<&'a mut M>
where
    M: AsStorage<Arc<G>> + AsStorageMut<Face<G>> + Consistent + Geometric<Geometry = G>,
    G: 'a + GraphGeometry,
{
    type Item = FaceOrphan<'a, G>;

    fn next(&mut self) -> Option<Self::Item> {
        FaceCirculator::next(self).map(|key| {
            let face = self.inner.storage.as_storage_mut().get_mut(&key).unwrap();
            let face = unsafe { mem::transmute::<&'_ mut Face<G>, &'a mut Face<G>>(face) };
            Orphan::bind_unchecked(face, key).into()
        })
    }
}

#[cfg(test)]
mod tests {
    use decorum::N64;
    use nalgebra::{Point2, Point3};

    use crate::graph::MeshGraph;
    use crate::index::HashIndexer;
    use crate::prelude::*;
    use crate::primitive::cube::Cube;
    use crate::primitive::generate::Position;
    use crate::primitive::sphere::UvSphere;
    use crate::primitive::Tetragon;

    type E3 = Point3<N64>;

    #[test]
    fn circulate_over_arcs() {
        let graph = UvSphere::new(3, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f64>>>();
        let face = graph.faces().nth(0).unwrap();

        // All faces should be triangles and should have three edges.
        assert_eq!(3, face.interior_arcs().count());
    }

    #[test]
    fn circulate_over_faces() {
        let graph = UvSphere::new(3, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f64>>>();
        let face = graph.faces().nth(0).unwrap();

        // No matter which face is selected, it should have three adjacent
        // faces.
        assert_eq!(3, face.adjacent_faces().count());
    }

    #[test]
    fn remove_face() {
        let mut graph = UvSphere::new(3, 2)
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f64>>>();

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
    fn split_face() {
        let mut graph = MeshGraph::<Point2<f32>>::from_raw_buffers_with_arity(
            vec![0u32, 1, 2, 3],
            vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            4,
        )
        .unwrap();
        let abc = graph.faces().nth(0).unwrap().key();
        let arc = graph
            .face_mut(abc)
            .unwrap()
            .split(ByIndex(0), ByIndex(2))
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
            .polygons::<Position<E3>>() // 6 triangles, 18 vertices.
            .collect::<MeshGraph<Point3<f64>>>();
        {
            let key = graph.faces().nth(0).unwrap().key();
            let face = graph
                .face_mut(key)
                .unwrap()
                .extrude_with_offset(1.0)
                .unwrap()
                .into_ref();

            // The extruded face, being a triangle, should have three adjacent
            // faces.
            assert_eq!(3, face.adjacent_faces().count());
        }

        assert_eq!(8, graph.vertex_count());
        // The mesh begins with 18 arcs. The extrusion adds three quadrilaterals
        // with four interior arcs each, so there are `18 + (3 * 4)` arcs.
        assert_eq!(30, graph.arc_count());
        // All faces are triangles and the mesh begins with six such faces. The
        // extruded face remains, in addition to three connective faces, each of
        // which is constructed from quadrilaterals.
        assert_eq!(9, graph.face_count());
    }

    #[test]
    fn merge_faces() {
        // Construct a graph with two connected quadrilaterals.
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
    fn poke_face() {
        let mut graph = Cube::new()
            .polygons::<Position<E3>>() // 6 quadrilaterals, 24 vertices.
            .collect::<MeshGraph<Point3<f64>>>();
        let key = graph.faces().nth(0).unwrap().key();
        let vertex = graph.face_mut(key).unwrap().poke_at_centroid();

        // Diverging a quadrilateral yields a tetrahedron.
        assert_eq!(4, vertex.adjacent_faces().count());

        // Traverse to one of the triangles in the tetrahedron.
        let face = vertex.into_outgoing_arc().into_face().unwrap();

        assert_eq!(3, face.arity());

        // Diverge the triangle.
        let vertex = face.poke_at_centroid();

        assert_eq!(3, vertex.adjacent_faces().count());
    }

    #[test]
    fn triangulate_mesh() {
        let (indices, vertices) = Cube::new()
            .polygons::<Position<E3>>() // 6 quadrilaterals, 24 vertices.
            .index_vertices::<Tetragon<usize>, _>(HashIndexer::default());
        let mut graph = MeshGraph::<Point3<N64>>::from_raw_buffers(indices, vertices).unwrap();
        graph.triangulate();

        assert_eq!(8, graph.vertex_count());
        assert_eq!(36, graph.arc_count());
        assert_eq!(18, graph.edge_count());
        // Each quadrilateral becomes 2 triangles, so 6 quadrilaterals become
        // 12 triangles.
        assert_eq!(12, graph.face_count());
    }

    #[test]
    fn ring_distance() {
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
        let ring = face.into_ring();
        assert_eq!(2, ring.distance(keys[0].into(), keys[2].into()).unwrap());
        assert_eq!(1, ring.distance(keys[0].into(), keys[3].into()).unwrap());
        assert_eq!(0, ring.distance(keys[0].into(), keys[0].into()).unwrap());
    }
}
