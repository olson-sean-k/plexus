use failure::{Error, Fail};
use std::collections::{HashMap, HashSet};
use std::mem;
use std::ops::{Deref, DerefMut};

use geometry::Geometry;
use graph::{GraphError, Mesh, Perimeter};
use graph::mesh::{Connectivity, Edge, Face, Vertex};
use graph::storage::{EdgeKey, FaceKey, VertexKey};

enum Mode<I = (), B = ()> {
    Immediate(I),
    Batch(B),
}

type Mutant<'a, G> = Mode<&'a mut Mesh<G>, Mesh<G>>;

impl<'a, G> Mutant<'a, G>
where
    G: 'a + Geometry,
{
    pub fn get(&self) -> &Mesh<G> {
        match *self {
            Mode::Batch(ref mesh) => mesh,
            Mode::Immediate(ref mesh) => mesh,
        }
    }

    pub fn get_mut(&mut self) -> &mut Mesh<G> {
        match *self {
            Mode::Batch(ref mut mesh) => mesh,
            Mode::Immediate(ref mut mesh) => mesh,
        }
    }

    pub fn take_immediate(self) -> Option<&'a mut Mesh<G>> {
        match self {
            Mode::Immediate(mesh) => Some(mesh),
            _ => None,
        }
    }

    pub fn take_batch(self) -> Option<Mesh<G>> {
        match self {
            Mode::Batch(mesh) => Some(mesh),
            _ => None,
        }
    }
}

impl<'a, G> Deref for Mutant<'a, G>
where
    G: 'a + Geometry,
{
    type Target = Mesh<G>;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<'a, G> DerefMut for Mutant<'a, G>
where
    G: 'a + Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

pub trait ModalMutation<'a, G>: Deref<Target = Mutation<'a, G>> + DerefMut
where
    G: 'a + Geometry,
{
    fn insert_face(
        &mut self,
        vertices: &[VertexKey],
        geometry: (G::Edge, G::Face),
    ) -> Result<FaceKey, Error>;

    fn remove_face(&mut self, face: FaceKey) -> Result<Face<G>, Error>;
}

/// Mesh mutation.
///
/// Mutates a `Mesh`. This type provides general operations that are supported
/// by both immediate and batch mutations.
pub struct Mutation<'a, G>
where
    G: 'a + Geometry,
{
    mesh: Mutant<'a, G>,
}

impl<'a, G> Mutation<'a, G>
where
    G: 'a + Geometry,
{
    pub fn immediate(mesh: &'a mut Mesh<G>) -> ImmediateMutation<'a, G> {
        ImmediateMutation::new(Mutation {
            mesh: Mode::Immediate(mesh),
        })
    }

    pub fn batch(mesh: Mesh<G>) -> BatchMutation<'a, G> {
        BatchMutation::new(Mutation {
            mesh: Mode::Batch(mesh),
        })
    }

    pub fn replace(mesh: &'a mut Mesh<G>, replacement: Mesh<G>) -> ReplaceMutation<'a, G> {
        ReplaceMutation::new(mesh, replacement)
    }

    pub fn as_mesh(&self) -> &Mesh<G> {
        self.mesh.get()
    }

    pub fn as_mesh_mut(&mut self) -> &mut Mesh<G> {
        self.mesh.get_mut()
    }
}

/// Vertex mutations.
impl<'a, G> Mutation<'a, G>
where
    G: 'a + Geometry,
{
    pub fn insert_vertex(&mut self, geometry: G::Vertex) -> VertexKey {
        self.mesh
            .vertices
            .insert_with_generator(Vertex::new(geometry))
    }
}

/// Edge mutations.
impl<'a, G> Mutation<'a, G>
where
    G: 'a + Geometry,
{
    pub fn insert_edge(
        &mut self,
        vertices: (VertexKey, VertexKey),
        geometry: G::Edge,
    ) -> Result<EdgeKey, Error> {
        let mesh = self.mesh.get_mut();
        let (a, b) = vertices;
        let ab = (a, b).into();
        let ba = (b, a).into();
        // If the edge already exists, then fail. This ensures an important
        // invariant: edges may only have two adjacent faces. That is, a
        // half-edge may only have one associated face, at most one preceding
        // half-edge, at most one following half-edge, and may form at most one
        // closed loop.
        if mesh.edges.contains_key(&ab) {
            return Err(GraphError::TopologyConflict.into());
        }
        let vertex = {
            if !mesh.vertices.contains_key(&b) {
                return Err(GraphError::TopologyNotFound.into());
            }
            match mesh.vertices.get_mut(&a) {
                Some(vertex) => vertex,
                _ => {
                    return Err(GraphError::TopologyNotFound.into());
                }
            }
        };
        let mut edge = Edge::new(b, geometry);
        // This is the point of no return. The mesh has been mutated. Unwrap
        // results.
        if let Some(opposite) = mesh.edges.get_mut(&ba) {
            edge.opposite = Some(ba);
            opposite.opposite = Some(ab);
        }
        mesh.edges.insert_with_key(&ab, edge);
        vertex.edge = Some(ab);
        Ok(ab)
    }

    fn get_or_insert_edge(
        &mut self,
        vertices: (VertexKey, VertexKey),
        geometry: G::Edge,
    ) -> Result<EdgeKey, Error> {
        self.insert_edge(vertices, geometry).or_else(|error| {
            match error.downcast::<GraphError>().unwrap() {
                GraphError::TopologyConflict => Ok(vertices.into()),
                error => Err(error.into()),
            }
        })
    }

    fn get_or_insert_composite_edge(
        &mut self,
        vertices: (VertexKey, VertexKey),
        geometry: G::Edge,
    ) -> Result<(EdgeKey, EdgeKey), Error> {
        let (a, b) = vertices;
        let ab = self.get_or_insert_edge((a, b), geometry.clone())?;
        let ba = self.get_or_insert_edge((b, a), geometry)?;
        Ok((ab, ba))
    }

    pub fn remove_edge(&mut self, edge: EdgeKey) -> Result<Edge<G>, Error> {
        if let Some(mut edge) = self.mesh.edge_mut(edge) {
            if let Some(mut next) = edge.raw_next_edge_mut() {
                next.previous = None;
            }
            if let Some(mut previous) = edge.raw_previous_edge_mut() {
                previous.next = None;
            }
            edge.source_vertex_mut().edge = None;
        }
        else {
            return Err(Error::from(GraphError::TopologyNotFound));
        }
        Ok(self.mesh.edges.remove(&edge).unwrap())
    }
}

/// Face mutations.
impl<'a, G> Mutation<'a, G>
where
    G: 'a + Geometry,
{
    fn connect_face_interior(&mut self, edges: &[EdgeKey], face: FaceKey) -> Result<(), Error> {
        for (ab, bc) in edges.perimeter() {
            {
                let mut edge = self.mesh.edge_mut(ab).unwrap();
                edge.next = Some(bc);
                edge.face = Some(face);
            }
            {
                let mut edge = self.mesh.edge_mut(bc).unwrap();
                edge.previous = Some(ab);
            }
        }
        Ok(())
    }

    fn connect_face_exterior(
        &mut self,
        edges: &[EdgeKey],
        connectivity: (Connectivity, Connectivity),
    ) -> Result<(), Error> {
        let (incoming, outgoing) = connectivity;
        for (a, b) in edges.iter().map(|edge| edge.to_vertex_keys()) {
            // Only boundary edges must be connected.
            if self.mesh.edge((b, a).into()).unwrap().face().is_none() {
                // The next edge of b-a is the outgoing edge of the destination
                // vertex A that is also a boundary edge or, if there is no
                // such outgoing edge, the next exterior edge of the face. The
                // previous edge is similar.
                let ax = outgoing[&a]
                    .iter()
                    .map(|ax| self.mesh.edge(*ax).unwrap())
                    .find(|edge| edge.face().is_none())
                    .or_else(|| {
                        self.mesh
                            .edge((a, b).into())
                            .unwrap()
                            .into_previous_edge()
                            .into_raw_opposite_edge()
                    })
                    .unwrap()
                    .key();
                {
                    self.mesh.edge_mut((b, a).into()).unwrap().next = Some(ax);
                    self.mesh.edge_mut(ax).unwrap().previous = Some((b, a).into());
                }
                let xb = incoming[&b]
                    .iter()
                    .map(|xb| self.mesh.edge(*xb).unwrap())
                    .find(|edge| edge.face().is_none())
                    .or_else(|| {
                        self.mesh
                            .edge((a, b).into())
                            .unwrap()
                            .into_next_edge()
                            .into_raw_opposite_edge()
                    })
                    .unwrap()
                    .key();
                {
                    self.mesh.edge_mut((b, a).into()).unwrap().previous = Some(xb);
                    self.mesh.edge_mut(xb).unwrap().next = Some((b, a).into());
                }
            }
        }
        Ok(())
    }

    fn disconnect_face_interior(&mut self, face: FaceKey) -> Result<(), Error> {
        for mut edge in self.mesh
            .face_mut(face)
            .ok_or_else(|| Error::from(GraphError::TopologyNotFound))?
            .edges_mut()
        {
            edge.face = None;
        }
        Ok(())
    }
}

/// Immediate mesh mutations.
///
/// This type provides mutations that are only available in immediate mode.
/// Immediate mutations are atomic, and fail immediately if the integrity of
/// the mesh is compromised.
pub struct ImmediateMutation<'a, G>
where
    G: 'a + Geometry,
{
    mutation: Mutation<'a, G>,
}

impl<'a, G> ImmediateMutation<'a, G>
where
    G: 'a + Geometry,
{
    fn new(mutation: Mutation<'a, G>) -> Self {
        ImmediateMutation { mutation }
    }

    pub fn commit(self) -> &'a mut Mesh<G> {
        self.mutation.mesh.take_immediate().unwrap()
    }
}

impl<'a, G> ModalMutation<'a, G> for ImmediateMutation<'a, G>
where
    G: 'a + Geometry,
{
    fn insert_face(
        &mut self,
        vertices: &[VertexKey],
        geometry: (G::Edge, G::Face),
    ) -> Result<FaceKey, Error> {
        // Before mutating the mesh, collect the incoming and outgoing edges
        // for each vertex.
        let ((incoming, outgoing), singularity) =
            self.mesh.region_connectivity(self.mesh.region(vertices)?);
        if singularity.is_some() {
            return Err(GraphError::TopologyMalformed
                .context("non-manifold connectivity")
                .into());
        }
        // Insert composite edges and collect the interior edges. This is the
        // point of no return; the mesh has been mutated. Unwrap results.
        let edges = vertices
            .perimeter()
            .map(|ab| {
                self.get_or_insert_composite_edge(ab, geometry.0.clone())
                    .unwrap()
                    .0
            })
            .collect::<Vec<_>>();
        let face = self.mesh
            .faces
            .insert_with_generator(Face::new(edges[0], geometry.1));
        self.connect_face_interior(&edges, face).unwrap();
        self.connect_face_exterior(&edges, (incoming, outgoing))
            .unwrap();
        Ok(face)
    }

    // TODO: This should participate in consistency checks. Removing a face
    //       could, for example, lead to singularities.
    // TODO: Face removal raises questions about "empty regions". Should a mesh
    //       be allowed to have "holes" in it? If not, what about orphaned
    //       edges? If meshes need not be continuous, then "holes" should
    //       probably be allowed.
    fn remove_face(&mut self, face: FaceKey) -> Result<Face<G>, Error> {
        self.disconnect_face_interior(face)?;
        Ok(self.mesh.faces.remove(&face).unwrap())
    }
}

impl<'a, G> Deref for ImmediateMutation<'a, G>
where
    G: 'a + Geometry,
{
    type Target = Mutation<'a, G>;

    fn deref(&self) -> &Self::Target {
        &self.mutation
    }
}

impl<'a, G> DerefMut for ImmediateMutation<'a, G>
where
    G: 'a + Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.mutation
    }
}

/// Batch mesh mutations.
///
/// This type provides mutations that are only available in batch mode. Batch
/// mutations defer certain types of integrity errors until a series of
/// mutations are complete. When `commit` is called, the integrity of the mesh
/// is verified before yielding it to the caller.
pub struct BatchMutation<'a, G>
where
    G: 'a + Geometry,
{
    mutation: Mutation<'a, G>,
    singularities: HashMap<VertexKey, HashSet<FaceKey>>,
}

impl<'a, G> BatchMutation<'a, G>
where
    G: 'a + Geometry,
{
    fn new(mutation: Mutation<'a, G>) -> Self {
        BatchMutation {
            mutation: mutation,
            singularities: HashMap::new(),
        }
    }

    pub fn commit(self) -> Result<Mesh<G>, Error> {
        let mesh = self.mutation.mesh.take_batch().unwrap();
        for (vertex, faces) in self.singularities {
            // TODO: This will not detect exactly two faces joined by a single
            //       vertex. This is technically supported, but perhaps should
            //       be rejected.
            // Determine if any unreachable faces exist in the mesh. This
            // cannot happen if the mesh is ultimately a manifold and edge
            // connectivity heals.
            if let Some(vertex) = mesh.vertex(vertex) {
                for unreachable in
                    faces.difference(&vertex.faces().map(|face| face.key()).collect())
                {
                    if mesh.face(*unreachable).is_some() {
                        return Err(GraphError::TopologyMalformed
                            .context("non-manifold connectivity")
                            .into());
                    }
                }
            }
        }
        Ok(mesh)
    }
}

impl<'a, G> ModalMutation<'a, G> for BatchMutation<'a, G>
where
    G: 'a + Geometry,
{
    fn insert_face(
        &mut self,
        vertices: &[VertexKey],
        geometry: (G::Edge, G::Face),
    ) -> Result<FaceKey, Error> {
        // Before mutating the mesh, collect the incoming and outgoing edges
        // for each vertex.
        let ((incoming, outgoing), singularity) =
            self.mesh.region_connectivity(self.mesh.region(vertices)?);
        // Insert composite edges and collect the interior edges. This is the
        // point of no return; the mesh has been mutated. Unwrap results.
        let edges = vertices
            .perimeter()
            .map(|ab| {
                self.get_or_insert_composite_edge(ab, geometry.0.clone())
                    .unwrap()
                    .0
            })
            .collect::<Vec<_>>();
        let face = self.mesh
            .faces
            .insert_with_generator(Face::new(edges[0], geometry.1));
        if let Some(singularity) = singularity {
            let faces = self.singularities
                .entry(singularity.0)
                .or_insert_with(Default::default);
            for face in singularity.1 {
                faces.insert(face);
            }
            faces.insert(face);
        }
        self.connect_face_interior(&edges, face).unwrap();
        self.connect_face_exterior(&edges, (incoming, outgoing))
            .unwrap();
        Ok(face)
    }

    // TODO: This should participate in consistency checks. Removing a face
    //       could, for example, lead to singularities.
    fn remove_face(&mut self, face: FaceKey) -> Result<Face<G>, Error> {
        self.disconnect_face_interior(face)?;
        Ok(self.mesh.faces.remove(&face).unwrap())
    }
}

impl<'a, G> Deref for BatchMutation<'a, G>
where
    G: 'a + Geometry,
{
    type Target = Mutation<'a, G>;

    fn deref(&self) -> &Self::Target {
        &self.mutation
    }
}

impl<'a, G> DerefMut for BatchMutation<'a, G>
where
    G: 'a + Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.mutation
    }
}

pub struct ReplaceMutation<'a, G>
where
    G: 'a + Geometry,
{
    mesh: &'a mut Mesh<G>,
    mutation: BatchMutation<'a, G>,
}

impl<'a, G> ReplaceMutation<'a, G>
where
    G: 'a + Geometry,
{
    fn new(mesh: &'a mut Mesh<G>, replacement: Mesh<G>) -> Self {
        let mutant = mem::replace(mesh, replacement);
        ReplaceMutation {
            mesh: mesh,
            mutation: Mutation::batch(mutant),
        }
    }

    pub fn commit(self) -> Result<&'a mut Mesh<G>, Error> {
        let ReplaceMutation { mesh, mutation } = self;
        mem::replace(mesh, mutation.commit()?);
        Ok(mesh)
    }
}

impl<'a, G> Deref for ReplaceMutation<'a, G>
where
    G: 'a + Geometry,
{
    type Target = BatchMutation<'a, G>;

    fn deref(&self) -> &Self::Target {
        &self.mutation
    }
}

impl<'a, G> DerefMut for ReplaceMutation<'a, G>
where
    G: 'a + Geometry,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.mutation
    }
}
