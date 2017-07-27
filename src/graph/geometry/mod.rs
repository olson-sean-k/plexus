use graph::mesh::Mesh;
use graph::topology::Face;
use graph::storage::Key;

pub trait Attribute: Default {}

impl Attribute for () {}

pub trait Geometry: Sized {
    type VertexData: Attribute;
    type EdgeData: Attribute;
    type FaceData: Attribute;

    fn compute_vertex_data<M, K>()
    where
        M: AsRef<Mesh<Self, K>>,
        K: Key,
    {
    }

    fn compute_edge_data<M, K>()
    where
        M: AsRef<Mesh<Self, K>>,
        K: Key,
    {
    }

    fn compute_face_data<M, K>(face: &mut Face<M, Self, K>)
    where
        M: AsRef<Mesh<Self, K>>,
        K: Key,
    {
    }
}

pub struct EmptyGeometry;

impl Geometry for EmptyGeometry {
    type VertexData = ();
    type EdgeData = ();
    type FaceData = ();
}
