use graph::mesh::Mesh;
use graph::topology::Face;
use graph::storage::AtomicKey;

pub trait Attribute: Default {}

impl Attribute for () {}

pub trait Geometry: Sized {
    type Vertex: Attribute;
    type Edge: Attribute;
    type Face: Attribute;

    fn compute_vertex_geometry<M, K>()
    where
        M: AsRef<Mesh<Self, K>>,
        K: AtomicKey,
    {
    }

    fn compute_edge_geometry<M, K>()
    where
        M: AsRef<Mesh<Self, K>>,
        K: AtomicKey,
    {
    }

    fn compute_face_geometry<M, K>(face: &mut Face<M, Self, K>)
    where
        M: AsRef<Mesh<Self, K>>,
        K: AtomicKey,
    {
    }
}

pub struct EmptyGeometry;

impl Geometry for EmptyGeometry {
    type Vertex = ();
    type Edge = ();
    type Face = ();
}
