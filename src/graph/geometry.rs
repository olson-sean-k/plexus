use graph::mesh::Mesh;
use graph::topology::Face;

pub trait Attribute: Default {}

pub trait Geometry: Sized {
    type Vertex: Attribute;
    type Edge: Attribute;
    type Face: Attribute;

    fn compute_vertex_geometry<M>()
    where
        M: AsRef<Mesh<Self>>,
    {
    }

    fn compute_edge_geometry<M>()
    where
        M: AsRef<Mesh<Self>>,
    {
    }

    fn compute_face_geometry<M>(face: &mut Face<M, Self>)
    where
        M: AsRef<Mesh<Self>>,
    {
    }
}

impl Attribute for () {}

impl Geometry for () {
    type Vertex = ();
    type Edge = ();
    type Face = ();
}
