use graph::topology::FaceMut;

pub trait Attribute: Default {}

pub trait Geometry: Sized {
    type Vertex: Attribute;
    type Edge: Attribute;
    type Face: Attribute;

    fn compute_vertex_geometry() {}

    fn compute_edge_geometry() {}

    fn compute_face_geometry(_: FaceMut<Self>) {}
}

impl Attribute for () {}

impl Geometry for () {
    type Vertex = ();
    type Edge = ();
    type Face = ();
}
