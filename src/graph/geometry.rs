use graph::topology::FaceMut;

pub trait FromGeometry<T> {
    fn from_geometry(_: T) -> Self;
}

pub trait IntoGeometry<T> {
    fn into_geometry(self) -> T;
}

impl<T, U> IntoGeometry<U> for T
where
    U: FromGeometry<T>,
{
    fn into_geometry(self) -> U {
        U::from_geometry(self)
    }
}

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

pub trait Normalize {
    fn normalize(self) -> Self;
}

pub trait Cross<T = Self> {
    type Output;

    fn cross(self, other: T) -> Self::Output;
}

pub trait AsPosition {
    type Target;

    fn as_position(&self) -> &Self::Target;
    fn as_position_mut(&mut self) -> &mut Self::Target;
}
