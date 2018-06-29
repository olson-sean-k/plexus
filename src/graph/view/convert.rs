pub trait FromKeyedSource<T>: Sized {
    fn from_keyed_source(source: T) -> Option<Self>;
}

pub trait IntoView<T>: Sized {
    fn into_view(self) -> Option<T>;
}

impl<T, U> IntoView<U> for T
where
    T: Sized,
    U: FromKeyedSource<T>,
{
    fn into_view(self) -> Option<U> {
        U::from_keyed_source(self)
    }
}
