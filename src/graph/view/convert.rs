pub trait FromKeyedSource<T> {
    fn from_keyed_source(source: T) -> Self;
}

pub trait IntoView<T> {
    fn into_view(self) -> T;
}

impl<T, U> IntoView<U> for T
where
    U: FromKeyedSource<T>,
{
    fn into_view(self) -> U {
        U::from_keyed_source(self)
    }
}
