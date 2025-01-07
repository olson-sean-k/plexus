pub trait Reborrow {
    type Target;

    fn reborrow(&self) -> &Self::Target;
}

pub trait ReborrowMut: Reborrow {
    fn reborrow_mut(&mut self) -> &mut Self::Target;
}

pub trait ReborrowInto<'a>: Reborrow {
    fn reborrow_into(self) -> &'a Self::Target;
}

impl<T> Reborrow for &'_ T {
    type Target = T;

    fn reborrow(&self) -> &Self::Target {
        self
    }
}

impl<T> Reborrow for &'_ mut T {
    type Target = T;

    fn reborrow(&self) -> &Self::Target {
        self
    }
}

impl<T> ReborrowMut for &'_ mut T {
    fn reborrow_mut(&mut self) -> &mut Self::Target {
        self
    }
}

impl<'a, T> ReborrowInto<'a> for &'a T {
    fn reborrow_into(self) -> &'a Self::Target {
        self
    }
}

impl<'a, T> ReborrowInto<'a> for &'a mut T {
    fn reborrow_into(self) -> &'a Self::Target {
        &*self
    }
}
