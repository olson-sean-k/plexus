pub trait Reborrow {
    type Target;

    fn reborrow(&self) -> &Self::Target;
}

pub trait ReborrowMut: Reborrow {
    fn reborrow_mut(&mut self) -> &mut Self::Target;
}

impl<'a, T> Reborrow for &'a T {
    type Target = T;

    fn reborrow(&self) -> &Self::Target {
        *self
    }
}

impl<'a, T> Reborrow for &'a mut T {
    type Target = T;

    fn reborrow(&self) -> &Self::Target {
        &**self
    }
}

impl<'a, T> ReborrowMut for &'a mut T {
    fn reborrow_mut(&mut self) -> &mut Self::Target {
        *self
    }
}
