use std::fmt::Debug;
use std::mem;

pub trait Transact<T = ()>: Sized {
    type Output;
    type Error: Debug;

    fn commit(self) -> Result<Self::Output, Self::Error>;

    fn commit_with<F, U, E>(mut self, f: F) -> Result<(Self::Output, U), Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<U, E>,
        E: Into<Self::Error>,
    {
        match f(&mut self) {
            Ok(value) => self.commit().map(|output| (output, value)),
            Err(error) => {
                self.abort();
                Err(error.into())
            }
        }
    }

    fn abort(self) {}
}

pub trait TransactFrom<T>: From<T> + Transact<T> {}

impl<T, U> TransactFrom<U> for T where T: From<U> + Transact<U> {}

pub trait Mutate<T>: Transact<T, Output = T> {
    fn replace(target: &mut T, replacement: T) -> Replace<T, Self>
    where
        Self: TransactFrom<T>,
    {
        Replace::replace(target, replacement)
    }
}

impl<T, U> Mutate<U> for T where T: Transact<U, Output = U> {}

pub trait ClosedInput: Transact<<Self as ClosedInput>::Input> {
    type Input;
}

// TODO: This type definition is part of the public API of this module and may
//       be useful if `transact` is further integrated.
#[allow(dead_code)]
pub type ClosedOutput<T> = <T as Transact<<T as ClosedInput>::Input>>::Output;

trait Drain<T> {
    fn as_option_mut(&mut self) -> &mut Option<T>;

    fn drain(&mut self) -> T {
        self.as_option_mut().take().expect("drained")
    }

    fn undrain(&mut self, value: T) {
        let drained = self.as_option_mut();
        if drained.is_some() {
            panic!("undrained");
        }
        else {
            *drained = Some(value);
        }
    }

    fn try_swap_or<F, U, E>(&mut self, value: T, mut f: F) -> Result<U, E>
    where
        F: FnMut(T) -> Result<(T, U), E>,
    {
        match f(self.drain()) {
            Ok((value, output)) => {
                self.undrain(value);
                Ok(output)
            }
            Err(error) => {
                self.undrain(value);
                Err(error)
            }
        }
    }
}

pub struct Replace<'a, T, M>
where
    M: Mutate<T> + TransactFrom<T>,
{
    inner: Option<(&'a mut T, M)>,
}

impl<'a, T, M> Replace<'a, T, M>
where
    M: Mutate<T> + TransactFrom<T>,
{
    pub fn replace(target: &'a mut T, replacement: T) -> Self {
        let mutant = mem::replace(target, replacement);
        Replace {
            inner: Some((target, M::from(mutant))),
        }
    }

    fn drain_and_commit(&mut self) -> Result<&'a mut T, <Self as Transact<&'a mut T>>::Error> {
        let (target, inner) = self.drain();
        let mutant = inner.commit()?;
        *target = mutant;
        Ok(target)
    }

    fn drain_and_abort(&mut self) {
        let (_, inner) = self.drain();
        inner.abort();
    }
}

impl<'a, T, M> AsRef<M> for Replace<'a, T, M>
where
    M: Mutate<T> + TransactFrom<T>,
{
    fn as_ref(&self) -> &M {
        &self.inner.as_ref().unwrap().1
    }
}

impl<'a, T, M> AsMut<M> for Replace<'a, T, M>
where
    M: Mutate<T> + TransactFrom<T>,
{
    fn as_mut(&mut self) -> &mut M {
        &mut self.inner.as_mut().unwrap().1
    }
}

impl<'a, T, M> Drain<(&'a mut T, M)> for Replace<'a, T, M>
where
    M: Mutate<T> + TransactFrom<T>,
{
    fn as_option_mut(&mut self) -> &mut Option<(&'a mut T, M)> {
        &mut self.inner
    }
}

impl<'a, T, M> Drop for Replace<'a, T, M>
where
    M: Mutate<T> + TransactFrom<T>,
{
    fn drop(&mut self) {
        self.drain_and_abort()
    }
}

impl<'a, T, M> From<&'a mut T> for Replace<'a, T, M>
where
    T: Default,
    M: Mutate<T> + TransactFrom<T>,
{
    fn from(target: &'a mut T) -> Self {
        Self::replace(target, Default::default())
    }
}

impl<'a, T, M> Transact<&'a mut T> for Replace<'a, T, M>
where
    M: Mutate<T> + TransactFrom<T>,
{
    type Output = &'a mut T;
    type Error = <M as Transact<T>>::Error;

    fn commit(mut self) -> Result<Self::Output, Self::Error> {
        let mutant = self.drain_and_commit();
        mem::forget(self);
        mutant
    }

    fn abort(mut self) {
        self.drain_and_abort();
        mem::forget(self);
    }
}
