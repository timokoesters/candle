use std::ops::{Deref, DerefMut, FromResidual, Try};

use crate::{Error, Result, Tensor};

macro_rules! ttry {
    ($v:expr) => {
        match &$v.inner {
            Ok(x) => x,
            Err(e) => {
                let Error::WithBacktrace { inner, backtrace } = e else {
                    unreachable!("We made sure to do a backtrace")
                };
                let clone = Error::WithBacktrace {
                    inner: inner.clone(),
                    backtrace: backtrace.clone(),
                };
                return MTensor::from(Err(clone.bt()));
            }
        }
    };
}
#[macro_export]
macro_rules! mttry {
    ($v:expr) => {
        match $v {
            Ok(x) => x,
            Err(e) => return MTensor::new(Err(e)),
        }
    };
}

pub struct MTensor {
    pub inner: Result<Tensor>,
}

impl MTensor {
    // Reimplement some functions that take ownership
    // e.g. you can do mtensor.unwrap() instead of mtensor.inner.unwrap():

    pub fn unwrap(self) -> Tensor {
        self.inner.as_ref().unwrap().clone()
    }
}

impl From<Result<Tensor>> for MTensor {
    fn from(value: Result<Tensor>) -> Self {
        MTensor { inner: value }
    }
}

impl From<Tensor> for MTensor {
    fn from(value: Tensor) -> Self {
        MTensor { inner: Ok(value) }
    }
}

impl Deref for MTensor {
    type Target = Result<Tensor>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for MTensor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl Try for MTensor {
    type Output = Tensor;
    type Residual = Result<std::convert::Infallible>;

    fn from_output(output: Self::Output) -> Self {
        output.into()
    }

    fn branch(self) -> std::ops::ControlFlow<Self::Residual, Self::Output> {
        match self.inner {
            Ok(v) => std::ops::ControlFlow::Continue(v),
            Err(e) => std::ops::ControlFlow::Break(Err(e)),
        }
    }
}
impl FromResidual for MTensor {
    fn from_residual(residual: <Self as std::ops::Try>::Residual) -> Self {
        match residual {
            Err(e) => Err(e).into(),
        }
    }
}

macro_rules! bin_trait {
    ($trait:ident, $a:ident, $fn:ident, $b:ident, $op:expr) => {
        impl std::ops::$trait<$b> for $a {
            type Output = MTensor;

            fn $fn(self, rhs: $b) -> Self::Output {
                $op(&self, &rhs)
            }
        }
        impl std::ops::$trait<$b> for &$a {
            type Output = MTensor;

            fn $fn(self, rhs: $b) -> Self::Output {
                $op(self, &rhs)
            }
        }
        impl std::ops::$trait<&$b> for $a {
            type Output = MTensor;

            fn $fn(self, rhs: &$b) -> Self::Output {
                $op(&self, &rhs)
            }
        }

        impl std::ops::$trait<&$b> for &$a {
            type Output = MTensor;

            fn $fn(self, rhs: &$b) -> Self::Output {
                $op(self, rhs)
            }
        }
    };
}

// FOR (MTensor, MTensor)

bin_trait!(Add, MTensor, add, MTensor, |a: &MTensor, b: &MTensor| {
    ttry!(a) + ttry!(b)
});
bin_trait!(Sub, MTensor, sub, MTensor, |a: &MTensor, b: &MTensor| {
    ttry!(a) - ttry!(b)
});
bin_trait!(Mul, MTensor, mul, MTensor, |a: &MTensor, b: &MTensor| {
    ttry!(a) * ttry!(b)
});
bin_trait!(Div, MTensor, div, MTensor, |a: &MTensor, b: &MTensor| {
    ttry!(a) / ttry!(b)
});

// FOR (MTensor, F64)

bin_trait!(Add, MTensor, add, f64, |a: &MTensor, b: &f64| {
    ttry!(a) + *b
});
bin_trait!(Sub, MTensor, sub, f64, |a: &MTensor, b: &f64| {
    ttry!(a) - *b
});
bin_trait!(Mul, MTensor, mul, f64, |a: &MTensor, b: &f64| {
    ttry!(a) * *b
});
bin_trait!(Div, MTensor, div, f64, |a: &MTensor, b: &f64| {
    ttry!(a) / *b
});

// FOR (F64, MTensor)

bin_trait!(Add, f64, add, MTensor, |a: &f64, b: &MTensor| {
    *a + ttry!(b)
});
bin_trait!(Sub, f64, sub, MTensor, |a: &f64, b: &MTensor| {
    *a - ttry!(b)
});
bin_trait!(Mul, f64, mul, MTensor, |a: &f64, b: &MTensor| {
    *a * ttry!(b)
});
bin_trait!(Div, f64, div, MTensor, |a: &f64, b: &MTensor| {
    *a / ttry!(b)
});

// FOR (Tensor, MTensor)

bin_trait!(Add, Tensor, add, MTensor, |a: &Tensor, b: &MTensor| {
    a + ttry!(b)
});
bin_trait!(Sub, Tensor, sub, MTensor, |a: &Tensor, b: &MTensor| {
    a - ttry!(b)
});
bin_trait!(Mul, Tensor, mul, MTensor, |a: &Tensor, b: &MTensor| {
    a * ttry!(b)
});
bin_trait!(Div, Tensor, div, MTensor, |a: &Tensor, b: &MTensor| {
    a / ttry!(b)
});

// FOR (MTensor, Tensor)

bin_trait!(Add, MTensor, add, Tensor, |a: &MTensor, b: &Tensor| {
    ttry!(a) + b
});
bin_trait!(Sub, MTensor, sub, Tensor, |a: &MTensor, b: &Tensor| {
    ttry!(a) - b
});
bin_trait!(Mul, MTensor, mul, Tensor, |a: &MTensor, b: &Tensor| {
    ttry!(a) * b
});
bin_trait!(Div, MTensor, div, Tensor, |a: &MTensor, b: &Tensor| {
    ttry!(a) / b
});

#[cfg(test)]
mod tests {
    use crate::Device;

    use super::*;

    #[test]
    fn test1() {
        let device = Device::Cpu;
        let a: MTensor = Tensor::from_slice(&[1.0, 2.0], (2,), &device);
        let b: MTensor = Tensor::from_slice(&[2.0, 3.0], (2,), &device);
        let c: MTensor = Tensor::from_slice(&[3.0, 4.0], (2,), &device);
        let x: Tensor = Tensor::from_slice(&[3.0, 4.0], (2,), &device)
            .inner
            .unwrap();

        (&a + &b + &c).unwrap();
        (a + &b + &c).unwrap();
        (&b + c).unwrap();
        (3.0 * &x * &x - 4.0 * &x - 5.0).unwrap();
    }
}
