
extern crate linear_algebra;

use ::std::ops::{Add, Sub, Mul};
use ::std::ops::{AddAssign, SubAssign, MulAssign};

pub trait NetworkParameter: 
	Parameter +
	Mul<Self, Output=Self> +
	Add<Self, Output=Self> +
	Sub<Self, Output=Self> +

	MulAssign<Self> +
	AddAssign<Self> +
	SubAssign<Self>
{}

pub trait Parameter: self::linear_algebra::traits::Parameter {
    fn from_f64(x: f64) -> Self;
    fn zero() -> Self;
	fn one() -> Self;
}

// Implements an unsafe trait for a list of types.
macro_rules! impl_from_f64 {
    ($( $ty:ident ),+) => {
        $( impl Parameter for $ty {
            fn from_f64(x: f64) -> $ty { x as $ty }
            fn zero() -> $ty { 0 as $ty }
			fn one() -> $ty { 1 as $ty }
        } 
		impl NetworkParameter for $ty {}
		)+
    }
}

impl_from_f64!(u8, i8, u16, i16, u32, i32, u64, i64, usize, isize, f32, f64);