
extern crate linear_algebra;

use ::std::ops::{ Add, Sub, Mul };
use ::std::ops::{AddAssign, SubAssign, MulAssign};


pub trait RealParameter:
	NetworkParameter + self::linear_algebra::traits::Real
{}

pub trait NetworkParameter:
	self::linear_algebra::traits::Parameter +
	Mul<Self, Output=Self> +
	Add<Self, Output=Self> +
	Sub<Self, Output=Self> +

	MulAssign<Self> +
	AddAssign<Self> +
	SubAssign<Self>
{
    fn from_f64(x: f64) -> Self;
	fn from_usize(x: usize) -> Self;
    fn zero() -> Self;
	fn one() -> Self;
}

macro_rules! impl_parameter {
    ($( $ty:ident ),+) => {
        $( impl NetworkParameter for $ty {
            fn from_f64(x: f64) -> $ty { x as $ty }
            fn from_usize(x: usize) -> $ty { x as $ty }
            fn zero() -> $ty { 0 as $ty }
			fn one() -> $ty { 1 as $ty }
        }
		)+
    }
}

impl_parameter!(u8, i8, u16, i16, u32, i32, u64, i64, usize, isize, f32, f64);

macro_rules! impl_real_parameter {
    ($( $ty:ident ),+) => {
        $( impl RealParameter for $ty {} )+
    }
}

impl_real_parameter!(f32, f64);