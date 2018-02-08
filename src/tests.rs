extern crate linear_algebra;


type TestType = f32;
use traits::{ NetworkParameter, RealParameter };

use network::Sample;

use network::Network;
use self::linear_algebra::vector::*;

//

#[test]
fn test_validate() {
    let a = Vector::new(1.00001f32, 1);
    let b = Vector::new(0.00001f32, 1);
    let c = Vector::new(0.00001f32, 1);

    assert_eq!(0.0, Network::validate_sample_helper(a, &b));
    assert_eq!(1.0, Network::validate_sample_helper(b, &c));
}

#[test]
fn test_simple_or() {
    let n = example_or::<TestType>();
    assert_simple_or(&n);
}

#[test]
fn test_save_open() {
    let path = "network.tmp";

    let n = example_or::<TestType>();
    assert_simple_or(&n);
    n.save(path).unwrap();

    let n = unsafe { Network::<TestType>::open(path).unwrap() };
    assert_simple_or(&n);

    ::std::fs::remove_file(path).unwrap();
}


/// Simple example usage of neural network for calculate OR on 3 inputs
fn example_or<T>() -> Network<T>
    where T:
    RealParameter +
    ::std::ops::Div<T, Output=T> +
    ::std::fmt::Display +
    ::std::cmp::PartialOrd +
    ::std::iter::Sum +
    ::std::ops::Neg<Output=T>
{
    let mut n = Network::new(&[
        3, 1
    ]);
    let _0: T = NetworkParameter::zero();
    let _1: T = NetworkParameter::one();

    let mut samples = vec![
        Sample {
            in_data: Vector::from_vec(vec![_0, _0, _0]),
            expected_result: Vector::from_vec(vec![_0])
        },
        Sample {
            in_data: Vector::from_vec(vec![_1, _1, _1]),
            expected_result: Vector::from_vec(vec![ _1 ])
        },
        Sample {
            in_data: Vector::from_vec(vec![_0, _1, _0]),
            expected_result: Vector::from_vec(vec![_1]),
        },
        Sample {
            in_data: Vector::from_vec(vec![_0, _1, _1]),
            expected_result: Vector::from_vec(vec![ _1 ]),
        },
        Sample {
            in_data: Vector::from_vec(vec![_0, _0, _0]),
            expected_result: Vector::from_vec(vec![ _0 ]),
        },
        Sample {
            in_data: Vector::from_vec(vec![_1, _0, _0]),
            expected_result: Vector::from_vec(vec![ _1 ]),
        },
        Sample {
            in_data: Vector::from_vec(vec![_0, _0, _1]),
            expected_result: Vector::from_vec(vec![ _1 ]),
        },
        Sample {
            in_data: Vector::from_vec(vec![_1, _1, _0]),
            expected_result: Vector::from_vec(vec![ _1 ])
        }
    ];

    let validation_samples = samples.clone();

    n.adam(
        T::from_f64(0.5), T::from_f64(0.9), T::from_f64(0.999), T::from_f64(1.0e-8),
        20, 8,
        &mut samples,
        |n, _|{
        let (avg, min, max) = n.validate(&validation_samples);
        println!("avg: {}, min: {}, max: {}", avg, min, max)
    });
    n
}

fn assert_simple_or<T>(n: &Network<T>)
    where T:
        RealParameter +
        ::std::ops::Div<T, Output=T> +
        ::std::ops::Neg<Output=T> +
        PartialOrd +
        ::std::iter::Sum
{
    let _0: T = NetworkParameter::zero();
    let _1: T = NetworkParameter::one();

    let low = |res: T| res < T::from_f64(0.1);
    let hi = |res: T| !(res < T::from_f64(0.1));

    test(vec![_0, _0, _0], n, &low);
    test(vec![_0, _0, _1], n, &hi);
    test(vec![_0, _1, _0], n, &hi);
    test(vec![_0, _1, _1], n, &hi);
    test(vec![_1, _0, _0], n, &hi);
    test(vec![_1, _0, _1], n, &hi);
    test(vec![_1, _1, _0], n, &hi);
    test(vec![_1, _1, _1], n, &hi);

    let validation_samples = vec![
        Sample {
            in_data: Vector::from_vec(vec![_0, _0, _0]),
            expected_result: Vector::from_vec(vec![_0])
        },
        Sample {
            in_data: Vector::from_vec(vec![_0, _0, _1]),
            expected_result: Vector::from_vec(vec![ _1 ])
        },
        Sample {
            in_data: Vector::from_vec(vec![_0, _1, _0]),
            expected_result: Vector::from_vec(vec![_1]),
        },
        Sample {
            in_data: Vector::from_vec(vec![_0, _1, _1]),
            expected_result: Vector::from_vec(vec![ _1 ]),
        },
        Sample {
            in_data: Vector::from_vec(vec![_1, _0, _0]),
            expected_result: Vector::from_vec(vec![ _1 ]),
        },
        Sample {
            in_data: Vector::from_vec(vec![_1, _0, _1]),
            expected_result: Vector::from_vec(vec![ _1 ]),
        },
        Sample {
            in_data: Vector::from_vec(vec![_1, _1, _0]),
            expected_result: Vector::from_vec(vec![ _1 ])
        },
        Sample {
            in_data: Vector::from_vec(vec ! [_1, _1, _1]),
            expected_result: Vector::from_vec(vec! [ _1 ])
        }
    ];

    let (avg, min, max) = n.validate(&validation_samples);
    assert_eq!(min, _1);
    assert_eq!(max, _1);
    assert!(avg > T::from_f64(0.99));
}


fn test<F: Fn(T) -> bool, T: NetworkParameter + PartialOrd + ::std::iter::Sum>(input: Vec<T>, n: &Network<T>, p: &F) {
    let res = n.feed_forward(Vector::from_vec(input.clone())).to_vec()[0];
    assert!(p(res), "{:?} -> {:?}", input, res);
}