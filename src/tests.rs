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


    let s = Network::validate_sample_helper(a, &b);
    println!("\n----------------");
    println!("Res: {}", s);
    println!("----------------");
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

    let samples = vec![
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

    for _ in 0..68 {
        n.learn(T::from_f64(0.5), &samples);
        let (avg, min, max) = n.validate(&validation_samples);

        println!("------------------------");
        println!("\tAvg: {}", avg);
        println!("\tmin: {}, max: {}", min, max);
    }
    n
}

fn assert_simple_or<T: NetworkParameter + PartialOrd + ::std::iter::Sum>(n: &Network<T>) {
    let _0 = T::zero();
    let _1 = T::one();

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
}


fn test<F: Fn(T) -> bool, T: NetworkParameter + PartialOrd + ::std::iter::Sum>(input: Vec<T>, n: &Network<T>, p: &F) {
    let res = n.feed_forward(Vector::from_vec(input.clone())).to_vec()[0];
    assert!(p(res), "{:?} -> {:?}", input, res);
}