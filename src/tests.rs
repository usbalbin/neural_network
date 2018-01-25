extern crate linear_algebra;

use traits::NetworkParameter;

use network::Sample;
use network::Network;
use self::linear_algebra::vector::*;

type TestType = f32;

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
    NetworkParameter +
    ::std::ops::Div<T, Output=T> +
    ::std::fmt::Display +
    ::std::cmp::PartialOrd
{
    let mut n = Network::new(&[
        3, 1
    ]);
    let _0 = T::zero();
    let _1 = T::one();

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

    for _ in 0..70 {
        n.learn(T::from_f64(0.5), &samples);
    }
    n
}

fn assert_simple_or<T: NetworkParameter + PartialOrd>(n: &Network<T>) {
    let _0 = T::zero();
    let _1 = T::one();

    let high = T::from_f64(0.9);
    let low = T::from_f64(0.1);


    assert!(n.feed_forward(Vector::from_vec(vec![_0, _0, _0])).to_vec()[0] < low);
    assert!(!(n.feed_forward(Vector::from_vec(vec![_0, _0, _1])).to_vec()[0] < high));
    assert!(!(n.feed_forward(Vector::from_vec(vec![_0, _1, _0])).to_vec()[0] < high));
    assert!(!(n.feed_forward(Vector::from_vec(vec![_0, _1, _1])).to_vec()[0] < high));
    assert!(!(n.feed_forward(Vector::from_vec(vec![_1, _0, _0])).to_vec()[0] < high));
    assert!(!(n.feed_forward(Vector::from_vec(vec![_1, _0, _1])).to_vec()[0] < high));
    assert!(!(n.feed_forward(Vector::from_vec(vec![_1, _1, _0])).to_vec()[0] < high));
    assert!(!(n.feed_forward(Vector::from_vec(vec![_1, _1, _1])).to_vec()[0] < high));
}