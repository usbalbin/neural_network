
extern crate neural_network;
extern crate linear_algebra;

use neural_network::network::*;
use neural_network::traits::NetworkParameter;

use linear_algebra::vector::Vector;


fn main() {
    example_or::<f32>();
}

/// Simple example usage of neural network for calculate OR on 3 inputs
fn example_or<T>() 
	where T: 
	NetworkParameter + 
	std::ops::Div<T, Output=T> +
	std::fmt::Display
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
            in_data: Vector::from_vec(vec![_1, _0, _1]),
            expected_result: Vector::from_vec(vec![ _1 ]),
        },
        Sample {
            in_data: Vector::from_vec(vec![_1, _1, _0]),
            expected_result: Vector::from_vec(vec![ _1 ])
        }
    ];

    loop {
        for _ in 0..10 {
            n.learn(T::from_f64(0.5), &samples);
        }
        println!("{} {}", n.feed_forward(Vector::from_vec(vec![_0, _0, _0])), n.feed_forward(Vector::from_vec(vec![_0, _1, _0])));

        std::thread::sleep(std::time::Duration::from_millis(1000));
    }
}
