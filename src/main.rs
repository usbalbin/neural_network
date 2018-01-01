
extern crate neural_network;
extern crate linear_algebra;

use neural_network::network::*;

use linear_algebra::vector::Vector;


fn main() {
    example_or();
}

/// Simple example usage of neural network for calculate OR on 3 inputs
fn example_or() {
    let mut n = Network::new(&[
        3, 1
    ]);

    loop {
        for _ in 0..10 {
            n.learn(0.5, vec![
                Sample {
                    in_data: Vector::from_vec(vec![0.0, 0.0, 0.0]),
                    expected_result: Vector::from_vec(vec![0.0])
                },
                    Sample {
                        in_data: Vector::from_vec(vec![1.0, 1.0, 1.0]),
                        expected_result: Vector::from_vec(vec![ 1.0 ])
                },
                Sample {
                    in_data: Vector::from_vec(vec![0.0, 1.0, 0.0]),
                    expected_result: Vector::from_vec(vec![1.0]),
                },
                Sample {
                    in_data: Vector::from_vec(vec![0.0, 1.0, 1.0]),
                    expected_result: Vector::from_vec(vec![ 1.0 ]),
                },
                Sample {
                    in_data: Vector::from_vec(vec![0.0, 0.0, 0.0]),
                    expected_result: Vector::from_vec(vec![ 0.0 ]),
                },
                Sample {
                    in_data: Vector::from_vec(vec![1.0, 0.0, 1.0]),
                    expected_result: Vector::from_vec(vec![ 1.0 ]),
                },
                Sample {
                    in_data: Vector::from_vec(vec![1.0, 1.0, 0.0]),
                    expected_result: Vector::from_vec(vec![ 1.0 ])
                }
            ]);
        }
        println!("{} {}", n.feed_forward(Vector::from_vec(vec![0.0, 0.0, 0.0])), n.feed_forward(Vector::from_vec(vec![0.0, 1.0, 0.0])));

        std::thread::sleep(std::time::Duration::from_millis(1000));
    }
}
