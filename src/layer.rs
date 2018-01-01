
extern crate linear_algebra;

use self::linear_algebra::vector::Vector;
use self::linear_algebra::matrix::Matrix;

pub struct Layer<T> {
    pub biases: Vector<T>,
    pub weights: Matrix<T>
}

impl<T> Layer<T> {
    pub fn new<F: FnMut() -> f64>(distribution: &mut F, input_count: usize, node_count: usize) -> Layer<f64> {
        let factor = 1.0 / (input_count as f64).sqrt();

        let biases = Vector::generate(
            |_| distribution(),
            node_count
        );

        let weights = Matrix::generate(
            |_|
                factor * distribution(),
            input_count, node_count
        );

        Layer{
            biases,
            weights
        }
    }
}