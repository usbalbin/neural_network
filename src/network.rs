
extern crate rand;
extern crate linear_algebra;

use self::rand::distributions::{ Normal, IndependentSample };

use self::linear_algebra::vector::*;
use self::linear_algebra::matrix::*;

use layer::Layer;

pub struct Sample<T> {
    pub in_data: Vector<T>,
    pub expected_result: Vector<T>
}

pub struct Network<T> {
    layers: Vec<Layer<T>>
}

impl Network<f64> {
    pub fn new(layer_sizes: &[usize]) -> Network<f64> {
        assert!(layer_sizes.len() > 1);

        let seed: &[_] = &[1, 2, 3, 4];
        let mut random_generator: rand::StdRng = rand::SeedableRng::from_seed(seed);
        let normal_distribute = Normal::new(0.0, 1.0);
        let mut distribution = || normal_distribute.ind_sample(&mut random_generator);

        let mut layers = Vec::with_capacity(layer_sizes.len());
        for data in layer_sizes.windows(2) {
            let (input_count, node_count) = (data[0], data[1]);
            layers.push(Layer::<f64>::new(&mut distribution, input_count, node_count));
        }

        Network {
            layers
        }
    }

    pub fn feed_forward(&self, input: Vector<f64>) -> Vector<f64> {
        assert_eq!(input.len(), self.layers[0].weights.get_row_count());

        let mut data = input;
        for layer in self.layers.iter() {
            data = activation_func_in_place(&(&data * &layer.weights) + &layer.biases);
        }

        data
    }

    fn back_propogate(&self, sample: Sample<f64>, gradients_weights: &mut Vec<Matrix<f64>>, gradients_biases: &mut Vec<Vector<f64>>) {
        assert_eq!(gradients_biases.len(), self.layers.len());
        assert_eq!(gradients_weights.len(), self.layers.len());

        let mut raw_data = Vec::new();
        let mut activations = Vec::new();

        //Feed forward
        {
            let ref mut first_layer = self.layers.first().unwrap();

            let raw = &(&sample.in_data * &first_layer.weights) + &first_layer.biases;

            activations.push(
                activation_func(&raw)
            );

            raw_data.push(raw);

            for layer in self.layers.iter().skip(1) {
                let raw = &(activations.last().unwrap() * &layer.weights) + &layer.biases;

                activations.push(
                    activation_func(&raw)
                );

                raw_data.push(raw);
            }
        }


        //Work backwards to calculate deltas
        {

            let mut delta;

            {
                let ref out = activations[activations.len() - 1];
                let in_ = if activations.len() > 1 {
                    &activations[activations.len() - 2]
                } else {
                    &sample.in_data
                };

                delta = out - &sample.expected_result;//Cross entropy cost
                //delta = (out - sample.expectedResult) * activationFuncPrime(rawData.back());//Quadratic cost

                *gradients_biases.last_mut().unwrap() += &delta;
                *gradients_weights.last_mut().unwrap() += &mul_column_row(in_, &delta);
            }

            if self.layers.len() > 2 {
                for i in (1..(self.layers.len() - 1)).rev() {
                    let ref in_ = activations[i - 1];

                    delta = &mul_transpose_mat(&delta, &self.layers[i + 1].weights) * &activation_func_prime(&raw_data[i]);

                    gradients_biases[i] += &delta;
                    gradients_weights[i] += &mul_column_row(in_, &delta);
                }
            }

            if activations.len() > 1 {
                let ref in_ = sample.in_data;

                delta = &mul_transpose_mat(&delta, &self.layers[1].weights) * &activation_func_prime(&raw_data[0]);
                gradients_biases[0] += &delta;
                gradients_weights[0] += &mul_column_row(in_, &delta);//TODO: Check if "in" and "delta" should be swapped
            }
        }
    }

    fn apply_gradients(&mut self, learning_rate: f64, gradients_weights: Vec<Matrix<f64>>, gradients_biases: Vec<Vector<f64>>) {
        for (layer, gradient_bias) in self.layers.iter_mut().zip(gradients_biases.iter()) {
            layer.biases -= &(gradient_bias * learning_rate);
        }

        for (layer, gradient_weight) in self.layers.iter_mut().zip(gradients_weights.iter()) {
            layer.weights -= &(gradient_weight * learning_rate);
        }
    }

    pub fn learn(&mut self, learning_rate: f64, samples: Vec<Sample<f64>>) {
        let mut gradients_biases = Vec::with_capacity(self.layers.len());
        let mut gradients_weights = Vec::with_capacity(self.layers.len());


        for i in 0..self.layers.len() {
            gradients_biases.push(Vector::new(0.0, self.layers[i].biases.len()));
            gradients_weights.push(Matrix::new(0.0,
                self.layers[i].weights.get_row_count(),
                self.layers[i].weights.get_col_count()
            ));
        }

        for sample in samples {
            self.back_propogate(sample, &mut gradients_weights, &mut gradients_biases);
        }

        self.apply_gradients(learning_rate, gradients_weights, gradients_biases);
    }
}




fn activation_func_in_place(mut data: Vector<f64>) -> Vector<f64> {
    data.map_mut(
        |e: &mut f64|
            *e = sigmoid(*e)
    );
    data
}

fn activation_func(data: &Vector<f64>) -> Vector<f64> {
    data.map(
        |e: &f64|
            sigmoid(*e)
    )
}

fn activation_func_prime(data: &Vector<f64>) -> Vector<f64> {
    data.map(
        |e: &f64|
            sigmoid_prime(*e)
    )
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_prime(x: f64) -> f64 {
    let sig = sigmoid(x);
    sig * (1.0 - sig)
}