
extern crate rand;

use self::rand::distributions::{ Normal, IndependentSample };

use layer::Layer;

pub struct Network<T> {
    layers: Vec<Layer<T>>
}

impl Network<f64> {
    pub fn new(layer_sizes: &[usize]) -> Network<f64> {
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
}