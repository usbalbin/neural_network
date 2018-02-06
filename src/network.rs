
extern crate rand;
extern crate linear_algebra;
extern crate ocl;



use self::rand::distributions::{ Normal, IndependentSample };

use self::linear_algebra::util::*;
use self::linear_algebra::vector::*;
use self::linear_algebra::matrix::*;

use traits::{ NetworkParameter, RealParameter };


use layer::Layer;

use self::linear_algebra::get_cl_data;
use self::linear_algebra::get_work_sizes;

pub struct Sample<T: NetworkParameter> {
    pub in_data: Vector<T>,
    pub expected_result: Vector<T>
}

use ::std::ops::Div;

pub struct Network<T: NetworkParameter> {
    layers: Vec<Layer<T>>
}

///
/// Artificial neural network
///
impl<T: NetworkParameter + ::std::iter::Sum<T>> Network<T> {

    /// Create new network, with layer_sizes.first() inputs, layer_sizes.last() outputs
    /// and layer_sizes[1..] size for every corresponding hidden layer.
    pub fn new(layer_sizes: &[usize]) -> Network<T> {
        assert!(layer_sizes.len() > 1);

        let seed: &[_] = &[1, 2, 3, 4];
        let mut random_generator: rand::StdRng = rand::SeedableRng::from_seed(seed);
        let normal_distribute = Normal::new(0.0, 1.0);
        let mut distribution = || normal_distribute.ind_sample(&mut random_generator);

        let mut layers = Vec::with_capacity(layer_sizes.len());
        for data in layer_sizes.windows(2) {
            let (input_count, node_count) = (data[0], data[1]);
            layers.push(Layer::new(&mut distribution, input_count, node_count));
        }

        Network {
            layers
        }
    }

    /// feed_forward() runs the input through the network returning its results
    pub fn feed_forward(&self, input: Vector<T>) -> Vector<T> {
        assert_eq!(input.len(), self.layers[0].weights.get_row_count());

        let mut data = input;
        for layer in self.layers.iter() {
            data = activation_func_in_place((&data * &layer.weights) + &layer.biases);
        }

        data
    }

    fn back_propagate(&self, sample: &Sample<T>, gradients_weights: &mut Vec<Matrix<T>>, gradients_biases: &mut Vec<Vector<T>>) {
        assert_eq!(gradients_biases.len(), self.layers.len());
        assert_eq!(gradients_weights.len(), self.layers.len());

        let mut raw_data = Vec::new();
        let mut activations = Vec::new();

        //Feed forward
        {
            let ref mut first_layer = self.layers.first().unwrap();

            let raw = (&sample.in_data * &first_layer.weights) + &first_layer.biases;

            activations.push(
                activation_func(&raw)
            );

            raw_data.push(raw);

            for layer in self.layers.iter().skip(1) {
                let raw = (activations.last().unwrap() * &layer.weights) + &layer.biases;

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

    fn apply_gradients(&mut self, learning_rate: T, gradients_weights: Vec<Matrix<T>>, gradients_biases: Vec<Vector<T>>) {
        for (layer, gradient_bias) in self.layers.iter_mut().zip(gradients_biases.into_iter()) {
            layer.biases -= &(gradient_bias * learning_rate);
        }

        for (layer, gradient_weight) in self.layers.iter_mut().zip(gradients_weights.into_iter()) {
            layer.weights -= &(gradient_weight * learning_rate);
        }
    }

    /// learn() trains network with given samples at specified rate.
    /// great care needs to be taken when selecting learning_rate
    pub fn learn(&mut self, learning_rate: T, samples: &Vec<Sample<T>>) {
        let mut gradients_biases = Vec::with_capacity(self.layers.len());
        let mut gradients_weights = Vec::with_capacity(self.layers.len());


        for i in 0..self.layers.len() {
            gradients_biases.push(Vector::new(T::zero(), self.layers[i].biases.len()));
            gradients_weights.push(Matrix::new(T::zero(),
                self.layers[i].weights.get_row_count(),
                self.layers[i].weights.get_col_count()
            ));
        }

        for sample in samples {
            self.back_propagate(sample, &mut gradients_weights, &mut gradients_biases);
        }

        self.apply_gradients(learning_rate, gradients_weights, gradients_biases);
    }

    /// Save Network to specified path.
    /// NOTE! The file will be encoded in the current systems endianness
    ///
    /// ---- Formated like ----
    /// layer_count,
    /// layers[0],
    /// ...
    /// ...
    /// layers[layerCount - 1]
    ///
    pub fn save(&self, path: &str) -> Result<(), ::std::io::Error> {
        use std::fs::File;

        let mut file = File::create(path)?;

        write_u64(&mut file,self.layers.len() as u64)?;

        for layer in self.layers.iter() {
            layer.write_to_file(&mut file)?;
        }
        Ok(())
    }

    /// Open Network from specified path.
    /// NOTE! The file will be interpreted in the current systems endianness
    ///
    /// ---- Formated like ----
    /// layer_count,
    /// layers[0],
    /// ...
    /// ...
    /// layers[layerCount - 1]
    ///
    pub unsafe fn open(path: &str) -> Result<Network<T>, ::std::io::Error> {
        use std::fs::File;

        let mut file = File::open(path)?;

        let layer_count = read_u64(&mut file)?;

        let mut layers = Vec::with_capacity(layer_count as usize);

        for _ in 0..layer_count {
            layers.push(
                Layer::<T>::read_from_file(&mut file)?
            );
        }

        Ok(Network {
            layers
        })
    }

    pub fn get_layer_sizes(&self) -> Vec<usize>{
        let mut res: Vec<usize> = self.layers.iter().map(|layer| layer.biases.len()).collect();
        res.push(self.layers.last().unwrap().weights.get_col_count());
        res
    }
}

impl<T> Network<T>
    where T: RealParameter + ::std::cmp::PartialOrd + ::std::iter::Sum + Div<T, Output=T> + ::std::ops::Neg<Output=T>
{
    /// Returns (avg, min, max)
    pub fn validate(&self, samples: &Vec<Sample<T>>) -> (T, T, T)
    {
        let mut min: T = NetworkParameter::one();
        let mut max: T = NetworkParameter::zero();
        let mut avg: T = NetworkParameter::zero();

        for sample in samples.iter() {
            let val = self.validate_sample(sample);
            if val < min {
                min = val;
            }
            if max < val {
                max = val;
            }
            avg += val;
        }

        let avg = avg / T::from_usize(samples.len());
        (avg, min, max)
    }

    /// Validate how well expected output matches with the computed
    /// Returns number between 0.0 and 1.0, where 1.0 means they match perfectly
    pub fn validate_sample(&self, sample: &Sample<T>) -> T {
        let out = self.feed_forward(sample.in_data.clone());
        let expected_result = &sample.expected_result;

        Self::validate_sample_helper(out, expected_result)
    }

    /// Validate how well two network outputs match
    /// Returns number between 0.0 and 1.0, where 1.0 means they match perfectly
    pub(crate) fn validate_sample_helper(a: Vector<T>, b: &Vector<T>) -> T {
        assert_eq!(a.len(), b.len());

        static mut KERNEL: Option<ocl::Kernel> = None;
        static mut INIT_ONCE: ::std::sync::Once = ::std::sync::ONCE_INIT;

        unsafe {
            INIT_ONCE.call_once(
                || {
                    KERNEL = Some(
                        linear_algebra::create_kernel::<T>(&format!("{}_{}", T::type_to_str(), "validate_sample"))
                            .arg_buf_named::<T, ocl::Buffer<T>>("data", None)
                            .arg_buf_named::<T, ocl::Buffer<T>>("expected_data", None)
                            .arg_buf_named::<T, ocl::Buffer<T>>("results", None)
                            .arg_scl_named::<i32>("count", None)
                    );
                }
            );
            let kernel = KERNEL.as_mut().unwrap();

            let kp = get_work_sizes(kernel);
            let queue = get_cl_data::<T>();


            if a.len() <= kp.work_group_count { // No need to reduce?
                let res: T = a.to_vec().iter().zip(b.to_vec().iter()).map(|(a, b)|
                    if abs_diff(*a, *b) > T::from_f64(0.5) {
                        NetworkParameter::zero()
                    } else {
                        NetworkParameter::one()
                    }
                ).sum();
                return res / T::from_usize(a.len())
            }


            let mut tmp = Vector::<T>::uninitialized_lock_free(kp.work_group_count, queue);

            kernel.set_arg_buf_named("data", Some(a.get_buf())).unwrap();
            kernel.set_arg_buf_named("expected_data", Some(b.get_buf())).unwrap();
            kernel.set_arg_buf_named("results", Some(tmp.get_buf_mut())).unwrap();
            kernel.set_arg_scl_named("count", a.len() as i32).unwrap();
            kernel.set_arg_unchecked(4, ocl::enums::KernelArg::Local::<T>(&kp.work_group_size))
                .unwrap();

            let mut event = ocl::Event::empty();
            kernel.cmd()
                .gws(kp.global_work_size)
                .enew(&mut event)
                .lws(kp.work_group_size)
                .enq()
                .unwrap();
            event.wait_for().unwrap();
            let sum = tmp.to_vec().into_iter().sum::<T>();
            let d = T::from_usize(a.len());

            sum / d // Normalize to range 0.0-1.0
        }
    }
}

macro_rules! helper {
    ($kernel:ident, $t:ty) => ();
    ($kernel:ident, $t:ty, $arg:expr) => {
        $kernel = $kernel.arg_buf_named::<$t, ocl::Buffer<$t>>($arg, None);
    };
    ($kernel:ident, $t:ty, $arg:expr, $($args:expr),+) => {
        helper!($kernel, $t, $arg);//$kernel.set_arg_buf_named::<$t, ocl::Buffer<$t>>($arg, None).unwrap();
        helper!($kernel, $t, $($args),*)
    };
}

macro_rules! kernel_helper {
    ($kernel_name:expr, $t:ty, $($args:expr),* ) => {

        static mut KERNEL: Option<ocl::Kernel> = None;
        static mut INIT_ONCE: ::std::sync::Once = ::std::sync::ONCE_INIT;

        unsafe {
            INIT_ONCE.call_once(
                || {
                    KERNEL = Some(
                        {
                            let mut kernel = linear_algebra::create_kernel::<$t>($kernel_name);
                            helper!(kernel, $t, $($args),*);
                            kernel
                        }
                    );
                }
            )
        }
    };
}

fn activation_func_in_place<T: NetworkParameter>(mut data: Vector<T>) -> Vector<T> {
    kernel_helper!(&format!("{}_{}", T::type_to_str(), "sigmoid_in_place"), T, "C");
    unsafe {
        let kernel = KERNEL.as_mut().unwrap();

        kernel.set_arg_buf_named("C", Some(data.get_buffer_mut())).unwrap();

        let mut event = ocl::Event::empty();
        kernel.cmd().enew(&mut event).gws(data.len()).enq().unwrap();
        event.wait_for().unwrap();

        data
    }
}

fn activation_func<T: NetworkParameter>(data: &Vector<T>) -> Vector<T> {
    kernel_helper!(&format!("{}_{}", T::type_to_str(), "sigmoid"), T, "C", "B");

    unsafe {
        let mut res = Vector::uninitialized(data.len());
        let kernel = KERNEL.as_mut().unwrap();

        kernel.set_arg_buf_named("C", Some(res.get_buffer_mut())).unwrap();
        kernel.set_arg_buf_named("B", Some(data.get_buffer())).unwrap();

        let mut event = ocl::Event::empty();
        kernel.cmd().enew(&mut event).gws(data.len()).enq().unwrap();
        event.wait_for().unwrap();

        res
    }
}

fn activation_func_prime<T: NetworkParameter>(data: &Vector<T>) -> Vector<T> {
    kernel_helper!(&format!("{}_{}", T::type_to_str(), "sigmoid_prime"), T, "C", "B");

    unsafe {
        let mut res = Vector::uninitialized(data.len());
        let kernel = KERNEL.as_mut().unwrap();

        kernel.set_arg_buf_named("C", Some(res.get_buffer_mut())).unwrap();
        kernel.set_arg_buf_named("B", Some(data.get_buffer())).unwrap();

        let mut event = ocl::Event::empty();
        kernel.cmd().enew(&mut event).gws(data.len()).enq().unwrap();
        event.wait_for().unwrap();

        res
    }
}

fn abs_diff<T: NetworkParameter + ::std::ops::Sub<T, Output=T> + ::std::ops::Neg<Output=T> + ::std::cmp::PartialOrd>(a: T, b: T) -> T {
    let diff = a - b;
    if diff < T::zero() {
        -diff
    } else {
        diff
    }
}