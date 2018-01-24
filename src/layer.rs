
extern crate linear_algebra;

use self::linear_algebra::vector::Vector;
use self::linear_algebra::matrix::Matrix;
use traits::NetworkParameter;

pub struct Layer<T: NetworkParameter> {
    pub biases: Vector<T>,
    pub weights: Matrix<T>
}

impl<T: NetworkParameter> Layer<T> {
    pub fn new<F: FnMut() -> f64>(distribution: &mut F, input_count: usize, node_count: usize) -> Layer<T> {
        let factor = T::from_f64(1.0 / (input_count as f64).sqrt());

        //TODO: make use of linear_math lib for this to speed up
        let biases = Vector::from_vec((0..node_count).map(
            |_| T::from_f64(distribution())
        ).collect());

        //TODO: make use of linear_math lib for this to speed up
        let weights =  Matrix::from_vec((0..(input_count * node_count)).map(
            |_|
                factor * T::from_f64(distribution())
        ).collect(),input_count, node_count);

        Layer{
            biases,
            weights
        }
    }

    /// Save Layer to specified path.
    /// NOTE! The file will be encoded in the current systems endianness
    ///
    /// ---- Formated like ----
    /// biases,
    /// weights
    ///
    pub fn write_to_file(&self, file: &mut ::std::fs::File) -> Result<(), ::std::io::Error> {
        self.biases.write_to_file(file)?;
        self.weights.write_to_file(file)
    }

    /// Open Layer from specified path.
    /// NOTE! The file will be interpreted in the current systems endianness
    ///
    /// ---- Formated like ----
    /// biases,
    /// weights
    ///
    pub unsafe fn read_from_file(file: &mut ::std::fs::File) -> Result<Layer<T>, ::std::io::Error> {
        Ok(Layer {
            biases: Vector::<T>::read_from_file(file)?,
            weights: Matrix::<T>::read_from_file(file)?
        })
    }
}