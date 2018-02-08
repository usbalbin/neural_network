

pub mod network;
pub mod layer;
pub mod traits;

#[cfg(test)]
mod tests;

#[macro_use]
extern crate lazy_static;

extern crate ocl;
extern crate linear_algebra;

use traits::NetworkParameter;

use std::sync::Mutex;
use std::sync::MutexGuard;
use std::collections::HashMap;
use std::ops::DerefMut;
use std::ops::Deref;




struct KernelsGuard<'a, 'b>(MutexGuard<'a, HashMap<String, Kernels>>, &'b str);

impl<'a, 'b> Deref for KernelsGuard<'a, 'b> {
    type Target = Kernels;
    fn deref(&self) -> &Self::Target {
        self.0.get(self.1).unwrap()
    }
}

impl<'a, 'b> DerefMut for KernelsGuard<'a, 'b> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.get_mut(self.1).unwrap()
    }
}

struct Kernels{
    sigmoid_in_place: ocl::Kernel,
    sigmoid: ocl::Kernel,
    sigmoid_prime: ocl::Kernel,

    relu_in_place: ocl::Kernel,
    relu: ocl::Kernel,
    relu_prime: ocl::Kernel
}









fn get_kernels<'a, T: NetworkParameter>() -> KernelsGuard<'a, 'static> {
    use std::collections::HashMap;

    lazy_static! {
        static ref KERNELS: Mutex<HashMap<String, Kernels>> = Mutex::new(HashMap::new());
    }

    let mut data = KERNELS.lock().unwrap();
    let ty = T::type_to_str().to_owned();

    if !data.contains_key(&ty){
        let kernels = {
            let sigmoid_in_place = linear_algebra::create_kernel::<T>(&format!("{}_{}", T::type_to_str(), "sigmoid_in_place"))
                .arg_buf_named::<T, ocl::Buffer<T>>("C", None);

            let sigmoid = linear_algebra::create_kernel::<T>(&format!("{}_{}", T::type_to_str(), "sigmoid"))
                .arg_buf_named::<T, ocl::Buffer<T>>("C", None)
                .arg_buf_named::<T, ocl::Buffer<T>>("B", None);


            let sigmoid_prime = linear_algebra::create_kernel::<T>(&format!("{}_{}", T::type_to_str(), "sigmoid_prime"))
                .arg_buf_named::<T, ocl::Buffer<T>>("C", None)
                .arg_buf_named::<T, ocl::Buffer<T>>("B", None);


            let relu_in_place = linear_algebra::create_kernel::<T>(&format!("{}_{}", T::type_to_str(), "relu_in_place"))
                .arg_buf_named::<T, ocl::Buffer<T>>("C", None)
                .arg_scl_named::<T>("A", None);

            let relu = linear_algebra::create_kernel::<T>(&format!("{}_{}", T::type_to_str(), "relu"))
                .arg_buf_named::<T, ocl::Buffer<T>>("C", None)
                .arg_scl_named::<T>("A", None)
                .arg_buf_named::<T, ocl::Buffer<T>>("B", None);


            let relu_prime = linear_algebra::create_kernel::<T>(&format!("{}_{}", T::type_to_str(), "relu_prime"))
                .arg_buf_named::<T, ocl::Buffer<T>>("C", None)
                .arg_scl_named::<T>("A", None)
                .arg_buf_named::<T, ocl::Buffer<T>>("B", None);

            Kernels{
                sigmoid,
                sigmoid_in_place,
                sigmoid_prime,

                relu_in_place,
                relu,
                relu_prime
            }
        };
        data.insert(ty, kernels);
    }

    KernelsGuard(data, T::type_to_str())
}