use ndarray as na;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use plsa_rs;

#[test]
fn integrate_test() {
    plsa_rs::logger::setup_logger();
    let input = na::Array::from_shape_vec(
        (5, 4),
        vec![
            20, 23, 1, 4, 25, 19, 3, 0, 2, 1, 31, 28, 0, 1, 22, 17, 1, 0, 18, 24,
        ],
    )
    .unwrap()
    .mapv(|x| x as f64);
    let mut params = plsa_rs::EMParams::new();
    let mut plsa = plsa_rs::PLSA::new(input, 2);
    plsa.train();
    println!("{:?}", plsa.prob_matricies.pz);
}

#[test]
fn heavy_integrate_test() {
    plsa_rs::logger::setup_logger();
    let input = na::Array::from_shape_vec(
        (5, 4),
        vec![
            20, 23, 1, 4, 25, 19, 3, 0, 2, 1, 31, 28, 0, 1, 22, 17, 1, 0, 18, 24,
        ],
    )
    .unwrap()
    .mapv(|x| x as f64);
    let mut params = plsa_rs::EMParams::new();
    let mut plsa = plsa_rs::PLSA::new(input, 2);
    plsa.train();
    println!("{:?}", plsa.prob_matricies.pz);
}
