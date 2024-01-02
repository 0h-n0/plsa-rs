use ndarray as na;
use plsa_rs;

fn main() {
    let input = na::Array::from_shape_vec(
        (5, 4),
        vec![
            20, 23, 1, 4, 25, 19, 3, 0, 2, 1, 31, 28, 0, 1, 22, 17, 1, 0, 18, 24,
        ],
    )
    .unwrap();
    let input = input.mapv(|x| x as f64);
    let mut plsa = plsa_rs::PLSA::new(input, 2);
    plsa.train();
    println!("hello world.");
}
