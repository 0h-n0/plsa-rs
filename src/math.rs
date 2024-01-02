use ndarray as na;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MathError {
    #[error("[thiserror] shape mismatch error.")]
    ShapeMismatchError(String),
}

pub fn tensor3_broadcast_numpy_like<S, S2>(
    tensor1: &na::ArrayBase<S, na::Ix3>,
    tensor2: &na::ArrayBase<S2, na::Ix3>,
    // https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html
) -> Result<na::Array3<f64>, String>
where
    S: na::Data<Elem = f64>,
    S2: na::Data<Elem = f64>,
{
    /// Ref: https://numpy.org/doc/stable/user/basics.broadcasting.html
    let shape1 = tensor1.dim();
    let shape2 = tensor2.dim();
    let mut output_shape = (0, 0, 0);
    if shape1.0 != shape2.0 && (shape1.0 == 1 || shape2.0 == 1) {
        output_shape.0 = std::cmp::max(shape1.0, shape2.0);
    } else if shape1.0 == shape2.0 {
        output_shape.0 = shape1.0;
    } else {
        Err::<String, MathError>(MathError::ShapeMismatchError(format!(
            "tensor1.shape = {:?}, tensor2.shape = {:?}",
            &shape1, &shape2
        )));
    }
    if shape1.1 != shape2.1 && (shape1.1 == 1 || shape2.1 == 1) {
        output_shape.1 = std::cmp::max(shape1.1, shape2.1);
    } else if shape1.1 == shape2.1 {
        output_shape.1 = shape1.1;
    } else {
        Err::<String, MathError>(MathError::ShapeMismatchError(format!(
            "tensor1.shape = {:?}, tensor2.shape = {:?}",
            &shape1, &shape2
        )));
    }
    if shape1.2 != shape2.2 && (shape1.2 == 1 || shape2.2 == 1) {
        output_shape.2 = std::cmp::max(shape1.2, shape2.2);
    } else if shape1.2 == shape2.2 {
        output_shape.2 = shape1.2;
    } else {
        Err::<String, MathError>(MathError::ShapeMismatchError(format!(
            "tensor1.shape = {:?}, tensor2.shape = {:?}",
            &shape1, &shape2
        )));
    }
    let mut output_tensor: na::Array3<f64> = na::Array3::zeros(output_shape);
    for i in 0..output_shape.0 {
        let shape1_i = std::cmp::min(i, shape1.0 - 1);
        let shape2_i = std::cmp::min(i, shape2.0 - 1);
        for j in 0..output_shape.1 {
            let shape1_j = std::cmp::min(j, shape1.1 - 1);
            let shape2_j = std::cmp::min(j, shape2.1 - 1);
            for k in 0..output_shape.2 {
                let shape1_k = std::cmp::min(k, shape1.2 - 1);
                let shape2_k = std::cmp::min(k, shape2.2 - 1);
                let v1 = tensor1[(shape1_i, shape1_j, shape1_k)];
                let v2 = tensor2[(shape2_i, shape2_j, shape2_k)];
                output_tensor[[i, j, k]] = v1 * v2;
            }
        }
    }
    Ok(output_tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor3_broadcast_numpy_like() -> Result<(), String> {
        let tensor1 = na::Array::from_shape_vec((3, 1, 3), (0..9).map(|x| x as f64).collect())
            .expect("failed: into shape");
        let tensor2 = na::Array::from_shape_vec((1, 1, 3), (0..3).map(|x| x as f64).collect())
            .expect("failed: into shape");
        let ans = na::Array::from_shape_vec(
            (3, 1, 3),
            vec![0, 1, 4, 0, 4, 10, 0, 7, 16]
                .iter()
                .map(|x| *x as f64)
                .collect(),
        )
        .expect("failed into shape");
        let result = tensor3_broadcast_numpy_like(&tensor1, &tensor2)?;
        let result2 = tensor3_broadcast_numpy_like(&tensor2, &tensor1)?;
        assert_eq!(result, ans, "First test failed");
        assert_eq!(result2, ans, "First test failed");
        let tensor1 = na::Array::from_shape_vec((5, 1, 2), (0..10).map(|x| x as f64).collect())
            .expect("failed: into shape");
        let tensor2 = na::Array::from_shape_vec((1, 4, 2), (0..8).map(|x| x as f64).collect())
            .expect("failed: into shape");
        let ans = vec![
            0, 1, 0, 3, 0, 5, 0, 7, 0, 3, 4, 9, 8, 15, 12, 21, 0, 5, 8, 15, 16, 25, 24, 35, 0, 7,
            12, 21, 24, 35, 36, 49, 0, 9, 16, 27, 32, 45, 48, 63,
        ]
        .iter()
        .map(|x| *x as f64)
        .collect();
        let ans = na::Array::from_shape_vec((5, 4, 2), ans).expect("failed into shape");
        let result = tensor3_broadcast_numpy_like(&tensor1, &tensor2)?;
        let result2 = tensor3_broadcast_numpy_like(&tensor2, &tensor1)?;
        assert_eq!(result, ans, "Seccond test failed");
        assert_eq!(result2, ans, "Seccond test failed");
        Ok(())
    }
}
