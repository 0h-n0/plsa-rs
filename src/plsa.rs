use ndarray as na;

pub struct EMParams {
    _max_iter: usize,
    _epslion: f64,
}

pub struct PLSA {
    n_docs: usize,
    n_words: usize,
    n_topics: usize,
    em_params: EMParams,
    prob_matricies: Matrices<f64>,
}

pub struct Matrices<T> {
    pub pz: na::Array1<T>,
    pub pw_z: na::Array2<T>,
    pub pd_z: na::Array2<T>,
    pub pz_wd: na::Array3<T>,
}

impl EMParams {
    pub fn new() -> EMParams {
        EMParams {
            _max_iter: 200,
            _epslion: 1e-6,
        }
    }
    pub fn max_iter(&mut self, max_iter: usize) -> &mut EMParams {
        self._max_iter = max_iter;
        self
    }
    pub fn epslion(&mut self, epslion: f64) -> &mut EMParams {
        self._epslion = epslion;
        self
    }
}

impl PLSA {
    pub fn new(input: na::Array2<f64>, n_topics: usize) -> PLSA {
        let n_docs = input.shape()[0];
        let n_words = input.shape()[1];
        let em_params = EMParams::new();
        let prob_matricies = Matrices {
            pz: na::Array1::<f64>::zeros(n_topics),
            pw_z: na::Array2::<f64>::zeros((n_words, n_topics)),
            pd_z: na::Array2::<f64>::zeros((n_docs, n_topics)),
            pz_wd: na::Array3::<f64>::zeros((n_topics, n_words, n_docs)),
        };
        PLSA {
            n_docs,
            n_words,
            n_topics,
            em_params,
            prob_matricies,
        }
    }
    fn normalize_vector(vec: na::Array1<f64>) -> na::Array1<f64> {
        let sum = vec.sum();        
        vec * (1. / sum)        
    }
    fn normalize_matrix_with_row(mut mat: na::Array2<f64>) -> na::Array2<f64> {
        let row_axis: usize = 1;
        let sum = mat.sum_axis(na::Axis(row_axis));
        let (n_row, _) = mat.dim();
        for row in 0..n_row {            
            let normalized_row = mat.slice(na::s![row, ..]);
            let normalize_row = normalized_row * (1. / sum[row]);
        }
        // let row[0] = mat.slice_axis_mut(na::Axis(0), na::Slice::from(..)) *= sum[0]
        // mat.slice_axis_mut(na::Axis(axis), na::Slice::from(..)).assign(&sum);
        mat
    }
}

#[cfg(test)]
mod tests {
    use super::*;    
    #[test]
    fn normalize_vec() {
        assert_eq!(PLSA::normalize_vector(na::array![1., 2., 3.]), na::array![1./6., 2./6., 3./6.]);
    }
    #[test]
    fn normalize_matrix_with_row() {
        assert_eq!(PLSA::normalize_matrix_with_row(na::array![[1., 2., 3.], [4., 5., 6.]]), na::array![[1./6., 2./6., 3./6.], [4./15., 5./15., 6./15.]]);
    }
}
