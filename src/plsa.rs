use crate::math;
use log::{debug, error, info, trace, warn};
use ndarray as na;
use ndarray_rand::{rand_distr::Uniform, RandomExt};

pub struct EMParams {
    _max_iter: usize,
    _epslion: f64,
}

pub struct PLSA {
    input: na::Array2<f64>,
    n_docs: usize,
    n_words: usize,
    n_topics: usize,
    pub em_params: EMParams,
    pub prob_matricies: Matrices<f64>,
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
        let mut pz = na::Array1::random(n_topics, Uniform::new(0.0, 1.0));
        let mut pw_z = na::Array2::random((n_topics, n_words), Uniform::new(0.0, 1.0));
        let mut pd_z = na::Array2::random((n_topics, n_docs), Uniform::new(0.0, 1.0));
        PLSA::normalize_vector(&mut pz);
        PLSA::normalize_matrix_with_row(&mut pw_z);
        PLSA::normalize_matrix_with_row(&mut pd_z);
        let prob_matricies = Matrices {
            pz: pz,
            pw_z: pw_z,
            pd_z: pd_z,
            pz_wd: na::Array3::random((n_topics, n_words, n_docs), Uniform::new(0.0, 1.0)),
        };
        PLSA {
            input,
            n_docs,
            n_words,
            n_topics,
            em_params,
            prob_matricies,
        }
    }
    pub fn train(&mut self) {
        let mut prev_llh = 1000.;
        let k = 200;
        let epsilon = 1.0e-6;

        for _ in 0..k {
            debug!("{:?}", self.prob_matricies.pz);
            self.e_step();
            self.m_step();
            let llh = self.log_likelifood();
            if ((llh - prev_llh) / prev_llh).abs() < epsilon {
                break;
            }
            debug!("{:?}", self.prob_matricies.pz);
            prev_llh = llh;
            break;
        }
    }
    fn e_step(&mut self) {
        let pz = self
            .prob_matricies
            .pz
            .view()
            .insert_axis(na::Axis(0))
            .insert_axis(na::Axis(1));
        let pw_z = self.prob_matricies.pw_z.t().insert_axis(na::Axis(0));
        let pd_z = self.prob_matricies.pd_z.t().insert_axis(na::Axis(1));
        debug!(
            "self.prob_matricies.pz.shape() = {:?}",
            self.prob_matricies.pz.shape()
        );
        debug!(
            "self.prob_matricies.pw_z.t().shape() = {:?}",
            self.prob_matricies.pw_z.shape()
        );
        debug!(
            "self.prob_matricies.pd_z.t().shape() = {:?}",
            self.prob_matricies.pd_z.shape()
        );
        debug!("pz.shape() = {:?}", pz.shape());
        debug!("pw_z.shape() = {:?}", pw_z.shape());
        debug!("pd_z.shape() = {:?}", pd_z.shape());
        self.prob_matricies.pz_wd = &pz * &pw_z * &pd_z;
        self.prob_matricies.pz_wd /= &self
            .prob_matricies
            .pz_wd
            .sum_axis(na::Axis(2))
            .insert_axis(na::Axis(2));
    }

    fn m_step(&mut self) {
        debug!("self.input: {:?}", &self.input);
        debug!("self.pz_wd: {:?}", &self.prob_matricies.pz_wd);
        let n_p = &self.input.view().insert_axis(na::Axis(2)) * &self.prob_matricies.pz_wd;
        debug!("n_p.shpae: {:?}", n_p.shape());
        debug!("n_p: {:?}", n_p);
        self.prob_matricies.pz = n_p.sum_axis(na::Axis(0)).sum_axis(na::Axis(0));
        debug!(
            "n_p, pz: {:?}",
            n_p.sum_axis(na::Axis(0)).sum_axis(na::Axis(0))
        );
        // (x, y, z).sum_axis(0) => (y, z)
        // (y, z).sum_axis(0) => (z)
        self.prob_matricies.pw_z = n_p.sum_axis(na::Axis(0)).reversed_axes();
        self.prob_matricies.pd_z = n_p.sum_axis(na::Axis(1)).reversed_axes();
        PLSA::normalize_vector(&mut self.prob_matricies.pz);
        PLSA::normalize_matrix_with_row(&mut self.prob_matricies.pw_z);
        PLSA::normalize_matrix_with_row(&mut self.prob_matricies.pd_z);
    }
    fn log_likelifood(&self) -> f64 {
        let pz = self
            .prob_matricies
            .pz
            .view()
            .insert_axis(na::Axis(0))
            .insert_axis(na::Axis(1));
        debug!("log_likelifood pz: {:?}", pz);
        let pw_z = self.prob_matricies.pw_z.t().insert_axis(na::Axis(0));
        let pd_z = self.prob_matricies.pd_z.t().insert_axis(na::Axis(1));
        let pz_pw_z = math::tensor3_broadcast_numpy_like(&pz, &pw_z).unwrap();
        let pz_wd = math::tensor3_broadcast_numpy_like(&pz_pw_z, &pd_z).unwrap();
        let mut pwd = pz_wd.sum_axis(na::Axis(2));
        let sum = pwd.sum();
        pwd /= sum;

        let log_pwd = pwd.mapv(|x| x.ln());
        (&self.input * &log_pwd).sum()
    }
    fn calc_aic(&self) -> f64 {
        2. * self.n_topics as f64 - 2. * self.log_likelifood()
    }
    fn normalize_vector<S>(vec: &mut na::ArrayBase<S, na::Ix1>)
    where
        S: na::Data<Elem = f64> + na::DataMut,
    {
        let sum = vec.sum();
        *vec /= sum;
    }
    fn normalize_matrix_with_row(mat: &mut na::Array2<f64>) {
        let row_axis: usize = 1;
        let sum = mat.sum_axis(na::Axis(row_axis));
        for (i, mut row) in mat.axis_iter_mut(na::Axis(0)).enumerate() {
            row /= sum[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalize_vec() {
        let mut array = na::array![1., 2., 3.];
        PLSA::normalize_vector(&mut array);
        println!("==> {:?}", array);
        assert_eq!(array, na::array![1. / 6., 2. / 6., 3. / 6.]);
    }
    #[test]
    fn normalize_matrix_with_row() {
        let mut mat = na::array![[1., 2., 3.], [4., 5., 6.]];
        PLSA::normalize_matrix_with_row(&mut mat);
        assert_eq!(
            mat,
            na::array![[1. / 6., 2. / 6., 3. / 6.], [4. / 15., 5. / 15., 6. / 15.]]
        );
    }
}
