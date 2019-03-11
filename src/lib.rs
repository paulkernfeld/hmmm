#![cfg_attr(feature = "benchmark", feature(test))]
//! This library contains a Rust implementation of a time-invariant Hidden Markov model with
//! discrete observations. It includes maximum likelihood estimation via the Baum-Welch
//! expectation-maximization algorithm and hidden state inference via the Viterbi algorithm.
//!
//! See [`hmmm::HMM`](struct.HMM.html) for detailed documentation on how to work with this library.
//!
//! Below, the HMM is trained to recognize the pattern `001001001...`
//!
//! ```
//! use hmmm::HMM;
//! use ndarray::{array, Array1};
//! use rand::{SeedableRng, XorShiftRng};
//!
//! fn main() {
//!     let training_ys = array![0, 0, 1, 0, 0, 1, 0];
//!     let mut rng = XorShiftRng::seed_from_u64(1337);
//!     let hmm = HMM::train(&training_ys, 3, 2, &mut rng);
//!     let sampled_ys: Array1<usize> = hmm.sampler(&mut rng)
//!         .map(|sample| sample.y)
//!         .take(10)
//!         .collect();
//!     assert_eq!(array![0, 0, 1, 0, 0, 1, 0, 0, 1, 0], sampled_ys);
//! }
//! ```
//!
//! ## Building
//!
//! This project uses `cargo-make`. See `Makefile.toml` for a full list of build commands, but the
//! main useful command for this project is `cargo make all`.
//!
//! There is a small amount of benchmarking functionality gated by the `benchmark` feature.
//!
//! ## Notes
//!
//! Sections 17.3 and 17.4 of *Machine Learning a Probabilistic Perspective* by Kevin Murphy, 2012
//! were invaluable as a reference, as was section 13.2 of *Pattern Recognition and Machine
//! Learning* by Christopher Bishop, 2016.
//!
//! I have attempted to make the math notation readable both as rendered HTML and from the source
//! code. The notation is strongly inspired by the Wikipedia page on the Baum-Welch algorithm.
#[cfg(feature = "benchmark")]
extern crate test;

use self::ndarray_utils::*;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::{array, s};
use rand::prelude::*;
#[cfg(test)]
use rand::XorShiftRng;
use spectral::prelude::*;
use std::f64;

const TOLERANCE: f64 = 1e-5; // Chosen completely arbitrarily

/// This struct represents a trained HMM, including values for each parameter.
///
/// # Math
///
/// The HMM is used to predict a sequence of observations:
///
/// $$Y=(Y_0=y_0, Y_1=y_1, \ldots, Y_{T-1}=y_{T-1})$$
///
/// ...where each $y_t \in [0, K)$.
///
/// It accomplishes this with latent variables for hidden state $X=(X_0, \ldots, X_{T-1})$ where each
/// $x_t \in [0, N)$.
///
/// A trained HMM has three parameters:
/// * $A$, the $N × N$ state transition matrix: $a_{ij}=P(X_t=j|X_{t-1}=i)$
/// * $B$, the $N × K$ observation matrix: $b_{ik}=P(Y_t=y_k|X_t=i)$
/// * $π$, the $N$-length initial state distribution: $π_i=P(X_1=i)$
#[derive(Debug)]
pub struct HMM {
    pub a: Array2<f64>,
    pub b: Array2<f64>,
    pub pi: Array1<f64>,
}

impl HMM {
    /// Create a new HMM with the given parameters.
    ///
    /// This could be useful for loading a saved trained model.
    ///
    /// Panics if any of:
    /// - Dimensions are invalid
    /// - Probability distributions are invalid
    pub fn new(a: Array2<f64>, b: Array2<f64>, pi: Array1<f64>) -> Self {
        // Check all dimensions
        {
            asserting("B must have a positive number of rows")
                .that(&b.rows())
                .is_greater_than(0);
            asserting("B must have a positive number of columns")
                .that(&b.cols())
                .is_greater_than(0);
            assert_eq!(
                a.rows(),
                b.rows(),
                "A and B must have the same number of rows"
            );
            assert_eq!(a.rows(), a.cols(), "A must be square");
            assert_eq!(a.rows(), pi.len(), "π must be of length N");
        }

        // Check that each row of A is a distribution
        {
            for a_ij in &a {
                assert_that(a_ij).is_greater_than_or_equal_to(0.0)
            }
            for row in a.genrows() {
                asserting("Each row of A must sum to 1")
                    .that(&row.sum())
                    .is_close_to(1.0, TOLERANCE);
            }
        }

        // Check that each row of B is a distribution
        {
            for b_ik in &b {
                assert_that(b_ik).is_greater_than_or_equal_to(0.0)
            }

            for row in b.genrows() {
                asserting("Each row of B must sum to 1")
                    .that(&row.sum())
                    .is_close_to(1.0, TOLERANCE);
            }
        }

        // Check that π is a distribution
        {
            for pi_i in &pi {
                assert_that(pi_i).is_greater_than_or_equal_to(0.0)
            }

            asserting("π must sum to 1")
                .that(&pi.sum())
                .is_close_to(1.0, TOLERANCE);
        }

        Self { a, b, pi }
    }

    /// $N$, the number of states in this HMM
    pub fn n(&self) -> usize {
        self.b.rows()
    }

    /// $K$, the number of possible observations that this model can emit
    pub fn k(&self) -> usize {
        self.b.cols()
    }

    pub fn sampler<'a, R: Rng + ?Sized>(&'a self, rng: &'a mut R) -> HMMSampleIter<R> {
        let a_weighted_choices = self
            .a
            .genrows()
            .into_iter()
            .map(|row| WeightedChoiceFloat::from_pmf(row.as_slice().unwrap()))
            .collect();
        let b_weighted_choices = self
            .b
            .genrows()
            .into_iter()
            .map(|row| WeightedChoiceFloat::from_pmf(row.as_slice().unwrap()))
            .collect();
        let c_weighted_choice = WeightedChoiceFloat::from_pmf(self.pi.as_slice().unwrap());
        HMMSampleIter {
            a_weighted_choices,
            b_weighted_choices,
            c_weighted_choice,
            rng,
            current_state: None,
        }
    }

    /// Given an iterator of observations, this returns a new iterator that yields the probability
    /// of being in each hidden state at each future time step. This method is relatively efficient
    /// from the standpoint of memory and computation time.
    ///
    /// If you can store the whole sequence in memory, there are more accurate ways to compute the
    /// probability of being in each state at a particular point in time, because it is possible to
    /// use the observations from the future to better inform the probability of being in each
    /// hidden state at any particular time t.
    ///
    /// This is not closely related to the meaning of "filter" as in `std::iter::Iterator::filter`.
    ///
    /// Panics if an observation is out of bounds.
    pub fn filter<I>(&self, ys: I) -> HMMFilterIter<I::IntoIter>
    where
        I: IntoIterator<Item = usize>,
    {
        HMMFilterIter {
            hmm: self,
            observations: ys.into_iter(),
            current_item: None,
        }
    }

    /// Given a distribution over states, calculate the probable distribution of states at a time in
    /// the future.
    ///
    /// This is currently only efficient for small values of `n_time_steps`. In order to be more
    /// efficient, we want to be able to efficiently raise `self.a` to a power, which means getting
    /// eigenvalues and eigenvectors.
    ///
    /// Panics if:
    /// - The length of `p_states` is invalid
    /// - `p_states` is not probability distribution
    pub fn predict(&self, mut p_states: Array1<f64>, n_time_steps: usize) -> Array1<f64> {
        asserting("p_states must sum to 1")
            .that(&p_states.sum())
            .is_close_to(1.0, TOLERANCE);
        for _ in 0..n_time_steps {
            p_states = p_states.dot(&self.a)
        }
        p_states
    }

    // Backwards: given observations after (not including) time t, what is the probability that
    // we are in each state at time t?
    //
    // Since this iterator runs backwards, it is collected backwards into the Vec. When iterated
    // backwards, element `t` of the iterator contains the p_states at time `t`.
    fn filter_backwards(&self, ys: &Array1<usize>) -> Vec<Array1<f64>> {
        ys.iter()
            .rev()
            .scan(
                None,
                |p_states_option: &mut Option<Array1<f64>>, &observation| {
                    let (new, old) = if let Some(p_states) = p_states_option {
                        (
                            self.a.dot(p_states) * self.b.column(observation),
                            self.a.dot(p_states),
                        )
                    } else {
                        (self.b.column(observation).to_owned(), uniform(self.n()))
                    };
                    *p_states_option = Some(new.normalize("filter_backwards_new"));
                    Some(old.normalize("filter_backwards_old"))
                },
            )
            .collect()
    }

    /// Given a sequence of observations, compute the probability of being in any given state at
    /// each point in time.
    ///
    /// Return a $T × N$ matrix where element (t, k) is the probability that we are in state k at
    /// time t.
    ///
    /// This is the forward-backward algorithm.
    pub fn smooth(&self, ys: &Array1<usize>) -> Array2<f64> {
        // Forwards: given observations up to and including t, what is the probability that we are
        // in each state at time t?
        let forwards = self.filter(ys.iter().cloned());

        // Backwards: given observations after (not including) time t, what is the probability that
        // we are in each state at time t?
        let backwards = self.filter_backwards(&ys);

        // Construct the result as an Array1 of length $T * N$, then reshape it it into an array of
        // shape $T × N$.
        let mut to_return = Array2::zeros((ys.len(), self.n()));
        forwards
            .zip(backwards.iter().rev())
            .enumerate()
            .for_each(|(t, (forward, backward))| {
                to_return
                    .slice_mut(s![t, ..])
                    .assign(&(forward.p_states * backward).normalize("smooth"))
            });
        to_return
    }

    /// This is the Viterbi algorithm. Given a sequence of observations, return the most likely
    /// sequence of states.
    ///
    /// It's possible to do this in log space but I normalized instead to make it feel more like the
    /// forwards-backwards algorithm.
    ///
    pub fn most_likely_sequence(&self, ys: &Array1<usize>) -> Array1<usize> {
        // Special-case when the sequence of observations is empty
        if ys.is_empty() {
            return array![];
        }

        // probs is a T × N matrix where probs[t, i] is the probability that we are in state i at
        // time t given all observations up to time t and assuming the most likely sequence of
        // hidden states up to time t.
        let mut probs = Array2::zeros((ys.len(), self.n()));

        // x_to_prev_x is a (T - 1) × N matrix where each entry x_to_prev_x[t, i] is the most likely
        // state that would have occurred at time t - 1 given that we're in state i at time t.
        let mut x_to_prev_x = Array2::zeros((ys.len() - 1, self.n()));

        probs
            .row_mut(0)
            .assign(&(self.pi.clone() * self.b.column(ys[0])).normalize("viterbi_0"));

        for t in 1..ys.len() {
            let y = ys[t];
            asserting("y is too big").that(&y).is_less_than(&self.k());
            for i in 0..self.n() {
                for j in 0..self.n() {
                    let prob_i_j = probs[(t - 1, i)] * self.a[(i, j)] * self.b[(j, y)];
                    if prob_i_j > probs[(t, j)] {
                        probs[(t, j)] = prob_i_j;
                        x_to_prev_x[(t - 1, j)] = i;
                    }
                }
            }

            // Normalize to prevent underflow
            probs.row_mut(t).nip("viterbi");
        }

        let (mut i, _p) = probs.row(ys.len() - 1).maxfx();
        let mut to_return = Array1::from_elem(ys.len(), usize::max_value());
        to_return[ys.len() - 1] = i;
        for t in (0..ys.len() - 1).rev() {
            i = x_to_prev_x[(t, i)];
            to_return[t] = i;
        }
        to_return
    }

    /// Find the maximum likelihood estimate for the parameters. Caveats:
    /// - If there is not enough data, the MLE is undefined. This implementation will
    ///   use a uniform prior for any parameters for which there isn't enough data.
    /// - This is not guaranteed to find a global minimal, only a local minimum.
    /// - Due to a lack of identifiability, an HMM with $N$ states has $N!$ equivalent solutions.
    /// - Taking the most likely state at each point in time doesn't necessarily result in the most
    ///   likely sequence of states, or even a possible sequence of states. If you want that, use
    ///   `HMM::most_likely_states`.
    ///
    /// Baum-Welch (Baum et. al. 1970) is a variant of the Expectation-Maximization algorithm for
    /// HMMs.
    ///
    /// Let $α_i(t) = P(Y_0=y_0, \ldots, Y_t=y_t, X_t=i | θ)$
    ///
    /// Let $β_i(t) = P(Y_{t+1}=y_{t+1}, \ldots, Y_T=y_T | X_t=i, θ)$
    ///
    /// $$
    /// γ_i(t) = P(X_t=i|Y,θ) = \frac{P(X_t=i,Y|θ)}{P(Y|θ)} =
    ///     \frac{α_i(t)β_i(t)}{\sum_{j=1}^N α_j(t)β_j(t)}
    /// $$
    ///
    /// $γ_i(t)$ is the same thing that the `smooth` method computes: the probability of being in
    /// state $i$ at time $t$, conditioned on both past and future observations.
    ///
    /// $ξ_{ij}(t)$ is the probability of being in state $i$ at time $t$ and in state $j$ at time
    /// $t + 1$:
    ///
    /// $$
    /// ξ_{ij}(t) = P(X_t=i,X_{t+1}=j|Y,θ)
    ///             = \frac{P(X_t=i,X_{t+1}=j,Y|θ)}{P(Y|θ)}
    ///             = \frac{α_i(t) a_{ij} β_j(t+1) b_j(y_{t+1})}
    ///               {\sum_{i=1}^N \sum_{j=1}^N α_i(t) a_{ij} β_j(t+1) b_j(y_{t+1}) }
    /// $$
    ///
    /// Updating the counts:
    ///
    /// $$
    /// π_i^* = γ_i(0)
    /// $$
    ///
    /// For $α$:
    ///
    /// $$
    /// α_{ij}^*=\frac{\sum^{T-1}\_{t=1}ξ\_{ij}(t)}{\sum^{T-1}\_{t=1}γ_i(t)}
    /// $$
    ///
    /// For $β$:
    ///
    /// $$
    /// β_i^*(v_k)=\frac{\sum^T_{t=1} 1_{y_t=v_k} γ_i(t)}{\sum^T_{t=1} γ_i(t)}
    /// $$
    pub fn train<R: Rng>(ys: &Array1<usize>, n: usize, k: usize, rng: &mut R) -> Self {
        for &y in ys {
            assert!(y < k);
        }

        let a = Array2::from_shape_fn((n, n), |_| rng.gen::<f64>()).normalize_rows();
        let b = Array2::from_shape_fn((n, k), |_| rng.gen::<f64>()).normalize_rows();
        let pi = Array1::from_shape_fn(n, |_| rng.gen::<f64>()).normalize("π");

        let uniform_states_dim = Array1::ones(n).normalize("N");
        let uniform_obs_dim = Array1::ones(k).normalize("K");

        let mut hmm = HMM::new(a, b, pi);

        for _ in 0..100 {
            let (a, b, pi) = {
                let alphas: Vec<Array1<f64>> = hmm
                    .filter(ys.iter().cloned())
                    .map(|alpha| alpha.p_states)
                    .collect_vec();
                let betas: Vec<Array1<f64>> =
                    hmm.filter_backwards(&ys).into_iter().rev().collect_vec();
                let gammas: Vec<Array1<f64>> = alphas
                    .iter()
                    .zip(betas.iter())
                    .map(|(alpha_t, beta_t)| (alpha_t * beta_t).normalize("γ"))
                    .collect_vec();

                let xis: Vec<Array2<f64>> = alphas
                    .iter()
                    .zip(betas.iter())
                    .zip(ys.iter())
                    .tuple_windows()
                    .map(
                        |(((alpha_t0, _beta_t0), _obs_t0), ((_alpha_t1, beta_t1), obs_t1))| {
                            let mut xi = Array2::zeros((n, n));
                            for i in 0..n {
                                for j in 0..n {
                                    xi[(i, j)] = alpha_t0[i]
                                        * hmm.a[(i, j)]
                                        * beta_t1[j]
                                        * hmm.b[(j, *obs_t1)]
                                }
                            }
                            let xi_sum: f64 = xi.iter().sum();
                            xi /= xi_sum;
                            xi
                        },
                    )
                    .collect_vec();

                // a is of shape N x N
                let mut a = Array2::zeros((n, n));
                for i in 0..n {
                    let t_minus_1 = 1.max(gammas.len()) - 1;
                    let gammas_sum: f64 = gammas[..t_minus_1].iter().map(|gamma| gamma[i]).sum();
                    if gammas_sum == 0.0 {
                        // If we have never seen a transition away from state i, fall back to a
                        // uniform prior.
                        a.row_mut(i).assign(&uniform_states_dim);
                    } else {
                        for j in 0..n {
                            let xis_sum: f64 = xis.iter().map(|xi| xi[(i, j)]).sum();
                            a[(i, j)] = xis_sum / gammas_sum;
                        }
                    }
                }

                // b is of shape N x K
                let mut b = Array2::zeros((n, k));
                for i in 0..n {
                    let gammas_sum: f64 = gammas.iter().map(|gamma| gamma[i]).sum();
                    if gammas_sum == 0.0 {
                        // If we have never seen state i, fall back to a uniform prior
                        b.row_mut(i).assign(&uniform_obs_dim);
                    } else {
                        for k in 0..k {
                            let numerator: f64 = gammas
                                .iter()
                                .zip(ys)
                                .map(|(gamma, obs)| if k == *obs { gamma[i] } else { 0.0 })
                                .sum();
                            b[(i, k)] = numerator / gammas_sum;
                        }
                    }
                }

                // pi is of length N
                let pi = if gammas.is_empty() {
                    // A special case for when there are no observations
                    uniform_states_dim.clone()
                } else {
                    gammas[0].to_owned()
                };

                (a, b, pi)
            };

            hmm = HMM::new(a, b, pi);
        }

        hmm
    }

    /// Return the log likelihood of a sequence of states and observations. This is not a typical
    /// use case, because often the vector of hidden states is not available.
    ///
    /// Panics if:
    /// - The number of states and observations is not equal
    /// - A state or observation is out of bounds
    pub fn ll_given_states(&self, xs: &[usize], ys: &[usize]) -> f64 {
        assert_eq!(xs.len(), ys.len());

        // This special case is required b/c we treat the first time step specially
        if xs.is_empty() {
            return 0.0;
        }

        // When looping, we skip the initial observation
        let initial_state_log_prob = self.pi[xs[0]].log2();
        let initial_observation_log_prob = self.b[(xs[0], ys[0])].log2();
        let the_rest: f64 = xs
            .iter()
            .zip(ys)
            .tuple_windows()
            .map(|((state0, _observation0), (state1, observation1))| {
                let transition_log_prob = self.a[(*state0, *state1)].log2();
                let observation_log_prob = self.b[(*state1, *observation1)].log2();
                transition_log_prob + observation_log_prob
            })
            .sum();

        initial_state_log_prob + initial_observation_log_prob + the_rest
    }
}

fn uniform(n: usize) -> Array1<f64> {
    Array1::from_elem(n, 1.0 / (n as f64))
}

/// The item yielded by `HMMSampleIter`
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HMMSample {
    pub x: usize,
    pub y: usize,
}

/// An iterator that returns random samples from an HMM
pub struct HMMSampleIter<'a, R: Rng + ?Sized + 'a> {
    a_weighted_choices: Vec<WeightedChoiceFloat>,
    b_weighted_choices: Vec<WeightedChoiceFloat>,
    c_weighted_choice: WeightedChoiceFloat,
    rng: &'a mut R,
    current_state: Option<usize>,
}

impl<'a, R: Rng + ?Sized> Iterator for HMMSampleIter<'a, R> {
    type Item = HMMSample;

    fn next(&mut self) -> Option<Self::Item> {
        let state = if let Some(current_state) = self.current_state {
            self.a_weighted_choices[current_state].sample(self.rng)
        } else {
            self.c_weighted_choice.sample(self.rng)
        };
        self.current_state = Some(state);
        Some(HMMSample {
            x: state,
            y: self.b_weighted_choices[state].sample(self.rng),
        })
    }
}

/// The item yielded by the `HMMFilterIter`.
#[derive(Clone, Debug, PartialEq)]
pub struct HMMFilterItem {
    p_states: Array1<f64>, // The probability that we are in each state currently
}

/// This is an iterator returned by `HMM::filter`.
pub struct HMMFilterIter<'a, I>
where
    I: Iterator<Item = usize>,
{
    hmm: &'a HMM,
    observations: I,
    current_item: Option<HMMFilterItem>,
}

impl<'a, I> Iterator for HMMFilterIter<'a, I>
where
    I: Iterator<Item = usize>,
{
    type Item = HMMFilterItem;

    fn next(&mut self) -> Option<Self::Item> {
        self.observations.next().map(|observation| {
            let observation_probs = self.hmm.b.column(observation).to_owned();
            let transition_probs = if let Some(ref current_item) = self.current_item {
                current_item.p_states.dot(&self.hmm.a)
            } else {
                self.hmm.pi.to_owned()
            };
            let mut p_states = observation_probs * transition_probs;
            // TODO what if no state is possible, eek
            let p_states_sum: f64 = p_states.iter().sum();
            p_states /= p_states_sum;

            let item = HMMFilterItem { p_states };
            self.current_item = Some(item.clone());
            item
        })
    }
}

/// Sample from a [categorical distribution](https://en.wikipedia.org/wiki/Categorical_distribution)
/// where the weight for each category is a float.
pub struct WeightedChoiceFloat {
    cmf: Vec<f64>,
}

impl WeightedChoiceFloat {
    pub fn from_pmf(pmf: &[f64]) -> Self {
        let cmf = pmf
            .iter()
            .scan(0.0, |state, x| {
                *state += x;
                Some(*state)
            })
            .collect();
        Self { cmf }
    }
}

impl Distribution<usize> for WeightedChoiceFloat {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let sampled_uniform = rng.gen::<f64>();
        let (i, _x) = self
            .cmf
            .iter()
            .enumerate()
            .find(|(_i, &x)| sampled_uniform < x)
            .unwrap();
        i
    }
}

/// Create a fast RNG that with reproducible outputs that isn't necessarily cryptographically
/// strong.
#[cfg(test)]
fn new_rng() -> impl Rng {
    XorShiftRng::seed_from_u64(1337)
}

#[cfg(test)]
mod tests_weighted_choice_float {
    use super::*;

    #[test]
    fn unit() {
        let wcf = WeightedChoiceFloat::from_pmf(&[1.0]);
        assert_eq!(0, wcf.sample(&mut new_rng()))
    }

    #[test]
    fn first() {
        let wcf = WeightedChoiceFloat::from_pmf(&[1.0, 0.0]);
        assert_eq!(0, wcf.sample(&mut new_rng()))
    }

    #[test]
    fn last() {
        let wcf = WeightedChoiceFloat::from_pmf(&[0.0, 1.0]);
        assert_eq!(1, wcf.sample(&mut new_rng()))
    }

    #[test]
    fn middle() {
        let wcf = WeightedChoiceFloat::from_pmf(&[0.0, 1.0, 0.0]);
        assert_eq!(1, wcf.sample(&mut new_rng()))
    }
}

mod ndarray_utils {
    use itertools::Itertools;
    use ndarray::prelude::*;
    use ndarray::*;
    use num_traits::{Float, Num, Zero};

    pub trait ArrayFloat<T: Float> {
        fn l2_distance(&self, rhs: &Self) -> T;
    }

    pub trait Array1Float<T: Float> {
        /// Along a 1D array, return the maximum float value and its index
        ///
        /// If there are multiple equal maximum values, one of them will be returned with its index.
        ///
        /// The behavior of this function is unspecified if the array contains NaNs.
        ///
        /// See also `maxfx`
        fn maxf(&self) -> Option<(usize, T)>;

        /// The "expecting" version of `maxf`
        fn maxfx(&self) -> (usize, T);
    }

    pub trait Array1FloatMut {
        fn nip(&mut self, label: &'static str);

        fn normalize(self, label: &'static str) -> Self;
    }

    pub trait Array1Num<T>
    where
        T: Copy + Num,
    {
        fn sum(&self) -> T;
    }

    pub trait Array2FloatMut {
        fn nip_rows(&mut self);

        fn normalize_rows(self) -> Self;
    }

    impl<D, S> ArrayFloat<f64> for ArrayBase<S, D>
    where
        D: Dimension,
        S: Data<Elem = f64>,
    {
        fn l2_distance(&self, rhs: &Self) -> f64 {
            assert_eq!(self.shape(), rhs.shape());
            self.iter()
                .zip(rhs.iter())
                .map(|(&x, &y)| (y - x).powi(2))
                .sum::<f64>()
                .sqrt()
        }
    }

    impl<T, S> Array1Float<T> for ArrayBase<S, Ix1>
    where
        T: Float,
        S: Data<Elem = T>,
    {
        fn maxf(&self) -> Option<(usize, T)> {
            self.iter()
                .enumerate()
                .fold1(|(i0, v0), (i1, v1)| if v0 > v1 { (i0, v0) } else { (i1, v1) })
                .map(|(i, &v)| (i, v))
        }

        fn maxfx(&self) -> (usize, T) {
            self.maxf()
                .expect("maxfx failed because the input had length 0")
        }
    }

    impl<S> Array1FloatMut for ArrayBase<S, Ix1>
    where
        S: DataMut + Data<Elem = f64>,
    {
        fn nip(&mut self, label: &'static str) {
            let sum: f64 = self.sum();
            assert!(
                sum.is_sign_positive(),
                format!("Sum of {} must be positive", label)
            );
            (*self) /= sum;
        }

        fn normalize(mut self, label: &'static str) -> Self {
            self.nip(label);
            self
        }
    }

    impl<T, S> Array1Num<T> for ArrayBase<S, Ix1>
    where
        T: Copy + Num,
        S: Data<Elem = T>,
    {
        fn sum(&self) -> T {
            self.iter().fold(Zero::zero(), |v0, &v1| v0 + v1)
        }
    }

    impl<S> Array2FloatMut for ArrayBase<S, Ix2>
    where
        S: DataMut + Data<Elem = f64>,
    {
        fn nip_rows(&mut self) {
            for mut row in self.genrows_mut() {
                let sum: f64 = row.sum();
                assert!(sum > Zero::zero());
                row /= sum;
            }
        }

        fn normalize_rows(mut self) -> Self {
            self.nip_rows();
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use counter::Counter;
    use lazy_static::lazy_static;
    use std::iter::repeat_with;

    lazy_static! {
        static ref HMM_UNIT: HMM = { HMM::new(array![[1.0]], array![[1.0]], array![1.0]) };
    }

    lazy_static! {
        static ref HMM_PERIODIC: HMM = {
            HMM::new(
                array![[0.0, 1.0], [1.0, 0.0]],
                array![[0.0, 1.0], [1.0, 0.0]],
                array![1.0, 0.0],
            )
        };
    }

    /// A hand-calculated table of paths where each path is equally likely.
    ///
    /// 1 time step (state only)
    /// 0
    /// 2
    ///
    /// 1 time step (state + observation)
    /// 0/0
    /// 2/1
    ///
    /// 2 time steps (states only)
    /// 0, 0
    /// 0, 1
    /// 0, 2
    /// 0, 2
    /// 2, 0
    /// 2, 1
    /// 2, 1
    /// 2, 2
    ///
    /// 2 time steps (states + observations)
    /// 0/0, 0/0
    /// 0/0, 0/0
    /// 0/0, 1/0
    /// 0/0, 1/1
    /// 0/0, 2/1
    /// 0/0, 2/1
    /// 0/0, 2/1
    /// 0/0, 2/1
    /// 2/1, 0/0
    /// 2/1, 0/0
    /// 2/1, 1/0
    /// 2/1, 1/0
    /// 2/1, 1/1
    /// 2/1, 1/1
    /// 2/1, 2/1
    /// 2/1, 2/1
    lazy_static! {
        static ref HMM_FANCY: HMM = {
            HMM::new(
                array![[0.25, 0.25, 0.5], [0.5, 0.25, 0.25], [0.25, 0.5, 0.25]],
                array![[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
                array![0.5, 0.0, 0.5],
            )
        };
    }

    /// 1 time step (state + observation)
    /// 0/0
    /// 0/0
    /// 0/1
    /// 0/1
    /// 1/0
    /// 1/0
    /// 1/0
    /// 1/1
    ///
    /// 2 time steps (states only)
    /// 0, 0
    /// 0, 1
    /// 0, 1
    /// 0, 1
    /// 1, 0
    /// 1, 0
    /// 1, 1
    /// 1, 1
    ///
    /// 2 time steps (states + observations)

    lazy_static! {
        static ref HMM_COOL: HMM = {
            HMM::new(
                array![[0.25, 0.75], [0.5, 0.5]],
                array![[0.5, 0.5], [0.75, 0.25]],
                array![0.5, 0.5],
            )
        };
    }

    #[test]
    fn sampler_unit() {
        let rng = &mut new_rng();
        let mut sampler = HMM_UNIT.sampler(rng);
        assert_eq!(HMMSample { x: 0, y: 0 }, sampler.next().unwrap());
        assert_eq!(HMMSample { x: 0, y: 0 }, sampler.next().unwrap());
    }

    #[test]
    fn sampler_periodic() {
        let rng = &mut new_rng();
        let mut sampler = HMM_PERIODIC.sampler(rng);
        assert_eq!(HMMSample { x: 0, y: 1 }, sampler.next().unwrap());
        assert_eq!(HMMSample { x: 1, y: 0 }, sampler.next().unwrap());
        assert_eq!(HMMSample { x: 0, y: 1 }, sampler.next().unwrap());
    }

    #[test]
    fn ll_given_states_empty() {
        assert!((0.0 - HMM_UNIT.ll_given_states(&[], &[])).abs() < f64::EPSILON)
    }

    #[test]
    fn ll_given_states_one() {
        assert!((0.0 -  HMM_UNIT.ll_given_states(&[0], &[0])).abs() < f64::EPSILON)
    }

    #[test]
    fn ll_given_states_certain() {
        assert!((0.0 - HMM_PERIODIC.ll_given_states(&[0, 1, 0], &[1, 0, 1])).abs() < f64::EPSILON)
    }

    #[test]
    fn ll_given_states_impossible_initial_state() {
        let r = HMM_PERIODIC.ll_given_states(&[1], &[0]);
        assert!(r.is_infinite() & r.is_sign_negative())
    }

    #[test]
    fn ll_given_states_impossible_transition() {
        let r = HMM_PERIODIC.ll_given_states(&[0, 0], &[1, 1]);
        assert!(r.is_infinite() & r.is_sign_negative())
    }

    #[test]
    fn ll_given_states_impossible_observation() {
        let r = HMM_PERIODIC.ll_given_states(&[0], &[0]);
        assert!(r.is_infinite() & r.is_sign_negative())
    }

    #[test]
    fn filter_empty() {
        assert_eq!(
            Vec::<HMMFilterItem>::new(),
            HMM_FANCY.filter(std::iter::empty()).collect_vec()
        )
    }

    #[test]
    fn filter_zero() {
        assert_eq!(
            vec![HMMFilterItem {
                p_states: array![1.0, 0.0, 0.0]
            }],
            HMM_FANCY.filter(vec![0]).collect_vec(),
            "must be in state 0 because we can't start in state 1 and state 2 can't emit 0"
        )
    }

    #[test]
    fn filter_one() {
        assert_eq!(
            vec![HMMFilterItem {
                p_states: array![0.0, 0.0, 1.0]
            }],
            HMM_FANCY.filter(vec![1]).collect_vec(),
            "must be in state 2 because we can't start in state 1 and state 0 can't emit 1"
        )
    }

    /// Calculated by hand using the probability table for HMM_FANCY
    #[test]
    fn filter_zero_zero() {
        assert_eq!(
            vec![
                HMMFilterItem {
                    p_states: array![1.0, 0.0, 0.0]
                },
                HMMFilterItem {
                    p_states: array![2.0 / 3.0, 1.0 / 3.0, 0.0]
                },
            ],
            HMM_FANCY.filter(vec![0, 0]).collect_vec(),
        )
    }

    /// Calculated by hand using the probability table for HMM_FANCY
    #[test]
    fn filter_zero_one() {
        assert_eq!(
            vec![
                HMMFilterItem {
                    p_states: array![1.0, 0.0, 0.0]
                },
                HMMFilterItem {
                    p_states: array![0.0, 1.0 / 5.0, 4.0 / 5.0]
                },
            ],
            HMM_FANCY.filter(vec![0, 1]).collect_vec(),
        )
    }

    /// Calculated by hand using the probability table for HMM_FANCY
    #[test]
    fn filter_one_zero() {
        assert_eq!(
            vec![
                HMMFilterItem {
                    p_states: array![0.0, 0.0, 1.0]
                },
                HMMFilterItem {
                    p_states: array![0.5, 0.5, 0.0]
                },
            ],
            HMM_FANCY.filter(vec![1, 0]).collect_vec(),
        )
    }

    /// Calculated by hand using the probability table for HMM_FANCY
    #[test]
    fn filter_one_one() {
        assert_eq!(
            vec![
                HMMFilterItem {
                    p_states: array![0.0, 0.0, 1.0]
                },
                HMMFilterItem {
                    p_states: array![0.0, 0.5, 0.5]
                },
            ],
            HMM_FANCY.filter(vec![1, 1]).collect_vec(),
        )
    }

    /// Calculated by hand by looking at the transition matrix
    #[test]
    fn predict_zero_steps() {
        assert_eq!(
            array![0.5, 0.5, 0.0],
            HMM_FANCY.predict(array![0.5, 0.5, 0.0], 0)
        )
    }

    /// Calculated by hand by looking at the transition matrix
    #[test]
    fn predict_one_step() {
        assert_eq!(
            array![0.375, 0.25, 0.375],
            HMM_FANCY.predict(array![0.5, 0.5, 0.0], 1)
        )
    }

    /// Calculated by hand by looking at the transition matrix
    #[test]
    fn predict_two_steps() {
        assert_eq!(
            array![0.3125, 0.34375, 0.34375],
            HMM_FANCY.predict(array![0.5, 0.5, 0.0], 2)
        )
    }

    /// Sample a sequence from this HMM that produces the given sequence of observations. This will
    /// only work well for few observations.
    fn sample_matching<R: Rng>(
        hmm: &HMM,
        observations: &Array1<usize>,
        rng: &mut R,
    ) -> Array1<HMMSample> {
        for _ in 0..10000 {
            let samples = hmm
                .sampler(rng)
                .take(observations.len())
                .collect::<Array1<_>>();
            let matches = observations
                .iter()
                .zip(&samples)
                .all(|(&observation, sample)| observation == sample.y);
            if matches {
                return samples;
            }
        }
        panic!("That was an unlikely sequence of observations")
    }

    /// Calculate an approximation of smoothing by sampling. Useful for testing `smooth`.
    fn smooth_sampled(hmm: &HMM, observations: &Array1<usize>, n_iterations: usize) -> Array2<f64> {
        let mut rng = new_rng();
        let mut counts = Array2::zeros((observations.len(), hmm.n()));
        for _ in 0..n_iterations {
            let samples = sample_matching(hmm, observations, &mut rng);
            for (t, sample) in samples.iter().enumerate() {
                counts[(t, sample.x)] += 1.0;
            }
        }

        counts.normalize_rows()
    }

    /// Calculate an approximation of most likely sequence by sampling. Useful for testing
    /// `most_likely_sequence`.
    fn most_likely_sequence_sampled(
        hmm: &HMM,
        observations: &Array1<usize>,
        n_iterations: usize,
    ) -> Array1<usize> {
        let mut rng = new_rng();
        let (sequence, count) = dbg!(repeat_with(|| sample_matching(hmm, observations, &mut rng))
            .take(n_iterations)
            .collect::<Counter<_>>()
            .most_common())
        .remove(0);
        assert_that(&count).is_greater_than(0);
        sequence.map(|hmm_sample| hmm_sample.x)
    }

    fn test_smooth(observations: Array1<usize>, n_iterations: usize) {
        let expected = smooth_sampled(&HMM_COOL, &observations, n_iterations);
        let actual = HMM_COOL.smooth(&observations);
        let distance = expected.l2_distance(&actual);
        assert_that(&distance).is_less_than(0.3);
    }

    #[test]
    fn test_smooth_zero() {
        test_smooth(array![0], 100);
    }

    #[test]
    fn test_smooth_one() {
        test_smooth(array![0], 100);
    }

    #[test]
    fn test_smooth_zero_zero() {
        test_smooth(array![0, 0], 100);
    }

    #[test]
    fn test_smooth_zero_one() {
        test_smooth(array![0, 1], 100);
    }

    #[test]
    fn test_smooth_zero_zero_zero() {
        test_smooth(array![0, 0, 0], 10000);
    }

    #[test]
    fn test_smooth_zero_one_one() {
        test_smooth(array![0, 1, 1], 10000);
    }

    #[test]
    fn test_train_n_1_k_1_no_observations() {
        let observations = array![];
        let hmm = HMM::train(&observations, 1, 1, &mut new_rng());
        assert_eq!(hmm.a, array![[1.0]]);
        assert_eq!(hmm.b, array![[1.0]]);
        assert_eq!(hmm.pi, array![1.0]);
    }

    #[test]
    fn test_train_n_2_k_2_no_observations() {
        let observations = array![];
        let hmm = HMM::train(&observations, 2, 2, &mut new_rng());
        assert_eq!(hmm.a, array![[0.5, 0.5], [0.5, 0.5]]);
        assert_eq!(hmm.b, array![[0.5, 0.5], [0.5, 0.5]]);
        assert_eq!(hmm.pi, array![0.5, 0.5]);
    }

    // An HMM with one state that always emits the same thing
    #[test]
    fn test_train_n_1_k_1_constant() {
        let observations = array![0, 0];
        let hmm = HMM::train(&observations, 1, 1, &mut new_rng());
        assert_eq!(hmm.a, array![[1.0]]);
        assert_eq!(hmm.b, array![[1.0]]);
        assert_eq!(hmm.pi, array![1.0]);
    }

    // An HMM with one state that always emits the same thing
    #[test]
    fn test_train_n_1_k_2_constant() {
        let observations = array![0, 0];
        let hmm = HMM::train(&observations, 1, 2, &mut new_rng());
        assert_eq!(hmm.a, array![[1.0]]);
        assert_eq!(hmm.b, array![[1.0, 0.0]]);
        assert_eq!(hmm.pi, array![1.0]);
    }

    // An HMM with one state that emits one of two things randomly
    #[test]
    fn test_train_n_1_k_2_random() {
        let observations = array![0, 1];
        let hmm = HMM::train(&observations, 1, 2, &mut new_rng());
        assert_eq!(hmm.a, array![[1.0]]);
        assert_eq!(hmm.b, array![[0.5, 0.5]]);
        assert_eq!(hmm.pi, array![1.0]);
    }

    /// This HMM should switch between its two states each time step
    #[test]
    fn test_train_n_2_k_2() {
        let observations = array![0, 1, 0];
        let hmm = HMM::train(&observations, 2, 2, &mut new_rng());

        // There are two solutions to this MLE problem
        assert_eq!(hmm.a, array![[0.0, 1.0], [1.0, 0.0]]);
        if hmm.pi == array![0.0, 1.0] {
            assert_eq!(hmm.b, array![[0.0, 1.0], [1.0, 0.0]]);
        } else {
            assert_eq!(hmm.b, array![[1.0, 0.0], [0.0, 1.0]]);
            assert_eq!(hmm.pi, array![1.0, 0.0]);
        }
    }

    #[test]
    fn test_viterbi_empty() {
        let ys = array![];
        assert_eq!(HMM_FANCY.most_likely_sequence(&ys), array![]);
    }

    #[test]
    fn test_viterbi_0() {
        let ys = array![0];
        assert_eq!(HMM_FANCY.most_likely_sequence(&ys), array![0]);
    }

    #[test]
    fn test_viterbi_1() {
        let ys = array![1];
        assert_eq!(HMM_FANCY.most_likely_sequence(&ys), array![2]);
    }

    #[test]
    fn test_viterbi_0_0() {
        let ys = array![0, 0];
        assert_eq!(
            HMM_FANCY.most_likely_sequence(&ys),
            most_likely_sequence_sampled(&HMM_FANCY, &ys, 1000)
        );
    }

    #[test]
    fn test_viterbi_0_1() {
        let ys = array![0, 1];
        assert_eq!(
            HMM_FANCY.most_likely_sequence(&ys),
            most_likely_sequence_sampled(&HMM_FANCY, &ys, 1000)
        );
    }

    #[test]
    fn test_viterbi() {
        let ys = array![0, 1, 0, 1];
        assert_eq!(
            HMM_FANCY.most_likely_sequence(&ys),
            most_likely_sequence_sampled(&HMM_FANCY, &ys, 10000)
        );
    }
}

#[cfg(feature = "benchmark")]
mod benchmark {
    use crate::*;
    use test::Bencher;

    #[bench]
    fn bench(b: &mut Bencher) {
        let mut rng = new_rng();
        let observations = [0, 1].into_iter().cycle().take(1001).cloned().collect();
        b.iter(|| {
            HMM::train(&observations, 1, 2, &mut rng);
        });
    }
}
