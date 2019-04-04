# 🤔 hmmm 🤔

Hidden Markov Models in Rust.

This library contains a Rust implementation of a time-invariant Hidden Markov model with
discrete observations. It includes maximum likelihood estimation via the Baum-Welch
expectation-maximization algorithm and hidden state inference via the Viterbi algorithm.

See `hmmm::HMM` for detailed documentation on how to work with this library.

Below, the HMM is trained to recognize the pattern `001001001...`

```rust
use hmmm::HMM;
use ndarray::{array, Array1};
use rand::{SeedableRng, XorShiftRng};

fn main() {
    let training_ys = array![0, 0, 1, 0, 0, 1, 0];
    let mut rng = XorShiftRng::seed_from_u64(1337);
    let hmm = HMM::train(&training_ys, 3, 2, &mut rng);
    let sampled_ys: Array1<usize> = hmm.sampler(&mut rng)
        .map(|sample| sample.y)
        .take(10)
        .collect();
    assert_eq!(array![0, 0, 1, 0, 0, 1, 0, 0, 1, 0], sampled_ys);
}
```

### Building

This project uses `cargo-make`. See `Makefile.toml` for a full list of build commands, but the
main useful command for this project is `cargo make all`.

There is a small amount of benchmarking functionality gated by the `benchmark` feature.

### Notes

Sections 17.3 and 17.4 of *Machine Learning a Probabilistic Perspective* by Kevin Murphy, 2012
were invaluable as a reference, as was section 13.2 of *Pattern Recognition and Machine
Learning* by Christopher Bishop, 2016.

I have attempted to make the math notation readable both as rendered HTML and from the source
code. The notation is strongly inspired by the Wikipedia page on the Baum-Welch algorithm.

License: MIT/Apache-2.0
