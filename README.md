# TRPO in TensorFlow

A TensorFlow implementation of **Trust Region Policy Optimization** ([Schulman et al., ICML 2015](https://arxiv.org/abs/1502.05477)), packaged as a drop-in **custom optimizer** rather than a monolithic training script.

The whole TRPO update lives in `trpo.ConjugateGradientOptimizer`, a subclass of `tf.train.Optimizer`. You give it your policy and cost tensors plus a set of cache variables, and it performs the constrained natural-gradient step entirely inside the TensorFlow graph.

## What the optimizer does

Inside `apply_gradients`, for each update it:

1. Flattens the policy-gradient `g` over all trainable variables.
2. Builds a **Hessian-vector product** `Hx_fn` of the mean KL divergence (Fisher information), using `tf.gradients` of `∇KL · m`.
3. Solves `Hx = g` with **conjugate gradient** (`cg`, default 10 iterations) to get the natural-gradient direction `x`.
4. Computes the maximal step `β = sqrt(2·δ / xᵀHx)` from the trust-region size `δ` (`delta`).
5. Runs a **backtracking line search** (`ls_max_iter`, shrink ratio `back_trace_ratio`) that restores parameters from cache and shrinks `β` until the KL constraint holds and the surrogate cost improves; NaN steps are rejected.

Cache variables (`var + 'cache'`) snapshot the parameters so the line search can roll back failed steps.

## Files

| File | Role |
| --- | --- |
| `trpo.py` | The `ConjugateGradientOptimizer` (TRPO step) + a tiny `TestModel` self-test |
| `train-atari.py` | Batch-A3C-style training driver that plugs TRPO in as the optimizer (adapted from tensorpack's A3C example by Yuxin Wu) |
| `simulator.py` | Multiprocess environment simulator (client/server over IPC) |
| `common.py` | Evaluation helpers (`Evaluator`, `eval_model_multithread`, `play_n_episodes`) |
| `atari_wrapper.py` | Gym environment wrappers (frame stack, fire-reset, episode length limit, etc.) |

## Usage

Built on [tensorpack](https://github.com/tensorpack/tensorpack), TensorFlow 1.x, and OpenAI Gym. The optimizer is wired up in `Model.optimizer()`:

```python
opt = trpo.ConjugateGradientOptimizer(self.policy, self.cost, delta=0.1)
```

Train on the default `CartPole-v0` (discrete actions) environment:

```bash
python train-atari.py --task train --env CartPole-v0 --gpu 0
```

Evaluate, play, or dump videos from a checkpoint:

```bash
python train-atari.py --task eval  --env CartPole-v0 --load /path/to/model
python train-atari.py --task play  --env CartPole-v0 --load /path/to/model
```

You can also run `trpo.py` directly to exercise the optimizer on a fake-data `TestModel`.

## Notes & scope

- Works with **discrete action spaces**; extending to continuous (Gaussian) policies is straightforward — swap the policy head and KL term.
- This is a TensorFlow 1.x / tensorpack-era project and depends on `tf.contrib`, so it needs that legacy stack to run.

## License

See [`LICENSE`](LICENSE).
