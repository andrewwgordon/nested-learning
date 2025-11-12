
# Nested Learning — minimal demo

This repository is a compact, educational implementation of a nested/meta-
learning idea inspired by the paper "Nested Learning: The Illusion of Deep
Learning Architectures". It purposely keeps the code small so the key
mechanisms are explicit and easy to inspect.

This project is not a full reproduction of any large-scale paper; instead it
implements the core mechanism: a fast inner learner whose parameter updates
are produced (and later improved) by a slow meta-learner. The code demonstrates
how to write a differentiable one-step unroll and how to stabilize the outer
optimization so the meta-learner can be trained.

Contents

- `main.py` — The demo implementation. Contains:
	- `MainModel` — a small feed-forward network (Levels 1–3 in the paper's
		hierarchy).
	- `MetaOptimizer` — a small network that generates per-step learning-rate and
		momentum signals (Level 4 in the hierarchy).
	- `nested_training_step` — a differentiable, one-step unrolled inner loop
		implemented with `torch.autograd.grad` and `functional_call`.
- `tests/` — Pytest tests (shape checks, one-step smoke test, and a
	multi-step stability test).
- `requirements.txt` — Python dependencies (torch, pytest, and optional dev
	tools).
- `.github/workflows/ci.yml` — GitHub Actions workflow: runs linters and tests.

Quick start

Install dependencies and run tests:

```bash
pip install -r requirements.txt
pytest -q
```

Run the demo (prints a short training simulation):

```bash
python3 main.py
```

How the code maps to the paper's ideas
--------------------------------------

The paper describes a hierarchical (nested) learning system. This repository
implements a minimal version and maps code components to the conceptual levels
used in that work:

- Level 1 — Fast, transient computation (attention/context): implemented as a
	normal forward pass of `MainModel` (a small feed-forward network). In real
	systems this might be a fast recurrent or attention mechanism that produces
	context-dependent activations.
- Level 2 — Short-term memory / fast weights: represented here by a simple
	momentum buffer that accumulates gradients and is used to update parameters
	within a single outer-step. The buffer is stored in the dummy `main_optim`
	state and updated using the meta-generated momentum.
- Level 3 — Slow, long-term weights: the parameters of `MainModel` are the
	longer-term weights that ultimately store learned knowledge. They are
	updated using the buffers described in Level 2.
- Level 4 — Meta-learner / slow optimizer: `MetaOptimizer` is a small
	feed-forward network that observes a simple context signal (the current
	loss) and outputs two scalars: a learning rate and a momentum. These are the
	meta-parameters that control how Level 2 and Level 3 change.

What the code actually does
---------------------------

1. Inner forward: compute the inner-model prediction and inner loss.
2. Compute inner gradients with `create_graph=True` so the outer meta-loss can
	 backpropagate through those gradients.
3. The meta-learner (`MetaOptimizer`) receives the inner loss and generates a
	 learning-rate and momentum scalar. In this minimal example the meta-learner
	 sees only the loss; richer implementations would provide additional
	 summaries (gradient statistics, activations, time context).
4. We form a differentiable, functional update: build a new set of parameters
	 using `new_param = param - lr * (momentum * prev_buf + grad)` without
	 applying those changes in-place. We then evaluate the next-step loss using
	 `functional_call(main_model, new_param_dict, (data,))`. That next-step loss
	 is the meta-loss used to update the meta-learner: gradients flow from that
	 meta-loss back through the functional update to the meta-parameters.
5. After the meta-optimizer updates, the code applies the real (in-place)
	 parameter updates to `MainModel` using detached values (so we avoid
	 modifying tensors that autograd still needs during the outer backward).

Why we implemented a one-step unroll
-----------------------------------

Fully unrolling many inner steps is expensive in memory and often unnecessary
for small demos. This repository implements a single-step differentiable
unroll which captures the essential pattern: the meta-learner proposes an
update, we evaluate the post-update performance, and we update the meta-learner
based on that performance. Extending to multiple inner steps is straightforward
but more demanding (memory/time) and usually benefits from libraries such as
`higher` or `functorch`.

Stability and implementation details
-----------------------------------

Two practical issues required attention while implementing a minimal, but
working, example:

- "double backward" and in-place modifications: naively computing an inner
	backward and then calling backward again for the outer loss caused runtime
	errors (graph freed or tensors modified in-place). To fix this we:
	- compute inner gradients with `torch.autograd.grad(..., create_graph=True)`
		instead of calling `loss.backward()` in the inner loop, and
	- avoid applying in-place updates to model parameters before the outer
		backward; instead the real updates are applied after the meta optimizer
		step using detached tensors.
- Numerical stability of the meta-controller: an unconstrained meta-learner can
	output extreme learning rates and momentum values that cause exploding loss
	or NaNs. We stabilize training by:
	- clamping the meta outputs to safe ranges (learning rate in [1e-8, 1e-2],
		momentum in [0.0, 0.99]),
	- reducing the meta-optimizer learning rate and adding small weight decay,
	- clipping meta-optimizer gradients before stepping.

These choices are pragmatic: they keep this small demo stable and make it a
useful starting point for experimentation.

Limitations vs a full paper implementation
-----------------------------------------

- The demo uses a very small network and a single-step unroll. Papers that
	explore nested learning typically experiment with deeper models, longer
	unrolls, and richer meta-learner inputs.
- We use a simple scalar context (the loss) for the meta-learner. Real
	meta-learners use many more signals.
- For multi-step unrolls and research-grade experiments, use `higher` or
	`functorch` to avoid manual bookkeeping and to gain performance and clarity.

Tests, CI, and style
--------------------

- `tests/test_main.py` contains unit tests that verify output shapes,
	a one-step end-to-end smoke test, and a multi-step stability test.
- A GitHub Actions workflow runs linters (isort, black, flake8) and tests on
	push/PR to `main`.
- Pre-commit configuration is included so you can run `pre-commit install` and
	have black/isort/flake8 run locally on commits.

How to extend this project
--------------------------

- Replace the scalar context with richer per-parameter summaries or global
	statistics (e.g., gradient norms, layer activations).
- Increase the inner unroll length and add gradient checkpointing or truncated
	unrolls to control memory.
- Replace the manual unroll with `higher` (https://github.com/facebookresearch/higher)
	which makes nested optimization code more concise and less error-prone.
- Experiment with different meta-learner architectures (RNNs, attention,
	hypernetworks) to generate more expressive update rules.

Acknowledgements and license
---------------------------

This example is educational and intentionally minimal. Use it as a starting
point for research or teaching. The repository includes a LICENSE file — read
it for reuse conditions.

If you'd like, I can add a more complete reproduction plan (suggested
architectures, hyperparameters, and experiments) based on the original paper
and help implement a multi-step unroll with `higher`.

References
----------

For background and inspiration, see:

- "Nested Learning: The Illusion of Deep Learning Architectures" — (add full
	citation here). [URL]](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)