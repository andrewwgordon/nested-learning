import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# This file implements a minimal nested / meta-learning example. The goal is
# to demonstrate the structure of a nested learning system in a very small
# self-contained script so the core ideas are easy to follow:
#   - MainModel: the inner learner whose parameters are the longer-term memory
#     (Level 3 in the conceptual hierarchy).
#   - MetaOptimizer: a small network producing per-step learning signals
#     (learning rate and momentum) that control how the inner learner updates
#     its parameters (Level 4, the slow meta-learner).
#
# Key implementation choices and simplifications used here:
#  - We implement a single-step differentiable unroll (one inner update) for
#    clarity. This is enough to demonstrate gradients flowing from a "meta"
#    loss back to the meta-parameters.
#  - Inner gradients are computed with `create_graph=True` and used to build a
#    functional updated copy of parameters. The outer (meta) loss is evaluated
#    on that functional copy so gradients flow back to the meta-learner.
#  - To avoid common autograd pitfalls we do NOT apply in-place parameter
#    updates until after the outer backward has completed. Instead we use the
#    detached lr/momentum/grads to perform the real in-place update.
#  - Simple numeric stabilizers are added (clamping meta outputs, low meta lr,
#    gradient clipping) to keep this tiny demo from diverging easily.

# Prefer the newer API when available, fall back to the older location for
# compatibility across PyTorch versions.
try:
    from torch.func import functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call


# --- 1. The Main Model (Levels 1-3) ---
class MainModel(nn.Module):
    """
    Represents the main network whose weights are the long-term memory (Level 3).
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        # Level 1 (fast processing / context) is simulated by a single
        # feed-forward pass. In richer models this could be an attention or
        # recurrent mechanism producing context-dependent activations.
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 2. The Meta-Optimizer (Level 4) ---
class MetaOptimizer(nn.Module):
    """
    The slowest, outer loop component. It learns to generate the update rules
    (learning rate and momentum) for the MainModel.
    """

    def __init__(self):
        super().__init__()
        # Input features: We simplify the 'context summary' to the current loss value (1 input)
        self.fc = nn.Linear(1, 16)

        # Output 1: Learning Rate (eta_t). We use softplus to keep it positive.
        self.output_lr = nn.Linear(16, 1)
        # Output 2: Momentum Term (gamma_t). We use sigmoid to keep it between (0, 1).
        self.output_momentum = nn.Linear(16, 1)

    def forward(self, loss_tensor):
        # The meta-learner receives a compact summary of context. Here we
        # simplify the context to the scalar inner loss. Real systems would
        # provide richer summaries (per-layer grad statistics, activations,
        # time/context embeddings, etc.).
        x = F.relu(self.fc(loss_tensor.view(1, -1)))

        # Generate the update rules (the meta-parameters).
        # We use softplus to ensure a positive learning rate and sigmoid for
        # momentum (in the unit interval). We then clamp values to avoid
        # numerically unstable extremes (a pragmatic choice for a toy demo).
        learning_rate = F.softplus(self.output_lr(x)) + 1e-8
        # Tighter upper bound to avoid large step sizes during unroll.
        learning_rate = torch.clamp(learning_rate, min=1e-8, max=1e-2)

        # Momentum in (0, 1); clamp slightly below 1 for numerical safety.
        momentum_term = torch.sigmoid(self.output_momentum(x))
        momentum_term = torch.clamp(momentum_term, min=0.0, max=0.99)

        # Squeeze to convert from (1,1) tensors to scalars for convenience.
        return learning_rate.squeeze(), momentum_term.squeeze()


# --- 3. The Nested Training Loop Simulation ---


def nested_training_step(main_model, meta_optimizer, data, labels, level3_optim):
    """
    Perform a one-step differentiable unrolled inner loop.

    Returns:
      meta_loss: the loss after applying the generated update to a functional
                 copy of the model (this is used to update the meta-optimizer).
      lr_t, momentum_t: the tensors produced by the meta-optimizer (scalar tensors).
      grads: tuple of gradients of the inner loss w.r.t. each parameter (used to
             apply the real update after meta step).
    """
    # --- Inner Loop: compute loss and gradients with create_graph=True so
    # the meta-optimizer can receive gradients through the update rule.
    #
    # We intentionally avoid calling `loss.backward()` here because that would
    # apply in-place changes to gradient buffers and potentially free parts of
    # the graph. Instead we compute parameter-wise gradients via
    # `autograd.grad(..., create_graph=True)` which returns tensors that remain
    # differentiable (they carry their own graph) so the outer meta-loss can
    # backpropagate through the inner update.
    output = main_model(data)
    loss = F.mse_loss(output, labels)

    # Compute gradients for the inner loss (used in the unrolled update).
    grads = torch.autograd.grad(loss, tuple(main_model.parameters()), create_graph=True)

    # Level 4 (Meta-Optimizer) Generates Rules based on Context (loss).
    # Note: we do NOT detach the loss when passing it to the meta-optimizer,
    # because the meta-parameters should receive gradients from the meta-loss
    # computed below. The meta-learner outputs are therefore part of the
    # differentiable computation graph.
    lr_t, momentum_t = meta_optimizer(loss.view(1, -1))

    # Build the updated parameters after one (differentiable) inner update.
    # We construct a new parameter dictionary rather than writing in-place so
    # autograd can track the computation and the meta-loss can flow back
    # through lr_t/momentum_t.
    new_param_dict = {}
    for (name, param), grad in zip(main_model.named_parameters(), grads):
        # Retrieve any existing momentum buffer from the dummy optimizer
        # state. For the differentiable unroll we treat the previous buffer as
        # a fixed tensor (detach it) so the only learnable influence comes
        # from lr_t and momentum_t.
        state = level3_optim.state.get(param, {})
        prev_buf = state.get("momentum_buffer", torch.zeros_like(param)).detach()

        # new buffer: v_t = momentum * v_{t-1} + grad
        new_buf = momentum_t * prev_buf + grad

        # new param: w_t = w_{t-1} - lr * v_t (functional, not in-place)
        new_param = param - lr_t * new_buf

        new_param_dict[name] = new_param

    # Evaluate the next-step (post-update) loss using the functional copy.
    # This is the meta-loss: it measures the effectiveness of the meta's
    # proposed update and is used to update the meta-learner itself.
    next_output = functional_call(main_model, new_param_dict, (data,))
    meta_loss = F.mse_loss(next_output, labels)

    return meta_loss, lr_t, momentum_t, grads


def sample_and_unroll(
    main_model, meta_optimizer, level3_optim, input_size=None, noise_scale=0.01
):
    """
    Helper that samples a single synthetic datapoint (mini-batch of size 1),
    builds a target, and runs a differentiable one-step unroll.

    This extracts the data/label creation logic out of `main()` so the main
    loop is shorter and the sampling behavior can be reused or tested
    independently.

    Returns the same tuple as `nested_training_step`.
    """
    # Simulate a single mini-batch / streaming datapoint. Kept small for demo
    # speed and deterministic unit tests.
    if input_size is None:
        input_size = INPUT_SIZE

    data = torch.randn(1, input_size)
    # Target is simple: sum of inputs. Add a small amount of noise to make the
    # learning task non-trivial but stable.
    labels = data.sum(dim=1, keepdim=True) + torch.randn(1, 1) * noise_scale

    return nested_training_step(main_model, meta_optimizer, data, labels, level3_optim)


# --- Simulation Setup ---
INPUT_SIZE = 4
OUTPUT_SIZE = 1
LEARNING_TASK = 50  # Number of steps to simulate

# Initialize models
main_model = MainModel(INPUT_SIZE, OUTPUT_SIZE)
meta_optimizer = MetaOptimizer()

# Separate Optimizer for the Level 4 Meta-Optimizer (the 'outer-loop' optimizer)
meta_optim = optim.Adam(meta_optimizer.parameters(), lr=1e-5, weight_decay=1e-6)

# Dummy optimizer for Level 3 weights (used only to hold the momentum buffers)
main_optim_dummy = optim.SGD(main_model.parameters(), lr=1.0)


def main():
    # Print a short header when running the script directly. The heavy
    # simulation is guarded by `if __name__ == '__main__'` so importing this
    # module in tests won't execute the training loop.
    print("Starting Nested Learning Simulation...")

    for step in range(LEARNING_TASK):
        # Simulate a single mini-batch / streaming datapoint. We intentionally
        # keep the batch size very small to keep the demo fast and deterministic
        # for the unit tests. The target is a simple function of the input so
        # we can observe learning quickly.
        data = torch.randn(1, INPUT_SIZE)
        # Target is simple: sum of inputs. Reduce label noise to help stability.
        labels = data.sum(dim=1, keepdim=True) + torch.randn(1, 1) * 0.01

        # Perform the nested step (differentiable unroll). This logic is
        # extracted into `sample_and_unroll` to keep `main()` concise and to
        # make sampling/test harnessing easier.
        meta_loss, current_lr, current_momentum, grads = sample_and_unroll(
            main_model, meta_optimizer, main_optim_dummy
        )

        # Backpropagate the Meta-Loss to update the Meta-Optimizer (Level 4).
        # At this point, `meta_loss` depends on lr_t/momentum_t and therefore on
        # the parameters of `meta_optimizer`. Backpropagating here updates the
        # meta-learner so it can propose better update rules in future steps.
        meta_optim.zero_grad()
        meta_loss.backward()

        # Clip meta-optimizer gradients to avoid very large update steps which
        # can destabilize the small meta-network. This is a pragmatic safety
        # measure for this toy example.
        from torch.nn.utils import clip_grad_norm_

        meta_grad_norm = clip_grad_norm_(meta_optimizer.parameters(), max_norm=1.0)
        meta_optim.step()

        # Apply the generated rules (Level 2 / Level 3 updates) AFTER updating
        # the meta-optimizer. We use detached copies of lr/momentum/grads here
        # because these in-place updates are not part of the differentiable
        # graph (they are the actual parameter changes that will be used on
        # the next iteration). Doing them earlier would corrupt the graph and
        # cause autograd errors.
        with torch.no_grad():
            for (name, param), grad in zip(main_model.named_parameters(), grads):
                if grad is not None:
                    state = main_optim_dummy.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(param.data)

                    buf = state["momentum_buffer"]

                    # v_t = momentum_t * v_{t-1} + grad_t
                    # We detach the tensors used for the in-place update so that
                    # these operations don't interact with the autograd graph.
                    buf.mul_(current_momentum.detach()).add_(grad.detach())

                    # w_t = w_{t-1} - lr_t * v_t
                    param.add_(buf, alpha=-current_lr.detach())

        if (step + 1) % 10 == 0:
            # Provide a bit more diagnostic info (meta grad norm) to detect
            # instabilities early.
            try:
                mg = float(meta_grad_norm)
            except Exception:
                mg = float("nan")
            print(
                f"Step {step + 1:3}: Loss={meta_loss.item():.4f} | "
                f"LR_Generated={current_lr.item():.6f} | "
                f"Momentum_Generated={current_momentum.item():.6f} | "
                f"MetaGradNorm={mg:.4f}"
            )

    print(
        "\nSimulation finished. The Meta-Optimizer has learned to adjust LR and Momentum."
    )


if __name__ == "__main__":
    main()
