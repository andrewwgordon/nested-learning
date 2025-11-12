import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
import math

# This file implements the Nested Learning framework (patched to avoid inplace ops
# that break autograd when using create_graph / retained graphs).

try:
    from torch.func import functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call


# --- 1. Newton-Schulz Normalization (Paper Section 2.3) ---
class NewtonSchulz(nn.Module):
    """
    Newton-Schulz iteration for matrix normalization.
    Used as non-linear output function in optimizers (Equation 24).
    """
    def __init__(self, num_iters=5):
        super().__init__()
        self.num_iters = num_iters

    def forward(self, x):
        """Apply Newton-Schulz normalization."""
        if x.dim() == 1:
            return x / (torch.norm(x) + 1e-7)

        # For matrices, approximate orthonormalization
        y = x / (torch.norm(x) + 1e-7)
        for _ in range(self.num_iters):
            y = 1.5 * y - 0.5 * y @ y.T @ y
        return y


# --- 2. Deep Momentum Optimizer (Paper Section 2.3) ---
class DeepMomentumOptimizer(nn.Module):
    """
    Deep Momentum Gradient Descent (DMGD) - Equation 23.
    Uses MLP to compress gradient history instead of linear momentum.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # Multi-layer momentum memory
        self.momentum_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        # Non-linear output (Newton-Schulz normalization)
        self.output_nonlinearity = NewtonSchulz(num_iters=3)
        self.momentum_state = None

    def forward(self, gradient, learning_rate, momentum_coef):
        """
        Compute momentum update using deep network.

        Args:
            gradient: Current gradient
            learning_rate: Scalar learning rate
            momentum_coef: Momentum coefficient
        """
        grad_flat = gradient.flatten()

        # Initialize momentum state if needed
        if self.momentum_state is None:
            self.momentum_state = torch.zeros_like(grad_flat)

        # Deep momentum: learn non-linear compression
        momentum_update = self.momentum_net(grad_flat)

        # Update momentum state
        self.momentum_state = momentum_coef * self.momentum_state + momentum_update

        # Apply non-linearity
        output = self.output_nonlinearity(self.momentum_state)

        return -learning_rate * output.reshape(gradient.shape)


# --- 3. Associative Memory Optimizer (Paper Section 2.1-2.3) ---
class AssociativeMemoryOptimizer(nn.Module):
    """
    Optimizer as associative memory that maps gradients to preconditioners.
    Implements L2 regression objective (Equations 19-20, 27).
    """
    def __init__(self, input_dim):
        super().__init__()
        # Preconditioner network (learns value mappings)
        self.preconditioner = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, input_dim)
        )
        self.momentum_buffer = None

    def forward(self, gradient, loss_context, learning_rate, momentum_coef):
        """
        Delta rule update with learned preconditioning (Equations 22, 28-29).

        Args:
            gradient: Parameter gradient
            loss_context: Context information (loss value)
            learning_rate: Scalar LR
            momentum_coef: Momentum coefficient
        """
        grad_flat = gradient.flatten()

        # Initialize momentum if needed
        if self.momentum_buffer is None:
            self.momentum_buffer = torch.zeros_like(grad_flat)

        # Learn preconditioner based on context (P_i in Equation 20)
        context_embedding = torch.cat([
            loss_context.view(-1),
            torch.tensor([grad_flat.norm().item()]).to(gradient.device)
        ])
        preconditioner = self.preconditioner(grad_flat)

        # Delta rule: Equation 22
        # m_t+1 = (αI - ∇L^T∇L)m_t - ηP∇L
        # Simplified version to avoid memory issues with large matrices
        delta_rule_term = momentum_coef * self.momentum_buffer - learning_rate * preconditioner

        # Update momentum with delta rule
        self.momentum_buffer = delta_rule_term

        return self.momentum_buffer.reshape(gradient.shape)


# --- 4. Continuum Memory System (Paper Section 3) ---
class ContinuumMemorySystem(nn.Module):
    """
    Multi-frequency memory hierarchy (CMS) - Equation 30-31.
    Each MLP updates at different rates representing different timescales.
    """
    def __init__(self, hidden_size, frequencies=[1, 10, 100]):
        super().__init__()
        self.frequencies = frequencies
        self.chunk_sizes = [max(frequencies) // f for f in frequencies]

        # Multiple MLP blocks, one per frequency level
        self.memory_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size),
            ) for _ in frequencies
        ])

        self.step_counter = 0
        self.accumulated_grads = [None] * len(frequencies)

    def forward(self, x, current_step=None):
        """
        Chain MLPs with different update frequencies.

        Args:
            x: Input tensor
            current_step: Current training step for frequency-based updates
        """
        if current_step is not None:
            self.step_counter = current_step

        output = x

        # Chain through memory blocks at different frequencies
        for i, (block, chunk_size) in enumerate(zip(self.memory_blocks, self.chunk_sizes)):
            # Residual connection
            output = output + block(output)

        return output

    def should_update_level(self, level_idx, step):
        """Check if a memory level should update at this step."""
        chunk_size = self.chunk_sizes[level_idx]
        return step % chunk_size == 0


# --- 5. Linear Attention as Associative Memory ---
class LinearAttention(nn.Module):
    """
    Linear attention as gradient descent on associative memory (Equations 13-16).
    Implements the inner optimization process.

    IMPORTANT: avoid mutating self.memory inside the autograd graph.
    We'll compute outputs using a local memory and then update the
    persistent buffer under torch.no_grad() afterwards.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=False)

        # Memory state (updated via gradient descent on dot-product loss)
        # register as buffer so it moves with device/model
        self.register_buffer('memory', torch.zeros(hidden_size, hidden_size))

    def forward(self, x):
        """
        Forward pass with memory update.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        q = self.wq(x)  # [batch, seq_len, hidden_size]
        k = self.wk(x)
        v = self.wv(x)

        outputs = []

        # Use a local copy of memory for differentiable computation
        local_memory = self.memory.clone()

        # Compute outputs using local_memory (so we don't mutate the buffer
        # that autograd might be tracking)
        for t in range(seq_len):
            qt = q[:, t, :]  # [batch, hidden_size]
            kt = k[:, t, :]
            vt = v[:, t, :]

            # Inner optimization: update local memory (this participates in graph)
            # local_memory is a new tensor, so accumulation here is safe
            for b in range(batch_size):
                local_memory = local_memory + torch.outer(vt[b], kt[b]) * 0.01

            # Output: y_t = M_t q_t
            out = torch.matmul(local_memory, qt.unsqueeze(-1)).squeeze(-1)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)

        # After computing outputs (graph built), update persistent buffer
        # without touching graph tensors by using detached versions in no_grad.
        # Recompute updates under no_grad using detached v/k to avoid interfering
        # with the computational graph.
        with torch.no_grad():
            for t in range(seq_len):
                kt = k[:, t, :].detach()
                vt = v[:, t, :].detach()
                for b in range(batch_size):
                    # persistent buffer updated in-place under no_grad
                    self.memory += torch.outer(vt[b], kt[b]) * 0.01

        return outputs


# --- 6. HOPE Layer (Self-Referential Learning) ---
class HOPELayer(nn.Module):
    """
    HOPE: Self-modifying layer combining working memory + CMS.
    Learns its own update rule (Section 3).
    """
    def __init__(self, hidden_size, num_memory_levels=3):
        super().__init__()
        self.hidden_size = hidden_size

        # Working memory (fast, token-level processing)
        self.attention = LinearAttention(hidden_size)

        # Continuum Memory System (multi-frequency hierarchy)
        self.cms = ContinuumMemorySystem(
            hidden_size,
            frequencies=[1, 10, 100]
        )

        # Self-referential: network learns how to modify itself
        self.self_modifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, step=0):
        """
        Forward pass with self-modification.

        Args:
            x: Input tensor
            step: Current training step
        """
        # Working memory processes context
        attended = self.attention(x)
        x = self.norm1(x + attended)

        # Continuum memory at multiple timescales
        memory_out = self.cms(x, step)
        x = self.norm2(x + memory_out)

        # Self-modification: learn parameter updates
        update = self.self_modifier(x)

        # Apply learned update
        return x + update


# --- 7. Main Model with Nested Levels ---
class NestedLearningModel(nn.Module):
    """
    Multi-level nested learning model.
    Implements the full hierarchy from the paper.
    """
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Level 1-2: HOPE layers (fast processing + working memory)
        self.hope_layers = nn.ModuleList([
            HOPELayer(hidden_size) for _ in range(num_layers)
        ])

        # Level 3: Output projection (slow weights)
        self.output_proj = nn.Linear(hidden_size, output_size)

        # Track update frequencies for each component
        self.update_frequencies = {
            'hope': 1,      # Updates every step
            'output': 1,    # Updates every step
        }

    def forward(self, x, step=0):
        """
        Forward pass through nested levels.

        Args:
            x: Input tensor [batch, input_size]
            step: Current training step
        """
        # Project to hidden dimension
        h = self.input_proj(x.unsqueeze(1))  # [batch, 1, hidden]

        # Process through HOPE layers
        for layer in self.hope_layers:
            h = layer(h, step)

        # Project to output
        output = self.output_proj(h.squeeze(1))

        return output


# --- 8. Meta-Optimizer (Level 4) ---
class AdvancedMetaOptimizer(nn.Module):
    """
    Enhanced meta-optimizer that generates update rules.
    Includes deep momentum and richer context processing.
    """
    def __init__(self, context_dim=16):
        super().__init__()
        # Process richer context (not just loss)
        self.context_processor = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Generate learning rate
        self.lr_generator = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Generate momentum coefficient
        self.momentum_generator = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Deep momentum optimizer
        self.deep_momentum = None

    def forward(self, context_summary):
        """
        Generate update rules based on context.

        Args:
            context_summary: Tensor with stats
        """
        # Process context
        processed = self.context_processor(context_summary)

        # Generate learning rate (positive via softplus)
        lr = F.softplus(self.lr_generator(processed)) + 1e-8
        lr = torch.clamp(lr, min=1e-8, max=1e-2)

        # Generate momentum (0-1 via sigmoid)
        momentum = torch.sigmoid(self.momentum_generator(processed))
        momentum = torch.clamp(momentum, min=0.0, max=0.99)

        return lr.squeeze(), momentum.squeeze()


# --- 9. Context Flow Tracker ---
class ContextFlowTracker:
    """
    Track context flows at each nesting level.
    Each level has its own gradient flow and optimization context.
    """
    def __init__(self, num_levels=4):
        self.num_levels = num_levels
        self.contexts = [[] for _ in range(num_levels)]

    def record_context(self, level, loss, grad_norm, step):
        """Record context summary for a level."""
        self.contexts[level].append({
            'step': step,
            'loss': loss,
            'grad_norm': grad_norm,
        })

    def get_context_summary(self, level, window_size=10):
        """Get recent context summary for a level."""
        recent = self.contexts[level][-window_size:]
        if not recent:
            return {
                'avg_loss': 0.0,
                'avg_grad_norm': 0.0,
                'loss_trend': 0.0
            }

        avg_loss = sum(c['loss'] for c in recent) / len(recent)
        avg_grad_norm = sum(c['grad_norm'] for c in recent) / len(recent)

        # Simple trend: difference between recent and older
        if len(recent) >= 2:
            loss_trend = recent[-1]['loss'] - recent[0]['loss']
        else:
            loss_trend = 0.0

        return {
            'avg_loss': avg_loss,
            'avg_grad_norm': avg_grad_norm,
            'loss_trend': loss_trend
        }


# --- 10. Nested Training Loop ---
def nested_training_step(
    main_model,
    meta_optimizer,
    data,
    labels,
    level3_optim,
    step,
    context_tracker
):
    """
    Perform nested learning step with multiple levels.
    Uses retain_graph=True strategically to allow multiple backward passes.

    Returns:
        meta_loss: Loss after meta-generated update
        lr_t, momentum_t: Generated hyperparameters
        grads_for_update: Detached gradients for actual parameter update
        loss_value: Detached loss for logging
    """
    # Forward pass to compute inner loss
    output = main_model(data, step=step)
    loss = F.mse_loss(output, labels)
    loss_value = loss.item()

    # Compute gradients with create_graph=True and retain_graph=True
    # This allows us to backprop through the gradient computation later
    params_list = list(main_model.parameters())
    grads = torch.autograd.grad(
        loss,
        params_list,
        create_graph=True,
        allow_unused=True,
        retain_graph=True  # KEEP the graph for meta-learning
    )

    # Create detached copies for the actual parameter update
    # Clone to avoid any reference issues
    grads_for_update = []
    for g, p in zip(grads, params_list):
        if g is not None:
            grads_for_update.append(g.detach().clone())
        else:
            grads_for_update.append(torch.zeros_like(p))
    grads_for_update = tuple(grads_for_update)

    # Compute gradient norm for context
    grad_norm = torch.norm(torch.cat([g.flatten() for g in grads_for_update]))

    # Build context summary (all detached values)
    context_stats = context_tracker.get_context_summary(level=3)
    context_tensor = torch.tensor([
        loss_value,
        grad_norm.item(),
        context_stats['avg_loss'],
        context_stats['avg_grad_norm'],
        context_stats['loss_trend'],
        float(step),
        float(math.log(step + 1)),
        float(math.sin(step * 0.1)),
        float(math.cos(step * 0.1)),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ], device=data.device, requires_grad=False)

    # Meta-optimizer generates rules based on context
    lr_t, momentum_t = meta_optimizer(context_tensor)

    # Build updated parameters using the DIFFERENTIABLE gradients
    # IMPORTANT: Don't modify parameters in-place during this phase
    new_param_dict = {}
    for (name, param), grad in zip(main_model.named_parameters(), grads):
        if grad is None:
            # No gradient - keep parameter unchanged
            new_param_dict[name] = param
            continue

        state = level3_optim.state.get(param, {})
        prev_buf = state.get('momentum_buffer', torch.zeros_like(param))

        # CRITICAL: Clone prev_buf and detach to avoid inplace modification issues
        prev_buf_detached = prev_buf.detach().clone()

        # Differentiable momentum update
        # Create NEW tensors instead of modifying existing ones
        new_buf = momentum_t * prev_buf_detached + grad
        new_param = param - lr_t * new_buf

        new_param_dict[name] = new_param

    # Evaluate meta-loss on functionally updated parameters
    # functional_call creates a temporary model with new parameters
    next_output = functional_call(main_model, new_param_dict, (data,), kwargs={'step': step})
    meta_loss = F.mse_loss(next_output, labels)

    # Record context
    context_tracker.record_context(
        level=3,
        loss=loss_value,
        grad_norm=grad_norm.item(),
        step=step
    )

    return meta_loss, lr_t, momentum_t, grads_for_update, torch.tensor(loss_value)


# --- 11. Main Simulation ---
INPUT_SIZE = 4
OUTPUT_SIZE = 1
HIDDEN_SIZE = 32
LEARNING_STEPS = 100

# Initialize models
main_model = NestedLearningModel(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE)
meta_optimizer = AdvancedMetaOptimizer(context_dim=16)
context_tracker = ContextFlowTracker(num_levels=4)

# Optimizers
meta_optim = optim.Adam(meta_optimizer.parameters(), lr=1e-5, weight_decay=1e-6)
main_optim_dummy = optim.SGD(main_model.parameters(), lr=1.0)


def main():
    print("=" * 70)
    print("Nested Learning: Full Implementation (patched)")
    print("=" * 70)

    # Enable anomaly detection for debugging (can be disabled in production)
    # torch.autograd.set_detect_anomaly(True)

    for step in range(LEARNING_STEPS):
        # Generate synthetic data
        data = torch.randn(1, INPUT_SIZE)
        labels = data.sum(dim=1, keepdim=True) + torch.randn(1, 1) * 0.01

        # Nested learning step
        # This returns detached gradients for the actual parameter update
        meta_loss, current_lr, current_momentum, grads_for_update, inner_loss = nested_training_step(
            main_model,
            meta_optimizer,
            data,
            labels,
            main_optim_dummy,
            step,
            context_tracker
        )

        # Update meta-optimizer (Level 4)
        meta_optim.zero_grad()
        try:
            # Backprop through meta-loss (graph retained inside nested_training_step)
            meta_loss.backward(retain_graph=True)
        except RuntimeError as e:
            print(f"Error during backward at step {step}: {e}")
            print("Skipping this step...")
            continue

        meta_grad_norm = clip_grad_norm_(meta_optimizer.parameters(), max_norm=1.0)
        meta_optim.step()

        # Apply generated rules to main model (Level 3)
        # Use the detached gradients returned from nested_training_step (grads_for_update).
        # CRITICAL: Do NOT use tensors that are part of the retained graph for in-place updates.
        with torch.no_grad():
            for (name, param), grad in zip(main_model.named_parameters(), grads_for_update):
                if grad is not None and not grad.eq(0).all():
                    state = main_optim_dummy.state.get(param, {})
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(param.data)
                        main_optim_dummy.state[param] = state

                    buf = state['momentum_buffer']
                    # Use the detached grad (safe) to update momentum buffer and parameter
                    new_buf = current_momentum.item() * buf + grad
                    state['momentum_buffer'] = new_buf
                    # In-place param update (safe inside no_grad)
                    param.add_(state['momentum_buffer'], alpha=-current_lr.item())

        # Logging
        if (step + 1) % 10 == 0:
            context = context_tracker.get_context_summary(level=3)
            print(f"Step {step + 1:3d} | "
                  f"Loss: {inner_loss.item():.4f} | "
                  f"MetaLoss: {meta_loss.item():.4f} | "
                  f"LR: {current_lr.item():.6f} | "
                  f"Mom: {current_momentum.item():.4f} | "
                  f"GradNorm: {context['avg_grad_norm']:.4f}")

    print("\n" + "=" * 70)
    print("Training completed (patched)!")
    print("=" * 70)

    # Final evaluation
    test_data = torch.randn(5, INPUT_SIZE)
    test_labels = test_data.sum(dim=1, keepdim=True)

    with torch.no_grad():
        predictions = main_model(test_data, step=LEARNING_STEPS)
        test_loss = F.mse_loss(predictions, test_labels)
        print(f"  Test Loss: {test_loss.item():.4f}")
        print(f"  Sample Predictions vs Targets:")
        for i in range(min(3, len(test_data))):
            print(f"    Pred: {predictions[i].item():7.3f}  |  True: {test_labels[i].item():7.3f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
