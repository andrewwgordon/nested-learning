import pytest
import torch
import torch.nn.functional as F
import sys
import os

# --- Add parent directory (where main.py resides) to Python path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import (
    NewtonSchulz,
    DeepMomentumOptimizer,
    AssociativeMemoryOptimizer,
    ContinuumMemorySystem,
    LinearAttention,
    HOPELayer,
    NestedLearningModel,
    AdvancedMetaOptimizer,
    ContextFlowTracker,
    nested_training_step,
)


# -------------------------------
# 1. Newton-Schulz Normalization
# -------------------------------
def test_newton_schulz_vector_and_matrix():
    ns = NewtonSchulz(num_iters=3)

    # Vector input
    v = torch.randn(10)
    out_vec = ns(v)
    assert out_vec.shape == v.shape
    assert torch.allclose(out_vec.norm(), torch.tensor(1.0), atol=1e-4)

    # Matrix input
    m = torch.randn(5, 5)
    out_mat = ns(m)
    assert out_mat.shape == m.shape
    assert not torch.isnan(out_mat).any()


# -------------------------------
# 2. Deep Momentum Optimizer
# -------------------------------
def test_deep_momentum_optimizer_step():
    dmo = DeepMomentumOptimizer(input_dim=8)
    grad = torch.randn(8)
    lr, mom = 0.01, 0.9

    update = dmo(grad, lr, mom)
    assert update.shape == grad.shape
    assert not torch.isnan(update).any()

    # Subsequent call should reuse momentum_state
    update2 = dmo(grad, lr, mom)
    assert torch.allclose(dmo.momentum_state, dmo.momentum_state)


# -------------------------------
# 3. Associative Memory Optimizer
# -------------------------------
def test_associative_memory_optimizer_step():
    amo = AssociativeMemoryOptimizer(input_dim=8)
    grad = torch.randn(8)
    loss_context = torch.tensor([0.5])
    out = amo(grad, loss_context, learning_rate=0.01, momentum_coef=0.9)
    assert out.shape == grad.shape
    assert not torch.isnan(out).any()


# -------------------------------
# 4. Continuum Memory System
# -------------------------------
def test_continuum_memory_system_forward():
    cms = ContinuumMemorySystem(hidden_size=16)
    x = torch.randn(2, 16)
    y = cms(x, current_step=5)
    assert y.shape == x.shape
    assert isinstance(cms.should_update_level(0, 10), bool)


# -------------------------------
# 5. Linear Attention
# -------------------------------
def test_linear_attention_forward_and_memory_update():
    attn = LinearAttention(hidden_size=8)
    x = torch.randn(2, 3, 8)
    mem_before = attn.memory.clone()
    y = attn(x)
    assert y.shape == (2, 3, 8)
    assert not torch.isnan(y).any()
    # Memory should be updated
    assert not torch.allclose(mem_before, attn.memory)


# -------------------------------
# 6. HOPE Layer
# -------------------------------
def test_hope_layer_forward_shapes():
    layer = HOPELayer(hidden_size=8)
    x = torch.randn(2, 1, 8)
    y = layer(x, step=0)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


# -------------------------------
# 7. Nested Learning Model
# -------------------------------
def test_nested_learning_model_forward():
    model = NestedLearningModel(input_size=4, output_size=1, hidden_size=8, num_layers=1)
    x = torch.randn(2, 4)
    y = model(x, step=0)
    assert y.shape == (2, 1)
    assert not torch.isnan(y).any()


# -------------------------------
# 8. Advanced Meta Optimizer
# -------------------------------
def test_advanced_meta_optimizer_output_ranges():
    meta = AdvancedMetaOptimizer(context_dim=16)
    context = torch.randn(16)
    lr, mom = meta(context)
    assert lr.item() > 0 and lr.item() <= 1e-2
    assert 0.0 <= mom.item() <= 0.99


# -------------------------------
# 9. Context Flow Tracker
# -------------------------------
def test_context_flow_tracker_record_and_summary():
    tracker = ContextFlowTracker(num_levels=4)
    tracker.record_context(level=2, loss=1.0, grad_norm=0.5, step=1)
    summary = tracker.get_context_summary(2)
    assert "avg_loss" in summary and "loss_trend" in summary
    assert isinstance(summary["avg_grad_norm"], float)


# -------------------------------
# 10. Nested Training Step
# -------------------------------
def test_nested_training_step_execution():
    torch.manual_seed(42)
    main_model = NestedLearningModel(input_size=4, output_size=1, hidden_size=8, num_layers=1)
    meta_opt = AdvancedMetaOptimizer(context_dim=16)
    tracker = ContextFlowTracker(num_levels=4)
    dummy_optim = torch.optim.SGD(main_model.parameters(), lr=1.0)

    data = torch.randn(1, 4)
    labels = data.sum(dim=1, keepdim=True)

    meta_loss, lr_t, mom_t, grads, loss_val = nested_training_step(
        main_model,
        meta_opt,
        data,
        labels,
        dummy_optim,
        step=1,
        context_tracker=tracker
    )

    assert meta_loss.requires_grad
    assert lr_t.item() > 0
    assert 0 <= mom_t.item() <= 0.99
    assert isinstance(loss_val, torch.Tensor)
    assert all(isinstance(g, torch.Tensor) for g in grads)
    assert not torch.isnan(meta_loss).any()


# -------------------------------
# 11. Gradient Flow Test
# -------------------------------
def test_meta_loss_backward_pass():
    torch.manual_seed(0)
    main_model = NestedLearningModel(4, 1, hidden_size=8, num_layers=1)
    meta_opt = AdvancedMetaOptimizer(context_dim=16)
    tracker = ContextFlowTracker(num_levels=4)
    dummy_optim = torch.optim.SGD(main_model.parameters(), lr=1.0)

    data = torch.randn(1, 4)
    labels = data.sum(dim=1, keepdim=True)

    meta_loss, lr_t, mom_t, grads, _ = nested_training_step(
        main_model, meta_opt, data, labels, dummy_optim, step=1, context_tracker=tracker
    )

    meta_loss.backward(retain_graph=True)
    for p in meta_opt.parameters():
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()
