# 🧠 Nested Learning Framework (HOPE + CMS + Meta-Optimizer)

This repository implements a **self-referential nested learning system** inspired by hierarchical optimization, associative memory, and meta-learning principles.  
The model combines **working memory**, **continuum memory**, and **meta-optimizers** into a unified architecture capable of **learning how to learn**.

---

## 📚 Theoretical Background

### 1. Overview

Traditional neural networks learn static weights through backpropagation with a fixed optimizer.  
In contrast, **Nested Learning** introduces multiple *hierarchies* of learning processes:

| Level | Function | Mechanism |
|:------|:----------|:----------|
| **Level 1–2** | *Fast learning / Working memory* | Self-modifying HOPE layers with linear attention |
| **Level 3** | *Slow weights / Model optimizer* | Gradient-based updates generated dynamically |
| **Level 4** | *Meta-optimizer* | Learns how to generate the learning rules themselves |

This architecture attempts to model **the meta-cognitive hierarchy** of intelligent systems: learning within learning.

---

### 2. Core Components

#### 🧩 HOPE Layer (Hierarchical Optimized Predictive Encoder)
The **HOPE** layer represents a self-referential neural block that combines:
- **Linear Attention** (for associative, differentiable working memory)
- **Continuum Memory System (CMS)** for multi-timescale state retention
- **Self-Modification** subnetwork that learns how to update its own parameters

Equation (conceptual):
\[
h_{t+1} = h_t + f_{\text{self}}(f_{\text{cms}}(f_{\text{attn}}(h_t)))
\]

---

#### 🔁 Continuum Memory System (CMS)
Models temporal memory as multiple *frequencies* or *timescales*:
\[
M_{i}(t+1) = M_i(t) + \Delta_i(h_t)
\]
Each memory level updates at a different frequency (`1`, `10`, `100`), capturing both fast and slow temporal patterns.

---

#### 🔄 Linear Attention (Associative Memory)
Implements a **differentiable associative memory**:
\[
M_t = M_{t-1} + \eta v_t k_t^T
\]
\[
y_t = M_t q_t
\]
where \(q, k, v\) are queries, keys, and values respectively.  
This allows information storage and retrieval similar to attention but linear in sequence length.

---

#### ⚙️ Meta-Optimizer (Deep Momentum + Newton-Schulz Normalization)
Instead of fixed learning rules (e.g., Adam, SGD), the system trains a **meta-optimizer** that generates its own:
- Learning rate (`η`)
- Momentum coefficient (`β`)

Based on observed context:
\[
[\eta_t, \beta_t] = f_{\text{meta}}(C_t)
\]
where \(C_t\) summarizes loss, gradient norms, and their temporal trends.

It uses:
- **Deep Momentum Network:** nonlinear mapping from gradient history → preconditioned direction  
- **Newton-Schulz Normalization:** stabilizes updates by approximating orthonormality.

---

#### 🧮 Context Flow Tracker
Tracks key statistics per level:
- Loss history
- Gradient norms
- Temporal trends

This provides feedback to the meta-optimizer to adapt its hyperparameters dynamically.

---

## 🏗️ Code Structure


> **Note:** The single `main.py` file in this project combines all modules for simplicity.

---

## ⚙️ How It Works (Step-by-Step)

### Step 1 — Forward Pass
1. Input → HOPE layers (self-modifying attention + CMS memory)
2. HOPE output → Output projection
3. Model produces prediction and loss

### Step 2 — Inner Optimization
- Gradients are computed (`create_graph=True`) to allow higher-order differentiation.
- Meta-optimizer observes loss trends and generates new learning rules (`η_t`, `β_t`).

### Step 3 — Meta-Learning
- A *differentiable update* of model parameters is simulated with functional calls.
- The **meta-loss** is computed on the updated parameters (how well the rule worked).
- Meta-optimizer parameters are updated through backpropagation.

### Step 4 — Real Parameter Update
- Actual model weights are updated with the detached gradients using the generated learning rule.
- The process repeats, allowing the system to *learn its own optimizer*.

---

## 🧰 Installation

### Requirements
- Python ≥ 3.9  
- PyTorch ≥ 2.2  
- NumPy (optional for analysis)

Install dependencies:

```bash
pip install torch numpy
git clone https://github.com/andrewwgordon/nested-learning-framework.git
cd nested-learning-framework
python main.py
```
Output will show
```yaml
======================================================================
Nested Learning: Full Implementation (patched)
======================================================================
Step  10 | Loss: 0.0082 | MetaLoss: 0.0079 | LR: 0.000482 | Mom: 0.7410 | GradNorm: 0.2731
Step  20 | Loss: 0.0051 | MetaLoss: 0.0047 | LR: 0.000395 | Mom: 0.8013 | GradNorm: 0.2510
...
======================================================================
Training completed (patched)!
======================================================================
  Test Loss: 0.0042
  Sample Predictions vs Targets:
    Pred:   2.914  |  True:   2.918
    Pred:   0.372  |  True:   0.377
======================================================================
```
## 🔬 Experimentation

You can adjust the following constants near the bottom of main.py:

INPUT_SIZE = 4
OUTPUT_SIZE = 1
HIDDEN_SIZE = 32
LEARNING_STEPS = 100


To scale up:

Increase HIDDEN_SIZE for richer internal dynamics

Extend LEARNING_STEPS for longer meta-training

Modify CMS frequencies for different memory update rates

## 🧩 Key Design Features
Feature	Description
Hierarchical learning	Each level learns rules for the level below it
Self-modifying layers	HOPE layers adapt their internal transformations
Linear attention	Scales efficiently with sequence length
Deep momentum meta-optimizer	Learns nonlinear update rules
Multi-timescale CMS	Retains long- and short-term information
Differentiable inner loop	Enables higher-order optimization
## ⚠️ Notes on Differentiability

This implementation uses retain_graph=True and create_graph=True carefully to enable meta-gradients.
The patched version avoids in-place tensor mutations that break autograd (e.g., in LinearAttention).

If debugging autograd issues, enable anomaly detection:

torch.autograd.set_detect_anomaly(True)

## 🧠 Conceptual Summary

This code simulates a recursive optimizer that:

Learns tasks,

Learns how to improve its own learning,

Learns how to improve that improvement process.

The result is a nested self-referential architecture loosely inspired by ideas in:

Meta-Learning (Hochreiter, 2001)

Predictive Coding & HOPE (Lotter et al., 2016)

Deep Memory Networks (Graves et al., 2016)

Hypernetworks (Ha et al., 2017)

Differentiable Optimizers (Andrychowicz et al., 2016)

## 🧾 Citation

If you use this framework for research or experiments, please cite as:

@software{nested_learning_framework_2025,
  author = {Andrew Gordon},
  title = {Nested Learning Framework: Hierarchical Optimizers and Self-Referential Memory Systems},
  year = {2025},
  url = {https://github.com/andrewwgordon/nested-learning-framework},
  version = {1.0.0}
}

## 🧩 License

MIT License — feel free to modify and extend for research or educational use.