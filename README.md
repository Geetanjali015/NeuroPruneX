# Self-Pruning Neural Network
### Tredence AI Engineering Intern — Case Study Submission

---

## Problem
Build a feed-forward neural network that **prunes its own weights during training** using learnable sigmoid gates and an L1 sparsity regularisation loss, evaluated on CIFAR-10.

---

## Repository Structure

```
.
├── self_pruning_network.py   # Complete implementation (run this)
├── report.md                 # Analysis report with results table
└── README.md                 # This file
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <repo-name>

# 2. Install dependencies
pip install torch torchvision matplotlib

# 3. Run training (CIFAR-10 downloads automatically)
python self_pruning_network.py
```

Outputs are saved to `./outputs/`:
- Gate distribution plots per λ
- Accuracy & sparsity curves
- Best model checkpoint (`best_model.pth`)

---

## Key Components

| Component | Location | Description |
|-----------|----------|-------------|
| `PrunableLinear` | `self_pruning_network.py` | Custom layer with learnable weight gates |
| `SelfPruningNet` | `self_pruning_network.py` | 4-layer network using PrunableLinear |
| `sparsity_loss()` | `self_pruning_network.py` | L1 norm of all gate values |
| `train_model()` | `self_pruning_network.py` | Training loop with combined loss |
| Report | `report.md` | Analysis, results table, λ trade-off |

---

## Results Summary

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|------------|------------------|--------------------|
| 1e-5 (Low) | ~52–54 | ~10–20 |
| 1e-4 (Medium) | ~48–51 | ~40–60 |
| 1e-3 (High) | ~40–45 | ~75–90 |

Higher λ → more pruning → lower accuracy. See `report.md` for full analysis.

---

## How It Works

```
Total Loss = CrossEntropyLoss + λ × Σ sigmoid(gate_scores)
```

Each weight is multiplied by `sigmoid(gate_score) ∈ (0,1)`. The L1 penalty provides a constant gradient push driving gate scores to −∞, collapsing gates to 0 and effectively removing weights. Unlike L2, L1's constant gradient reaches exactly zero — enabling true sparsity.

---

## Requirements

- Python 3.8+
- PyTorch ≥ 1.12
- torchvision
- matplotlib
