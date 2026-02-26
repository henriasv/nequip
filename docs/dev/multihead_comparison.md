# Multi-Head Training: NequIP vs. MACE Comparison

## Introduction

NequIP's multi-head training feature is inspired by MACE's multi-head approach. Both share the same high-level concept — a shared backbone with multiple readout heads — but differ in several architecturally significant ways that affect performance, memory, gradient flow, and scaling. This document describes these differences for developers working on or extending the feature.

> **Note:** This document is developer/contributor documentation and is not wired into the Sphinx toctree.

## Architecture Overview

| Aspect | MACE | NequIP |
|---|---|---|
| Readout | Single wider MLP producing `[n_atoms, n_heads]` | Separate `ScalarMLP` + `PerTypeScaleShift` per head |
| Head selection | `mask_head()` zeros non-active heads, then fancy-index `[arange, node_heads]` | Loop over heads, stack, fancy-index `[arange, node_heads, :]` |
| Batching | `ConcatDataset` (mixed-head single batch) | Lightning `CombinedLoader` (dict of per-head batches) |
| Scale/shift | `ScaleShiftBlock` — fixed buffers, one scalar per head, not per-type | `PerTypeScaleShift` — trainable parameters, per-element per-head |
| E0 (reference energies) | `AtomicEnergiesBlock` — `[n_elements, n_heads]` buffer, matmul + index | Folded into per-head `PerTypeScaleShift` shifts |
| Head extraction | N/A | `extract_head()` strips a single head; `extract_summed_heads()` sums multiple heads into one deployable model |

## Conceptual Differences That Impact Performance

### Forward pass: loop vs. single wider layer

NequIP runs N sequential forward passes through the readout (one per head), each on a `data.copy()`. Cost scales linearly with head count.

MACE runs one forward pass through a wider readout layer. The `mask_head()` operation is cheap relative to an extra MLP forward.

**Practical impact:** Negligible at 2 heads, significant at 5+ heads. MACE's approach is more GPU-friendly (single GEMM vs. N sequential GEMMs).

**Code refs:**
- `nequip/nn/multihead.py:130-143` — per-head loop, stack, and fancy-index
- `mace/modules/blocks.py:107-114` — single forward with `mask_head()` call
- `mace/modules/irreps_tools.py:111-116` — `mask_head()` implementation

### Gradient flow through non-selected heads

- **NequIP:** Non-selected heads are simply not executed → zero compute wasted, but also zero gradient signal for unused head parameters on that sample.
- **MACE:** All heads are computed in the wider layer, but `mask_head()` zeros non-selected outputs → gradients for non-selected head weights are zero (due to multiplication by zero mask), but the shared hidden layer still sees gradients routed through the selected head's portion.

**Practical impact:** Similar in practice — in both cases, only the selected head's readout parameters receive non-zero gradients. The backbone receives gradients from the selected head in both cases. No meaningful convergence difference expected from this alone.

### Batching strategy: CombinedLoader vs. ConcatDataset

NequIP's `CombinedLoader` yields one batch per head per step. The training loop iterates over the dict, calling `self(head_batch)` for each head. The backbone sees one head's data distribution at a time within a step, but the accumulated loss includes all heads.

MACE's `ConcatDataset` shuffles all heads together. A single batch contains mixed-head samples processed in one forward pass.

**Practical impact:** NequIP's approach guarantees balanced head representation per step (one batch from each head). MACE's random sampling can lead to imbalanced batches, especially when head datasets differ in size. However, MACE's mixed batches are more computationally efficient (one forward pass vs. N).

**Code refs:**
- `nequip/data/datamodule/_base_datamodule.py:284-295` — `CombinedLoader` construction
- `nequip/train/lightning.py:265-301` — multi-head training step iterating over head batches
- `mace/cli/run_train.py:679` — `ConcatDataset` construction from per-head train sets

### Energy normalization: trainable per-type vs. fixed scalar

This is arguably the most consequential difference.

- **NequIP:** Each head has its own `PerTypeScaleShift` with **trainable** per-element scales and shifts (`nequip/nn/atomwise.py:116-195`). The model can learn different energy zero-points and scale corrections for each element under each head.
- **MACE:** `ScaleShiftBlock` stores **fixed buffers** (non-trainable) with one scalar scale and one scalar shift per head — no per-element resolution (`mace/modules/blocks.py:1329-1344`). Per-element E0s are handled separately in `AtomicEnergiesBlock`, also a fixed buffer (`mace/modules/blocks.py:307-331`).

**Practical impact:** NequIP's trainable per-type shifts can adapt during training, potentially improving convergence for systems where different heads have element-dependent energy offsets. MACE relies on accurate pre-computed statistics. For heads with very different energy scales (e.g. HF vs. MP2), NequIP's approach is more flexible; MACE's approach is simpler and avoids the risk of overfitting normalization parameters.

### Memory

- **NequIP:** `data.copy()` per head duplicates the data dict (shallow copy of tensors, but still N dict allocations and N separate MLP activations in the autograd graph).
- **MACE:** Single pass, wider intermediate tensors, no data copying.

**Practical impact:** MACE uses less peak memory. The difference grows with head count.

## Scaling Summary

| Heads | NequIP forward cost | MACE forward cost |
|---|---|---|
| 1 | 1x (identical path) | 1x |
| 2 | ~2x readout | ~1.1x readout |
| 5 | ~5x readout | ~1.3x readout |
| 10 | ~10x readout | ~1.6x readout |

Backbone cost is identical and dominates; these are readout-only multipliers.

## Design Rationale

Why NequIP chose separate modules per head:

- **Simpler to implement** within NequIP's `GraphModuleMixin` / `SequentialGraphNetwork` architecture
- **Enables `extract_head()`** — trivially extract one head's modules into a standalone model (`nequip/model/extract_head.py`)
- **Per-head `PerTypeScaleShift`** gives finer-grained, trainable normalization
- **Backward compatible** — the single-head path is unchanged

## Force Transfer Considerations

### The per-atom decomposition non-uniqueness problem

In multi-head training, an energy-only head (e.g. RPA) learns total energies but never receives force supervision. Because the total energy is a sum of per-atom contributions, and only the sum is constrained, there are N-1 unconstrained degrees of freedom in the per-atom decomposition for each N-atom frame. The independent per-head readout MLP can redistribute energy among atoms freely without affecting the training loss, but this redistribution corrupts the position gradients (autograd forces).

This is a fundamental limitation of energy-only training with flexible per-atom energy models, not specific to NequIP's architecture.

### Shared readout (`shared_readout=True`)

When `shared_readout=True`, `MultiHeadReadout` creates:
1. A **shared `ScalarMLP`** that produces base per-atom energies — this module receives gradients from ALL heads
2. Per-head **correction `ScalarMLP`s** that learn the residual

The forward pass becomes: `per_atom_energy = shared_readout(features) + correction_head(features)`, followed by per-head `PerTypeScaleShift` as before.

**Gradient flow:** Force supervision from head 0 (e.g. DFT) constrains the shared readout's weight matrix, imposing a physically meaningful per-atom decomposition. Head 1 (e.g. RPA) inherits this decomposition through the shared readout, and its correction MLP only needs to learn the (small) difference.

**Head extraction:** When extracting a head from a shared-readout model via `extract_head()`, the shared readout and per-head correction are fused:
- For depth-0 (linear) readouts: weights are algebraically combined into a single linear layer via `_fuse_linear_readouts()`
- For deeper readouts: a `SharedPlusCorrectionReadout` wrapper module sums both outputs

**Code refs:**
- `nequip/nn/multihead.py` — `shared_readout_mode` flag, shared + correction forward
- `nequip/model/extract_head.py` — `_fuse_linear_readouts()`, `SharedPlusCorrectionReadout`

### Head summing for deployment (`extract_summed_heads`)

For delta-learning workflows, `extract_summed_heads(model, ["base", "delta"])` creates a single model that sums the per-atom energies from multiple heads. The `SummedHeadsReadout` wrapper runs each head's readout → scale_shift pipeline independently and sums the results. Forces are obtained via autograd of the summed energy. This is invoked at compile time via `nequip-compile --head base+delta`.

**Code refs:**
- `nequip/model/extract_head.py` — `SummedHeadsReadout`, `extract_summed_heads()`
- `nequip/scripts/compile.py` — `+` syntax parsing in `--head`

### Cross-head force regularization (`cross_head_force_reg`)

This adds a regularization term to the training loss: `lambda * MSE(F_target_head, F_reference_head.detach())`, computed on the reference head's training data with the head index swapped to the target head.

**How it works in `training_step`:**
1. The normal per-head loss loop runs, caching each head's output
2. For each target head, the reference head's batch is cloned with the `HEAD_KEY` changed to the target head index
3. A fresh forward pass computes the target head's forces on the reference head's structures
4. The MSE between target and (detached) reference forces is added to the total loss

**Key detail:** `ref_forces.detach()` ensures only the target head receives gradients from this term. The reference head's readout is unaffected.

**Code refs:**
- `nequip/train/lightning.py` — `cross_head_force_reg` config, regularization in `training_step`
