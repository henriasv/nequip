# Changes to Support Multi-Head Extension Packages

This document describes the changes in `henriasv/nequip@feature/parameterized-modifiers` that enable the `nequip-multihead` extension package. These are minimal, non-breaking additions to the NequIP infrastructure.

## 1. Parameterized Modifiers in `nequip-compile`

**File:** `nequip/scripts/compile.py`

**What:** The `--modifiers` flag now supports `key=value` arguments, parsed by grouping tokens:

```bash
nequip-compile model.ckpt out.pt2 --modifiers extract_head head_name=dft
```

This produces `{"modifier": "extract_head", "head_name": "dft"}` which `modify()` passes as kwargs. The `modify()` infrastructure already supported kwargs — this change exposes it through the CLI.

**Why:** Extension packages define model modifiers (e.g. `extract_head` for multi-head models) that need arguments. Without this, `--modifiers` only supported argument-free modifiers like `enable_OpenEquivariance`.

**Backward compatible:** Existing argument-free modifiers work unchanged.

## 2. Modifiers in Model Builder Config

**File:** `nequip/model/utils.py`

**What:** The `@model_builder` wrapper accepts an optional `modifiers` list in the model config, applied after model construction but before `GraphModel` wrapping:

```yaml
model:
  _target_: nequip.model.NequIPGNNModel
  modifiers:
    - enable_OpenEquivariance
```

**Why:** Enables training-time modifiers (e.g. OEQ acceleration) via config without modifying model code. Also allows extension packages to apply their own modifiers during model construction.

**Backward compatible:** `modifiers` is popped from kwargs before passing to the builder function. Existing configs without `modifiers` work unchanged.

## 3. EMA Checkpoint Fix

**File:** `nequip/train/ema.py`

**What:** `EMALightningModule.on_save_checkpoint()` detects and corrects the swapped EMA weight state before checkpoint saving. `EMAWeights.set_extra_state()` tolerates old checkpoints saved in the swapped state.

**Why:** Lightning's `ModelCheckpoint.on_validation_end` callback fires *before* `LightningModule.on_validation_end`. During validation, EMA weights are swapped into the model. `ModelCheckpoint` saves `best.ckpt` while the weights are swapped, producing a checkpoint where `is_holding_ema_weights=False`. Loading this checkpoint for packaging or compilation fails with:

```
AssertionError: EMA module loaded in a state where it does not contain EMA weights
```

**Reproduction:**

```bash
# Train any model with EMALightningModule
nequip-train --config-dir=. --config-name=config

# Try to package best.ckpt
nequip-package build best.ckpt model.nequip.zip
# → AssertionError
```

**Hook ordering verification:**

```python
# Lightning 2.6.1: callback fires before module
CALLBACK on_validation_end    ← ModelCheckpoint saves here (swapped)
MODULE on_validation_end      ← EMA swaps back here (too late)
```

**Backward compatible:** `on_save_checkpoint` is a no-op when weights are not swapped. Old checkpoints are handled via `_needs_post_load_swap` flag.

## 4. Small Molecule FX Tracing Fix

**File:** `nequip/utils/fx.py`

**What:** `nequip_make_fx` ensures at least 3 atoms remain in the augmented batch used for FX shape validation.

**Why:** For very small molecules (e.g. 3-atom water monomers), the previous code removed `max(2, ceil(N*0.1))` = 2 atoms, leaving 1 atom with 0 edges. This caused the FX tracer to specialize on zero-edge shapes, failing the shape comparison check:

```
RuntimeError: the fx'ed models for different input shapes do not agree
```

**Reproduction:**

```bash
# Train on 3-atom water monomers
nequip-train --config-dir=. --config-name=water_monomer_config

# Try to compile
nequip-compile last.ckpt model.pt2 --mode aotinductor --device cuda --target ase
# → RuntimeError (FX shape mismatch)
```

**Backward compatible:** Only affects molecules with ≤ 5 atoms. Larger systems are unaffected.

## 5. MetricsManager Tensor Caching

**File:** `nequip/train/metrics_manager.py`

**What:** `MetricsManager.forward()` stores live loss tensors (with grad) in `metrics_tensors_step`, alongside the existing `metrics_values_step` (detached scalars).

**Why:** The `GradientNormFractionScheduler` callback in `nequip-multihead` needs access to individual loss component tensors (with autograd graph attached) to compute per-component gradient norms. The existing `metrics_values_step` stores `.item()` values which don't have grad.

**Backward compatible:** `metrics_tensors_step` is an additional cache that doesn't affect existing behavior. Only consumed by extension callbacks that opt in.
