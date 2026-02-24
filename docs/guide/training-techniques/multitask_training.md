# Multitask Training

NequIP framework models are typically trained to predict multiple targets simultaneously, such as energies and forces, and sometimes additional properties like stresses or custom targets for specialized applications.

The standard approach to multitask training is to combine multiple loss components using a weighted sum.
Each target (energy, forces, stress, etc.) contributes to the total loss with a user-defined coefficient that controls its relative importance during training.
For detailed information on configuring loss functions and coefficients, see the [Loss Functions and Metrics](../configuration/metrics.md) guide, particularly the sections on [simplified wrapper classes](../configuration/metrics.md#simplified-wrappers) and [coefficient configuration](../configuration/metrics.md#coefficients-and-weighted-sum).

## Advanced Multitask Training Strategies

NequIP provides several advanced techniques organized into training modules and callbacks.

### Training Modules

Training modules are configured in the [`training_module` section](../configuration/config.md#training_module) of your config file:

- **{class}`~nequip.train.ConFIGLightningModule`** - Implements the Conflict-free inverse gradient (ConFIG) approach to multitask learning, which optimizes gradient conflicts between different tasks by solving a linear system to find optimal update directions. See [ConFIG paper](https://arxiv.org/abs/2408.11104).

- **{class}`~nequip.train.EMAConFIGLightningModule`** - Combines the ConFIG approach with exponential moving averages for enhanced stability in multitask scenarios.

### Callbacks

Callbacks are configured in the [`trainer` section](../configuration/config.md#trainer) of your config file and provide dynamic behavior during training:

- **{class}`~nequip.train.callbacks.LossCoefficientScheduler`** - A callback that dynamically adjusts loss coefficients during training based on predefined schedules, allowing you to emphasize different targets at different stages of training.

- **{class}`~nequip.train.callbacks.LossCoefficientMonitor`** - A callback for tracking and logging loss coefficients over time, useful for monitoring how coefficient scheduling affects training dynamics.

- **{class}`~nequip.train.callbacks.SoftAdapt`** - An adaptive callback that automatically adjusts loss coefficients based on the relative rate of learning of different tasks. See [SoftAdapt paper](https://www.sciencedirect.com/science/article/pii/S0927025624003768).

## Multi-Head Training

Multi-head training allows a single model to learn from multiple datasets that share the same chemical system but use different levels of theory (e.g. DFT and RPA, or different DFT functionals). The model shares its representation layers (embedding, message-passing) across all heads but uses separate readout networks (MLP + per-type scale/shift) for each head.

### When to use multi-head training

Multi-head training is useful when:

- You have a large, inexpensive dataset (e.g. PBE-D3 with energies, forces, and stresses) and a smaller, expensive dataset (e.g. RPA or CCSD(T) with energies only)
- Both datasets describe the same chemical system and atom types
- You want the expensive-level head to benefit from the structural information learned by the cheaper-level head

### Configuration

Multi-head training requires changes to three parts of the config: **data**, **model**, and **trainer callbacks**.

#### Data

Each dataset must be stamped with a head index using the {class}`~nequip.data.transforms.HeadStamper` transform. Provide multiple datasets as a list under `train_dataset` (and optionally `val_dataset`), and set `combined_loader_mode` to control how datasets of different sizes are iterated together:

```yaml
data:
  _target_: nequip.data.datamodule.NequIPDataModule
  seed: ${seed}
  combined_loader_mode: max_size_cycle  # or min_size

  train_dataset:
    # Head 0: e.g. DFT data with energies, forces, stresses
    - _target_: nequip.data.dataset.ASEDataset
      file_path: dft_train.xyz
      transforms:
        - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
          model_type_names: ${model_type_names}
        - _target_: nequip.data.transforms.NeighborListTransform
          r_max: ${cutoff_radius}
        - _target_: nequip.data.transforms.HeadStamper
          head_index: 0

    # Head 1: e.g. higher-level data with energies only
    - _target_: nequip.data.dataset.ASEDataset
      file_path: rpa_train.xyz
      transforms:
        - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
          model_type_names: ${model_type_names}
        - _target_: nequip.data.transforms.NeighborListTransform
          r_max: ${cutoff_radius}
        - _target_: nequip.data.transforms.HeadStamper
          head_index: 1

  val_dataset:
    # Validation set for head 0
    - _target_: nequip.data.dataset.ASEDataset
      file_path: dft_val.xyz
      transforms:
        - _target_: nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper
          model_type_names: ${model_type_names}
        - _target_: nequip.data.transforms.NeighborListTransform
          r_max: ${cutoff_radius}
        - _target_: nequip.data.transforms.HeadStamper
          head_index: 0

  train_dataloader: ${dataloader}
  val_dataloader: ${dataloader}
  stats_manager:
    _target_: nequip.data.CommonDataStatisticsManager
    type_names: ${model_type_names}
```

The `combined_loader_mode` controls how the dataloaders are iterated:
- `max_size_cycle` (recommended): cycles through the smaller dataset(s) until the largest one is exhausted, ensuring the model sees all data from every head each epoch.
- `min_size`: stops when the smallest dataset is exhausted.

```{important}
The `head_index` in {class}`~nequip.data.transforms.HeadStamper` must correspond to the position of the head name in the `head_names` list in the model config. For example, if `head_names: [dft, rpa]`, then the DFT dataset should use `head_index: 0` and the RPA dataset `head_index: 1`.
```

#### Model

Enable multi-head by specifying `head_names` and per-head energy shifts/scales in the model config:

```yaml
training_module:
  model:
    _target_: nequip.model.NequIPGNNModel
    # ... standard hyperparameters (l_max, num_layers, etc.) ...

    head_names: [dft, rpa]

    per_type_energy_shifts:
      dft:
        H: -13.587
        O: -431.267
      rpa:
        H: -12.718
        O: -421.420

    per_type_energy_scales:
      dft: ${training_data_stats:forces_rms}
      rpa: 1.0  # energy-only head: do NOT use forces_rms (see warning below)
```

Each head gets its own {class}`~nequip.nn.ScalarMLP` readout and {class}`~nequip.nn.PerTypeScaleShift`, while the embedding and message-passing layers are shared.

```{tip}
It is recommended to use isolated atom energies from each level of theory as the per-head energy shifts. Since different methods can have very different absolute energies, using a single `per_atom_energy_mean` for all heads forces the network to absorb large energy offsets, which hurts training.
```

`per_type_energy_scales` and `per_type_energy_shifts` each accept a dict mapping head names to either a single value (broadcast to all types) or a dict mapping type names to values. Use the special key `all` to broadcast the same value to every head:

```yaml
    per_type_energy_scales:
      all: ${training_data_stats:forces_rms}
```

#### Per-head loss weights

If the datasets are imbalanced or you want to prioritize one head, use `per_head_loss_weights` in the `training_module`:

```yaml
training_module:
  per_head_loss_weights:
    "0": 1.0   # keys are string indices matching dataloader order
    "1": 5.0   # upweight the second head
```

```{note}
The keys in `per_head_loss_weights` are the string dataloader indices (`"0"`, `"1"`, ...), not the head names. This matches the keys used by Lightning's {class}`~lightning.pytorch.utilities.combined_loader.CombinedLoader`.
```

#### Trainer callbacks

For multi-head models, the `ModelCheckpoint` callback should monitor a validation metric from one of the heads. Validation metrics are prefixed with `val{i}_epoch/` where `i` is the validation dataset index:

```yaml
trainer:
  callbacks:
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      monitor: val0_epoch/weighted_sum   # monitor first val dataset
      dirpath: ${hydra:runtime.output_dir}
      filename: best
      save_last: true
```

### Data preparation for energy-only heads

When a head has no force labels (energy-only training), the force and stress fields must still be present in the data but populated with NaN values. This allows {class}`~nequip.train.EnergyForceStressLoss` with ``ignore_nan`` to skip these entries during loss computation.

```{important}
{class}`~nequip.train.EnergyForceLoss` does **not** support ``ignore_nan``. For multi-head training with energy-only heads, you must use {class}`~nequip.train.EnergyForceStressLoss` even if you don't need stress predictions. Set the stress coefficient to ``0.0`` or ``null``.
```

```{important}
The stress field must be present in the data even when ``ignore_nan: {stress: true}`` is set — a missing field will cause a ``KeyError``. Populate stress labels with NaN in your data preprocessing if stress is not available.
```

Non-periodic structures (e.g. gas-phase clusters) require a finite dummy cell because {class}`~nequip.nn.ForceStressOutput` computes stress via ``virial / volume``. A zero cell produces ``inf``. Add a large dummy cell (e.g. ``100 * eye(3)``) with ``pbc=False`` during data preparation.

### Force quality for energy-only heads

A common multi-head scenario is training one head with energies and forces (e.g. DFT/PBE) and another with energies only (e.g. RPA or hybrid DFT). While the energy-only head can learn accurate total energies, its autograd forces may be poor because the per-atom energy decomposition is unconstrained: the total energy provides only one constraint per frame, but N atoms have N-1 unconstrained degrees of freedom. The readout MLP can redistribute energy among atoms freely without affecting the total, corrupting the position gradients.

```{warning}
**Energy scales for energy-only heads**: do **not** use ``forces_rms`` from a force-supervised head as the ``per_type_energy_scales`` for an energy-only head. This causes the readout to operate at very small values, degrading the numerical quality of autograd forces and producing unstable molecular dynamics. For energy-only heads, use ``1.0`` (no scaling) or the energy standard deviation of that head's dataset. This is a critical setting for delta-learning workflows.
```

```{tip}
Use the default linear readout (``readout_mlp_hidden_layers_depth: 0``) for multi-head models where some heads are energy-only. In our testing, adding hidden layers (depth > 0) degraded force quality for energy-only heads. The linear readout preserves a tighter coupling between backbone features and per-atom energies, which is beneficial when forces must come purely from autograd.
```

```{important}
Standard validation metrics (energy MAE, force MAE) are not sufficient to assess whether an energy-only head will produce physically meaningful forces for molecular dynamics. A model with better validation metrics can produce catastrophically wrong MD trajectories (e.g. wrong density or structural collapse). Always validate delta-learning models with short NPT simulations before production use.
```

### Packaging and compilation

Multi-head models follow the standard [workflow](../getting-started/workflow.md) with one additional step: you select which head to compile for deployment.

**Package** the full multi-head model (all heads are preserved):

```bash
nequip-package build path/to/best.ckpt model.nequip.zip
```

**Inspect** the package to see available heads:

```bash
nequip-package info model.nequip.zip
```

**Compile** a single head for deployment using `--head`:

```bash
nequip-compile model.nequip.zip dft.nequip.pt2 \
  --mode aotinductor --device cuda --target ase --head dft

nequip-compile model.nequip.zip rpa.nequip.pt2 \
  --mode aotinductor --device cuda --target ase --head rpa
```

The compiled model is a standard single-head model and can be used with any [integration](../../integrations/all.rst) (ASE, LAMMPS, etc.) without any multi-head awareness downstream.

**Compile a summed model** for delta-learning deployment using `+`:

```bash
# Deploy E_dft + E_rpa_delta as a single model
nequip-compile model.nequip.zip target.nequip.pt2 \
  --mode aotinductor --device cuda --target ase --head dft+rpa_delta
```

When `+` is used, each head's readout and scale/shift pipeline runs independently and the resulting per-atom energies are summed. This is useful for delta-learning workflows where the base head (e.g. DFT with forces) and a correction head (e.g. RPA−DFT, energy-only) are trained jointly, and the deployed model should predict the target-level energy surface ``E_base + E_delta``. Forces are computed via autograd of the summed energy.

```{important}
Compiling a multi-head model without `--head` will raise an error listing the available heads. Conversely, using `--head` on a single-head model will also raise an error.
```
