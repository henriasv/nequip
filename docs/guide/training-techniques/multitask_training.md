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

This approach is inspired by the multi-head architecture in [MACE-MP](https://arxiv.org/abs/2401.00096), where a shared backbone learns a general representation and individual heads specialize to different energy surfaces.

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

    per_head_energy_shifts:
      dft:
        H: -13.587
        O: -431.267
      rpa:
        H: -12.718
        O: -421.420

    per_head_energy_scales:
      dft: ${training_data_stats:forces_rms}
      rpa: ${training_data_stats:forces_rms}
```

Each head gets its own {class}`~nequip.nn.ScalarMLP` readout and {class}`~nequip.nn.PerTypeScaleShift`, while the embedding and message-passing layers are shared.

```{tip}
It is recommended to use isolated atom energies from each level of theory as the per-head energy shifts. Since different methods can have very different absolute energies, using a single `per_atom_energy_mean` for all heads forces the network to absorb large energy offsets, which hurts training.
```

`per_head_energy_scales` and `per_head_energy_shifts` each accept a dict mapping head names to either a single value (broadcast to all types) or a dict mapping type names to values.

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
nequip-compile model.nequip.zip dft.nequip.pth \
  --mode torchscript --device cuda --target ase --head dft

nequip-compile model.nequip.zip rpa.nequip.pth \
  --mode torchscript --device cuda --target ase --head rpa
```

The compiled model is a standard single-head model and can be used with any [integration](../../integrations/all.rst) (ASE, LAMMPS, etc.) without any multi-head awareness downstream.

```{important}
Compiling a multi-head model without `--head` will raise an error listing the available heads. Conversely, using `--head` on a single-head model will also raise an error.
```
