# Multi-GPU `pair_nequip`

`pair_style nequip` is ordinarily restricted to a single MPI rank: a message-passing model needs the features of atoms owned by neighboring ranks at every interaction layer.
The `pair_nequip_multirank` compile target removes that restriction.
Models compiled with it perform a per-layer ghost-feature exchange through LAMMPS' own communication machinery — the exchange is a Torch custom operator baked into the compiled artifact, and the pair style installs the real implementation at load time — so NequIP models run LAMMPS-domain-decomposed across many ranks and GPUs like any other pair style.

## Compiling a multi-rank model

```bash
nequip-compile \
  path/to/ckpt_file/or/package_file \
  path/to/compiled_model.nequip.pt2 \
  --device cuda \
  --mode aotinductor \
  --target pair_nequip_multirank \
  --modifiers enable_OpenEquivariance   # optional acceleration
```

Notes:

- Only `--mode aotinductor` is supported for this target.
- The target automatically applies the `enable_PairNequIPGhostExchange` modifier — you never pass it yourself (passing it with any *other* target is an error).
- Acceleration modifiers such as [`enable_OpenEquivariance`](../../guide/accelerations/openequivariance.md) compose with this target. For `pair_*` targets the required custom-op shared libraries are embedded into the `.nequip.pt2` so the artifact is self-contained for LAMMPS.
- Compiling an **older packaged model** (a `.nequip.zip` whose bundled code predates multi-rank support) triggers an automatic re-bundle that grafts the multi-rank support onto the package's own bundled source; predictions are verified unchanged in the process. Models re-exported from a checkpoint with a current `nequip` never need this.

## Running in LAMMPS

Use the model exactly like a single-rank `pair_nequip` model, with two requirements:

```
newton          on
pair_style      nequip
pair_coeff      * * compiled_model.nequip.pt2 <type name 1> <type name 2> ...
```

- `newton pair on` is required (ghost-atom forces return to their owners through LAMMPS' standard reverse communication, as for `pair_allegro`).
- Run with one MPI rank per GPU. With the Kokkos package (`pair_style` resolves to `nequip/kk`), the feature exchange stays device-resident and uses GPU-aware MPI where available.

The pair style detects multi-rank capability from the compiled model itself; a single-rank `.nequip.pt2` run on more than one rank aborts with an error that includes the exact recompile command.

## Verifying a compiled model

With no ghost atoms the exchange is the identity, so a **single-rank** run of the multirank artifact must reproduce a single-rank `pair_nequip` artifact compiled from the same model.
This makes a cheap deployment gate: compile both targets from the same package, run both on one rank on a small periodic cell (periodic boundaries ensure `nghost > 0`, so the exchange path actually executes), and compare potential energies.
Multi-rank potential energy must also be invariant to the number of ranks.
The framework's test suite runs the `nghost = 0` equivalence check on CPU; the multi-rank invariance check requires a LAMMPS build.

## Performance notes

- At large per-GPU atom counts the computation is kernel-bound and multi-rank `pair_nequip` performs comparably to the ML-IAP route with the same acceleration; its advantage (2× and more in throughput ceiling) is in the strong-scaling / small-per-GPU-load regime.
- The per-layer exchange moves node features, whose width the compiler stamps into the artifact (`pair_nequip_feature_width`) so the pair style sizes communication buffers exactly.
