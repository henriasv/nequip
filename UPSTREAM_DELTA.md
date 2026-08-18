# Fork delta vs `mir-group/nequip`

**Fork:** `henriasv/nequip`, branch `pr/multigpu`
**Upstream base:** `mir-group/nequip` `main` @ `e07489d4` (current at the time of writing;
the branch is rebased onto it)
**Delta:** ~920 insertions / ~16 deletions across 12 files (2 new files), all under
`nequip/{data,nn,scripts,utils}`.

This document is the complete, annotated description of what this fork changes, written
for upstream review. Companion: the `pair_nequip_allegro` fork's `UPSTREAM_DELTA.md`
(`henriasv/pair_nequip_allegro`, branch `pr/multigpu`) — the two deltas form one feature
(native multi-GPU `pair_nequip`) and are intended to be reviewed together.

---

## What this fork adds

### A. `pair_nequip_multirank` compile target — multi-GPU-capable AOT models

A new `nequip-compile` target (`--target pair_nequip_multirank`, AOTInductor mode) that
produces a `.nequip.pt2` able to run LAMMPS-domain-decomposed across MPI ranks/GPUs. Two
ideas make this exportable:

**1. Per-layer ghost exchange as a registered custom op** (`nequip/nn/_ghost_exchange_pair.py`,
new). A message-passing model needs neighbor-rank atom features at every layer. The
existing ML-IAP ghost exchange is a Python `autograd.Function` holding a `lmp_data`
object, which cannot survive AOTInductor (the artifact runs without a Python
interpreter). Here the exchange is a Torch custom operator,
`nequip_lammps::ghost_exchange` (with `ghost_exchange_reverse` registered as its
autograd transpose), carrying **no LAMMPS handle in its signature** — the C++ pair style
registers the real implementation in the LAMMPS process and reaches the live pair via a
thread-local. In any other context (ASE, export tracing, single rank) the op is the
identity. The op pattern is the same one OpenEquivariance uses, so it traces into the
`.pt2` the same way.

**2. Truncate-to-nlocal via a backed dynamic dimension.** Computing all per-node ops on
`ntotal` (owned + ghost) rows wastes work; but slicing to the owned count `nlocal` read
from a tensor *value* (`.item()`) creates an unbacked symint that `torch.export` cannot
guard. Solution: a new graph-level input `num_local_nodes_marker`, a `(nlocal,)` tensor
whose **shape** carries the owned count. `InteractionBlock` reads
`marker.shape[0]` — a backed dim — and every per-node op is sliced to owned atoms; the
ghost-exchange module re-expands `[nlocal, F] → [ntotal, F]` with `slice_scatter`
(deliberately not `cat`, which would bake a `nghost >= 1` or `<= 0` specialization)
before each exchange. The export example data uses `nghost > 0` so the `nlocal` and
`node` dims stay independent.

Supporting plumbing: new key `NUM_LOCAL_NODES_MARKER_KEY` registered as a *graph* field
(it has `nlocal` rows, not `num_nodes` rows); a dedicated `nlocal` dynamic dim in
`get_dynamic_shapes`; `GraphModel` whitelists and passes the marker through (by literal
string — see feature C for why); `AtomwiseReduce` masks ghost contributions with a
guard-free elementwise comparison so energies (and hence forces) are owned-only; the
`enable_PairNequIPGhostExchange` model modifier swaps the no-op exchange module for the
real one and is **auto-applied by the target** (it is not a user-facing choice; a
self-correcting error rejects it with any other target). `nequip-compile` additionally
stamps the widest per-node feature width crossing the exchange
(`pair_nequip_feature_width` metadata) so the pair style sizes its LAMMPS comm buffers
exactly.

### B. Custom-op `.so` embedding for C++ consumers

For `pair_*` targets, `nequip-compile` now embeds the actual compiled custom-op shared
libraries (e.g. OpenEquivariance's `libtorch_tp_jit.so`) into the `.pt2` zip, stored
uncompressed under `nequip_custom_op_libs/`, so the artifact is **self-contained** for a
pure-C++ consumer that cannot `import openequivariance`
(`nequip/utils/aoti_metadata.py`: `resolve_custom_op_so_paths` — via the library's
declared accessor with a `/proc/self/maps` fallback — and `embed_custom_op_so_libs`).
The required libs are derived from model metadata **unioned with the requested
modifiers** (previously metadata-only; a missing entry would silently produce a
non-self-contained `.pt2`). The Python/ASE path keeps the existing lighter name-based
mechanism unchanged.

### C. Version-robust auto-rebundle for packaged models

A `.nequip.zip` (torch.package) model carries its own frozen copy of `nequip.nn` source
*and* pickled module instances whose `__init__` never re-runs. A model packaged before
this fork therefore has no truncate-to-nlocal / ghost-exchange code in its bundle, and —
this is the dangerous part — would **silently** run the non-truncating path (energy sums
include ghosts ⇒ wrong physics on >1 rank; observed in production as a +0.774 eV/atom
offset on a fine-tuned model before this machinery existed).

When compiling a stale package for the `pair_nequip_multirank` target, `nequip-compile`
now refreshes the bundle first (`_maybe_rebundle_multirank` in `scripts/compile.py`):
it **derives an era-matched overlay from the package's own bundled `nn` source** and
applies only the bounded, additive multirank grafts
(`nequip/scripts/_multirank_graft.py`, new): the marker branch in `InteractionBlock`,
the marker passthrough in `GraphModel.forward`, the modifier + real exchange module,
custom-ops metadata collection, and the owned-only `AtomwiseReduce`/ZBL fixes. Deriving
from the package's own source (rather than injecting the installed or a fixed-era `nn`)
keeps the model's frozen instances attribute- and import-compatible **by construction**,
regardless of the package's nequip era. Each graft is idempotent and anchored on stable
code patterns; a missed anchor logs and degrades gracefully. The rebundle runs through
`nequip-package update`, which verifies predictions are unchanged before writing. Models
whose bundles already carry the support are detected (sentinel scan) and left alone.
A `pair_nequip_multirank` metadata stamp (gated on the target, so it always agrees with
the declared inputs) lets the C++ side guard single-rank models on N ranks.

Because the marker key must exist in *old* bundles after grafting, the shipped sources
reference `"num_local_nodes_marker"` by **literal string** rather than the
`AtomicDataDict.NUM_LOCAL_NODES_MARKER_KEY` attribute in the few places that get
repacked into older models (each such site carries a comment saying so).

An env-gated escape hatch (`$NEQUIP_MULTIRANK_PKGSRC`) can pin an explicit overlay
directory instead of deriving one.

---

## Backward-compatibility audit

Claim: **all existing entry points, targets, and integrations behave exactly as before;
the new machinery activates only through the new target/modifier/input.** The new model
input is absent from every existing dataset and integration, the new target is opt-in,
and the modifier cannot be applied outside it.

All touches of *shared* (pre-existing) code paths, exhaustively:

| # | File / change | Effect on existing behavior |
|---|---------------|------------------------------|
| 1 | `_key_registry.py`: `NUM_LOCAL_GHOST_NODES_KEY` added to `_DEFAULT_GRAPH_FIELDS`; listed in the rank-1 special case of `get_dynamic_shapes` | Registers the (pre-existing, ML-IAP) key as graph-typed for export machinery. The ML-IAP integration supplies this key outside `torch.export` (eager), so its behavior is unchanged; the registration is exercised only by the new target. |
| 2 | `atomwise.py` `AtomwiseReduce`: owned-mask when `num_local_ghost_atoms` present; `BATCH_KEY` sliced to the field length | For every existing path this is a no-op, provably: without the key the branch is skipped; in the ML-IAP path the field is already truncated to `nlocal`, so the mask is all-True and the batch slice changes nothing. |
| 3 | `interaction_block.py`: `x = tp_scatter(...)[:num_local_nodes]` split into two statements; new `elif` for the marker | Identical semantics for existing paths (`num_local_nodes` falls through to `num_nodes` exactly as upstream). |
| 4 | `pair_potential.py` ZBL: scatter over `ntotal` then truncate, instead of scattering with `dim_size = per-atom-energy length` | **This is an upstream bug fix.** In a local-ghost context (ML-IAP today), the per-atom energy field spans `nlocal` while `edge_center` indexes up to `ntotal`; upstream's `dim_size=nlocal` scatter is an out-of-bounds write whenever an edge is centered on a ghost atom. In the no-ghost case the two formulations are identical. Could be split out as its own upstream PR. |
| 5 | `graph_model.py`: marker whitelisted in `__init__` irreps and passed through in `forward` | Pure addition; keys absent ⇒ both are no-ops. |
| 6 | `compile.py`: custom-ops embedding block restructured (metadata ∪ modifier-derived libs) | Name-based embedding behaves as before when metadata is present; the union only *adds* libs that the requested modifiers demonstrably require (with a warning when metadata was missing them). `.so` embedding is gated on `pair_*` targets. |

Everything else is new files (`_ghost_exchange_pair.py`, `_multirank_graft.py`) or new,
gated blocks (new target entry, rebundle helpers, marker key definition, modifier
classmethod).

Removed relative to the deployed LUMI line (not upstream): one env-gated debug dump in
`utils/aot.py` that was development scaffolding.

---

## Verification

- **Upstream unit tests** run against this branch (CPU) — see the PR conversation for
  the current pass state.
- **On LUMI-G (MI250X, 1–32 GCDs), production models:** single-rank vs multi-rank
  PE agreement ≈ exact at 1 rank (gate threshold 1e-4 eV/atom); rank-invariance 1/8/32
  GCDs ≈ 1e-6 eV/atom; max |ΔF| vs single-rank `pair_nequip` ≈ 1e-5 eV/Å; agreement
  with the independent `mliap/kk` engine on the same cell. The **era-robustness** of the
  rebundle (feature C) is itself gated: a nequip-0.17-packaged fine-tuned model compiled
  on a nequip-0.15-era stack reproduces the single-rank ground truth exactly (the
  original silent-offset scenario), and a 0.15-era base model still passes regression.
- **Benchmarks** (bound-corrected, OEQ on both engines): parity with `mliap/kk` when
  compute-bound; ~2.2–2.6× higher strong-scaling ceiling at small per-GCD load; details
  in the pair_nequip_allegro delta doc.

Since resolved on this branch: a user-facing docs page
(`docs/integrations/lammps/multirank.md`, linked from the LAMMPS docs index) and a
CPU-runnable regression test (`PairNequIPMultirankMixin`,
`test_pair_nequip_multirank_matches_single_rank`: single-rank vs multirank artifacts
from the same model must agree at `nghost = 0`, plus the metadata-stamp asserts). The
ZBL fix (audit row 4) is also available as the standalone branch
`fix/zbl-local-ghost-scatter` for a separate PR.

Still open for maintainer review:
- the `slice_scatter` export reasoning and marker-key design vs upstream's export-guard
  conventions;
- whether the auto-rebundle machinery (feature C) belongs upstream or should remain a
  deployment-side tool.
