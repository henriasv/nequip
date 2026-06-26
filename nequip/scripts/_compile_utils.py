# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
from nequip.data import AtomicDataDict
from typing import Dict, List, Callable, Union

# === Inputs and Outputs for AOT Compile ===
# standard sets of input and output fields for specific integrations

AOTI_PAIR_NEQUIP_TARGET = "pair_nequip"
AOTI_PAIR_NEQUIP_MULTIRANK_TARGET = "pair_nequip_multirank"
AOTI_ASE_TARGET = "ase"
AOTI_BATCH_TARGET = "batch"

PAIR_NEQUIP_INPUTS = [
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
    AtomicDataDict.CELL_KEY,
    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
]

# multi-rank native `pair_nequip` additionally takes the owned/ghost atom counts
# (`[nlocal, nghost]`) so the per-layer ghost exchange can route the feature halo, plus a
# `(nlocal,)` marker tensor that carries the owned count as a *backed* dynamic dimension. The
# truncate-to-nlocal reformulation reads `nlocal` from `marker.shape[0]` (a tensor dimension,
# never `.item()`), so per-node ops are computed only on owned atoms while staying AOT-exportable.
PAIR_NEQUIP_MULTIRANK_INPUTS = PAIR_NEQUIP_INPUTS + [
    AtomicDataDict.NUM_LOCAL_GHOST_NODES_KEY,
    AtomicDataDict.NUM_LOCAL_NODES_MARKER_KEY,
]

BATCH_INPUTS = PAIR_NEQUIP_INPUTS + [
    AtomicDataDict.BATCH_KEY,
    AtomicDataDict.NUM_NODES_KEY,
]

LMP_OUTPUTS = [
    AtomicDataDict.PER_ATOM_ENERGY_KEY,
    AtomicDataDict.FORCE_KEY,
    AtomicDataDict.VIRIAL_KEY,
]

ASE_OUTPUTS = [
    AtomicDataDict.PER_ATOM_ENERGY_KEY,
    AtomicDataDict.TOTAL_ENERGY_KEY,
    AtomicDataDict.FORCE_KEY,
    AtomicDataDict.STRESS_KEY,
]


# === batch map rules ===
def single_frame_batch_map_settings(batch_map):
    # make num_frames batch dims static, for single frame case
    # relevant for single-frame use cases, e.g. pair_nequip and ase
    batch_map["graph"] = torch.export.Dim.STATIC
    return batch_map


# === data rules ===
def single_frame_data_settings(data):
    # because of the 0/1 specialization problem,
    # and the fact that the LAMMPS pair style (and ASE) requires `num_frames=1`
    # we need to augment to data to remove the `BATCH_KEY` and `NUM_NODES_KEY`
    # to take more optimized code paths
    if AtomicDataDict.BATCH_KEY in data:
        data.pop(AtomicDataDict.BATCH_KEY)
        data.pop(AtomicDataDict.NUM_NODES_KEY)
    return data


def single_frame_pair_nequip_multirank_batch_map_settings(batch_map):
    # single-frame settings (graph dim static), plus a dedicated `nlocal` dynamic dimension for
    # the owned-atom marker, independent of `node` (== ntotal). The relation `nlocal <= ntotal`
    # is not declared here; the truncate-to-nlocal slice/`slice_scatter` ops impose it as a
    # deferred runtime assert during export, which is satisfiable at `nghost == 0` (the
    # single-rank correctness gate) and `nghost > 0` (multi-rank) alike.
    batch_map = single_frame_batch_map_settings(batch_map)
    batch_map["nlocal"] = torch.export.dynamic_shapes.Dim(
        "nlocal", min=1, max=torch.inf
    )
    return batch_map


def single_frame_pair_nequip_multirank_data_settings(data):
    # single-frame settings, plus the two multi-rank runtime inputs (this only sets the tracing
    # example; the pair style overrides the values at runtime):
    #   * `num_local_ghost_atoms` = `[nlocal, nghost]` (drives the guard-free owned mask), and
    #   * `num_local_nodes_marker` = a `(nlocal,)` tensor whose dim-0 carries the *backed* owned
    #     count that the truncate-to-nlocal reformulation slices on.
    # CRUCIAL: the example MUST have `nghost > 0` (here `nlocal = n_nodes // 2`). With
    # `nghost == 0` the `node` and `nlocal` dims would coincide in the example and export
    # specializes the ghost block away (baking `ntotal <= nlocal`), which then fails for real
    # multi-rank runs. With genuine ghosts the two dims stay independent and the `slice_scatter`
    # expansion in `PairNequIPGhostExchangeModule` exports guard-cleanly for any `nlocal <=
    # ntotal`. Example values are otherwise physically irrelevant (the AOT-vs-eager sanity check
    # only needs self-consistency, and the single-rank gate runs `nghost == 0`).
    data = single_frame_data_settings(data)
    device = data[AtomicDataDict.POSITIONS_KEY].device
    n_nodes = data[AtomicDataDict.POSITIONS_KEY].shape[0]
    n_local = max(1, n_nodes // 2)
    n_ghost = n_nodes - n_local
    data[AtomicDataDict.NUM_LOCAL_GHOST_NODES_KEY] = torch.tensor(
        [n_local, n_ghost], dtype=torch.int64, device=device
    )
    data[AtomicDataDict.NUM_LOCAL_NODES_MARKER_KEY] = torch.zeros(
        n_local, dtype=torch.int64, device=device
    )
    return data


def batched_data_settings(data):
    assert AtomicDataDict.BATCH_KEY in data
    assert AtomicDataDict.NUM_NODES_KEY in data
    # just make a batch of 2 frames to avoid 0/1 specialization problem later on
    data = AtomicDataDict.batched_from_list([data, data])
    return data


PAIR_NEQUIP_TARGET = {
    "input": PAIR_NEQUIP_INPUTS,
    "output": LMP_OUTPUTS,
    "batch_map_settings": single_frame_batch_map_settings,
    "data_settings": single_frame_data_settings,
}
PAIR_NEQUIP_MULTIRANK_TARGET = {
    "input": PAIR_NEQUIP_MULTIRANK_INPUTS,
    "output": LMP_OUTPUTS,
    "batch_map_settings": single_frame_pair_nequip_multirank_batch_map_settings,
    "data_settings": single_frame_pair_nequip_multirank_data_settings,
}
ASE_TARGET = {
    "input": PAIR_NEQUIP_INPUTS,
    "output": ASE_OUTPUTS,
    "batch_map_settings": single_frame_batch_map_settings,
    "data_settings": single_frame_data_settings,
}
BATCH_TARGET = {
    "input": BATCH_INPUTS,
    "output": ASE_OUTPUTS,
    "batch_map_settings": lambda batch_map: batch_map,  # no static shapes
    "data_settings": batched_data_settings,
}

COMPILE_TARGET_DICT = {
    AOTI_PAIR_NEQUIP_TARGET: PAIR_NEQUIP_TARGET,
    AOTI_PAIR_NEQUIP_MULTIRANK_TARGET: PAIR_NEQUIP_MULTIRANK_TARGET,
    AOTI_ASE_TARGET: ASE_TARGET,
    AOTI_BATCH_TARGET: BATCH_TARGET,
}


def register_compile_targets(
    target_dict: Dict[str, Union[List[str], Callable]],
) -> None:
    """Register compile targets for AOT compilation.

    The intended clients of this function are NequIP extension packages to register their custom compilation targets.

    Args:
        target_dict: dict containing keys ``input``, ``output``, ``batch_map_settings``, ``data_settings``
    """
    # update target dict
    global COMPILE_TARGET_DICT
    COMPILE_TARGET_DICT.update(target_dict)
