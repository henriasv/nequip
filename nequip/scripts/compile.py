# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch

from ._workflow_utils import set_workflow_state
from ._compile_utils import COMPILE_TARGET_DICT
from nequip.model.utils import _EAGER_MODEL_KEY
from nequip.model.saved_models.load_utils import load_saved_model
from nequip.model.modify_utils import modify
from nequip.train.lightning import _SOLE_MODEL_KEY
from nequip.data import AtomicDataDict
from nequip.utils.logger import RankedLogger
from nequip.utils.global_state import set_global_state, get_latest_global_state
from nequip.utils.versions import _TORCH_GE_2_10
from nequip.utils.aoti_metadata import (
    NEQUIP_AOTI_INPUTS_KEY,
    NEQUIP_AOTI_OUTPUTS_KEY,
    NEQUIP_CUSTOM_OPS_LIBS_KEY,
    serialize_aoti_keys,
    embed_custom_ops_libs,
    embed_custom_op_so_libs,
    resolve_custom_op_so_paths,
)
from omegaconf import OmegaConf
import hydra

import yaml
import argparse
import pathlib
from typing import Final


# === setup logging ===
hydra.core.utils.configure_log(None)
logger = RankedLogger(__name__, rank_zero_only=True)

# hardcode a global seed for `nequip-compile`
_COMPILE_SEED: Final[int] = 1

# === AOT keys ===
_AOT_METADATA_KEY = "aot_inductor.metadata"
_AOT_OUTPUT_PATH_KEY = "aot_inductor.output_path"


def _parse_bounds_to_Dim(name: str, bounds_str: str):
    if bounds_str == "static":
        return torch.export.Dim.STATIC
    else:
        min_val, max_val = bounds_str.split(",")
        return torch.export.dynamic_shapes.Dim(
            name,
            min=int(min_val),
            max=torch.inf if max_val == "inf" else int(max_val),
        )


# === pair_nequip multi-GPU (multirank) re-bundle support ===
# The native multi-GPU pair_nequip path needs the truncate-to-nlocal model code in the
# packaged model's *bundled* `nn` source. A `.nequip.zip` carries its own frozen copy of
# that source (`torch.package` isolates it), so a model packaged before this support — or
# with stock upstream nequip — loads stale `nn` and would silently run the all-`ntotal`
# (non-truncating) path or fault under the multi-rank ghost exchange. When compiling for the
# pair_nequip multirank target we detect a stale bundle and refresh it in place from the
# *installed* (container) nequip before loading. See `nequip/nn/_ghost_exchange_pair.py`.
_PAIR_NEQUIP_MULTIRANK_MODIFIER: Final[str] = "enable_PairNequIPGhostExchange"
_PAIR_NEQUIP_MULTIRANK_META_KEY: Final[str] = "pair_nequip_multirank"
# nn source files that carry the truncate-to-nlocal / marker plumbing (validated repack set)
_MULTIRANK_NN_FILES: Final[tuple] = (
    "_ghost_exchange_base.py",
    "_ghost_exchange_pair.py",
    "interaction_block.py",
    "atomwise.py",
    "graph_model.py",
    "pair_potential.py",
)
# literal marker key the patched nn references (the bundled AtomicDataDict predates the
# `NUM_LOCAL_NODES_MARKER_KEY` attribute, so the patched source keys it by this string)
_MULTIRANK_SENTINEL: Final[str] = "num_local_nodes_marker"


def _bundle_is_multirank_capable(zip_path) -> bool:
    """Whether a packaged model's bundled ``nn`` already carries truncate-to-nlocal support.

    Detected by the ``num_local_nodes_marker`` sentinel in the bundled
    ``interaction_block.py``. Returns ``True`` (i.e. skip the re-bundle) if the package
    cannot be inspected in the expected layout — refreshing a non-standard package would be
    unsafe, and the pair style's runtime multi-rank guard is the backstop.
    """
    import zipfile

    try:
        with zipfile.ZipFile(zip_path) as zf:
            hits = [
                n
                for n in zf.namelist()
                if n.endswith("/nequip/nn/interaction_block.py")
            ]
            if not hits:
                return True
            return _MULTIRANK_SENTINEL in zf.read(hits[0]).decode("utf-8", "replace")
    except (zipfile.BadZipFile, OSError):
        return True


def _multirank_zip_prefix(zip_path):
    """The ``torch.package`` ``<pkg_dir>`` prefix inside a packaged model, or ``None``."""
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        hits = [
            n for n in zf.namelist() if n.endswith("/nequip/nn/interaction_block.py")
        ]
    return hits[0].rsplit("/nequip/nn/", 1)[0] if hits else None


def _maybe_rebundle_multirank(input_path, mode, target, modifiers):
    """Refresh a stale packaged model for the pair_nequip multirank target.

    Returns the path to load from: the original ``input_path`` when no refresh is
    needed/possible, otherwise a fresh ``.nequip.zip`` whose bundled ``nn`` source has been
    replaced from the installed nequip. The refresh runs through ``nequip-package update``,
    which verifies the model's predictions are unchanged before writing.
    """
    if mode != "aotinductor" or target != "pair_nequip":
        return input_path
    if _PAIR_NEQUIP_MULTIRANK_MODIFIER not in (modifiers or []):
        return input_path

    p = pathlib.Path(input_path)
    if not (str(input_path).endswith(".nequip.zip") and p.is_file()):
        # checkpoint / nequip.net ref / missing file: nothing to refresh here. If the
        # bundled code is stale the pair style's runtime multi-rank guard reports it.
        return input_path

    if _bundle_is_multirank_capable(p):
        logger.info(
            "pair_nequip multirank: bundled nequip.nn already supports truncate-to-nlocal; "
            "no re-bundle needed."
        )
        return input_path

    prefix = _multirank_zip_prefix(p)
    if prefix is None:
        logger.warning(
            "pair_nequip multirank: could not find `nequip/nn/` inside %s to re-bundle; "
            "proceeding as-is. If this is a packaged NequIP model, re-bundle it manually "
            "with `nequip-package update` (see the multi-GPU pair_nequip docs).",
            input_path,
        )
        return input_path

    from nequip.scripts.package import main as _package_main

    tmp_out = p.with_name(p.name[: -len(".nequip.zip")] + ".mrt-auto.nequip.zip")
    if tmp_out.exists():
        tmp_out.unlink()
    replace_args = []
    for f in _MULTIRANK_NN_FILES:
        # 1-arg `--replace`: nequip-package auto-resolves the local file from the *installed*
        # nequip package — i.e. the container's patched source, exactly what we want.
        replace_args += ["--replace", f"{prefix}/nequip/nn/{f}"]
    logger.warning(
        "pair_nequip multirank: bundled nequip.nn predates truncate-to-nlocal; re-bundling "
        "%s from the installed nequip (%d nn files) -> %s (predictions verified unchanged).",
        p.name,
        len(_MULTIRANK_NN_FILES),
        tmp_out.name,
    )
    try:
        _package_main(["update", str(p), str(tmp_out), *replace_args])
    except Exception as e:
        raise RuntimeError(
            f"pair_nequip multirank auto re-bundle of '{input_path}' failed: {e}\n"
            "The model could not be refreshed from the installed nequip. Fix options:\n"
            "  (1) ensure the container's nequip carries the multi-GPU pair_nequip patches "
            "(the pair-nequip-multigpu build);\n"
            "  (2) re-bundle manually: `nequip-package update <src> <out> --replace "
            "<pkg>/nequip/nn/interaction_block.py ...` then pass <out> to nequip-compile;\n"
            "  (3) re-export the model from its checkpoint with a current nequip."
        ) from e
    return str(tmp_out)


def main(args=None):
    # === parse inputs ===
    parser = argparse.ArgumentParser(
        description="Compiles NequIP framework models from checkpoint or package files."
    )

    # positional arguments:
    parser.add_argument(
        "input_path",
        help="path to a packaged model file (local `.nequip.zip` file), a nequip.net model (`nequip.net:group-name/model-name:version`), or a checkpoint file (any other local path)",
        type=str,
    )

    parser.add_argument(
        "output_path",
        help="path to write compiled model file. NOTE: a `.nequip.pth` extension is required if `--mode torchscript` is used and a `.nequip.pt2` extension is required if `--mode aotinductor` is used",
        type=pathlib.Path,
    )

    # required named arguments:
    required_named = parser.add_argument_group("required arguments")
    required_named.add_argument(
        "--mode",
        help="whether to use `torchscript` or `aotinductor` to compile the model",
        choices=["torchscript", "aotinductor"],
        type=str,
        required=True,
    )

    required_named.add_argument(
        "--device",
        help="device to run the model on",
        type=str,
        required=True,
    )

    # optional named arguments:
    parser.add_argument(
        "--model",
        help=f"name of model to compile -- this option is only relevant when using multiple models (default: {_SOLE_MODEL_KEY}, meant to work for the conventional single model case)",
        type=str,
        default=_SOLE_MODEL_KEY,
    )

    parser.add_argument(
        "--tf32",
        help="whether to use TF32 or not (default: False)",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--modifiers",
        help="modifiers to apply to the model before compiling",
        nargs="+",
        type=str,
        default=[],
    )

    # args specific to export
    parser.add_argument(
        "--target",
        help="target application for AOT export (`input-fields` and `output-fields` need not be specified if `target` is specified)",
        choices=COMPILE_TARGET_DICT.keys(),
        type=str,
        default=None,
    )

    parser.add_argument(
        "--input-fields",
        help="input fields to the model for export",
        nargs="+",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-fields",
        help="output fields of the model for export",
        nargs="+",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data-path",
        help="path to data (with realistic shapes) for compilation (if unspecified, training data will be used for compilation)",
        type=pathlib.Path,
        default=None,
    )
    # for configuring dynamic shape bounds
    parser.add_argument(
        "--num-frames",
        type=str,
        default="2,inf",
        help="bounds for num-frames in format `min,max` or `static` (default: 2,inf)",
    )
    parser.add_argument(
        "--num-edges",
        type=str,
        default="2,inf",
        help="bounds for num-edges in format `min,max` or `static` (default: 2,inf)",
    )
    parser.add_argument(
        "--num-nodes",
        type=str,
        default="2,inf",
        help="bounds for num-nodes in format `min,max` or `static` (default: 2,inf)",
    )
    parser.add_argument(
        "--inductor-configs",
        help="options for AOTInductor (default: {})",
        nargs="+",
        type=str,
        default=[],
    )
    parser.add_argument(
        "--constant-fold",
        help="enable constant folding optimization for AOTInductor (requires PyTorch 2.8+). May fail on some models. Please report any errors at https://github.com/mir-group/nequip (default: False)",
        action="store_true",
        default=False,
    )
    args = parser.parse_args(args=args)

    set_workflow_state("compile")

    # === check for deprecated torchscript mode ===
    if args.mode == "torchscript" and _TORCH_GE_2_10:
        raise ValueError(
            "TorchScript compilation is deprecated and not supported in PyTorch >= 2.10. "
            "Please use `--mode aotinductor` instead. "
            "See https://pytorch.org/blog/pytorch-2-10-release-blog/ for more information."
        )

    # === initialize global state ===
    set_global_state(allow_tf32=args.tf32)

    # == device ==
    device = args.device
    device = torch.device(device)
    logger.info(f"Compiling for device: {device}")

    # == output path extension ==
    if args.mode == "torchscript":
        assert str(args.output_path).endswith(".nequip.pth"), (
            "`output-path` must end with the `.nequip.pth` extension for `torchscript` compile mode"
        )
    elif args.mode == "aotinductor":
        assert str(args.output_path).endswith(".nequip.pt2"), (
            "`output-path` must end with the `.nequip.pt2` extension for `aotinductor` compile mode"
        )

    # === pair_nequip multirank: auto-refresh stale bundled nn (see helper above) ===
    args.input_path = _maybe_rebundle_multirank(
        args.input_path, args.mode, args.target, args.modifiers
    )

    # === load model ===
    # For aotinductor mode, we also need the data dict (unless data_path is provided)
    need_data_from_model = args.mode == "aotinductor" and args.data_path is None

    model = load_saved_model(
        args.input_path,
        _EAGER_MODEL_KEY,
        args.model,
        return_data_dict=need_data_from_model,
    )
    if need_data_from_model:
        model, data_from_loaded_model = model

    # === modify model ===
    # for now, we restrict modifiers to those without arguments, i.e. accelerations
    model = modify(model, [{"modifier": modifier} for modifier in args.modifiers])

    # === combine model and global options metadata ===
    # note that model.metadata can be dynamic and so can account for things that change as a result of modifiers
    # reference the implementation of model.metadata to check whether this is true for any particular metadata key
    metadata = model.metadata.copy()
    global_metadata_state = get_latest_global_state(only_metadata_related=True)
    assert set(metadata.keys()).isdisjoint(global_metadata_state.keys())
    metadata.update(global_metadata_state)
    del global_metadata_state
    assert all(isinstance(k, str) for k in metadata.keys())
    assert all(isinstance(v, (str, bool)) for v in metadata.values())
    # ensure bool -> str(int) for metadata
    metadata = {
        k: str(int(v)) if isinstance(v, bool) else v for k, v in metadata.items()
    }

    # stamp multirank capability so the pair style can guard multi-rank runs (the C++ side
    # aborts with a clear message if a single-rank `.pt2` is run on >1 MPI rank).
    if _PAIR_NEQUIP_MULTIRANK_MODIFIER in (args.modifiers or []):
        metadata[_PAIR_NEQUIP_MULTIRANK_META_KEY] = "1"

    logger.debug(model)

    # === TorchScript ===
    if args.mode == "torchscript":
        from nequip.model.inference_models.torchscript import save_torchscript_model

        save_torchscript_model(model, metadata, args.output_path, device)
        logger.info(f"TorchScript model saved to {args.output_path}")
        set_workflow_state(None)
        return

    # === AOTInductor ===
    if args.mode == "aotinductor":
        # === sanity check and guarded imports ===
        from nequip.utils.versions import check_pt2_compile_compatibility

        check_pt2_compile_compatibility()
        from nequip.utils.aot import aot_export_model

        # === get data for compilation ===
        if args.data_path is not None:
            # we use `torch.jit.load` to future proof for the case where C++ clients like LAMMPS would need to provide data
            data = {}
            for k, v in torch.jit.load(args.data_path).state_dict().items():
                data[k] = v
        else:
            data = data_from_loaded_model
        data = AtomicDataDict.to_(data, device)

        # === parse batch dims range ===
        batch_map = {
            "graph": _parse_bounds_to_Dim("num_frames", args.num_frames),
            "node": _parse_bounds_to_Dim("num_nodes", args.num_nodes),
            "edge": _parse_bounds_to_Dim("num_edges", args.num_edges),
        }

        # === get target specific settings ===
        if args.target is None:
            assert args.input_fields is not None and args.output_fields is not None, (
                "Either `target` or `input-fields` and `output-fields` must be provided for `aotinductor` compile mode"
            )
            input_fields = args.input_fields
            output_fields = args.output_fields
        else:
            # no checks necessary here as they would have been caught by argparse earlier
            tdict = COMPILE_TARGET_DICT[args.target]
            input_fields = tdict["input"]
            output_fields = tdict["output"]
            batch_map = tdict["batch_map_settings"](batch_map)
            data = tdict["data_settings"](data)

        logger.debug(
            "Dynamic shapes:\n"
            + "\n".join(
                [
                    f"{dim.__name__:^12} range: [{dim.min}, {dim.max}]"
                    for dim in batch_map.values()
                    if dim != torch.export.Dim.STATIC
                ]
            )
        )

        # === inductor configs ===
        inductor_configs = dict(item.split("=") for item in args.inductor_configs)

        # torch will also error out later on but we can be pre-emptive
        assert _AOT_OUTPUT_PATH_KEY not in inductor_configs

        # we use the metadata key to keep our own metadata
        assert _AOT_METADATA_KEY not in inductor_configs
        metadata[NEQUIP_AOTI_INPUTS_KEY] = serialize_aoti_keys(input_fields)
        metadata[NEQUIP_AOTI_OUTPUTS_KEY] = serialize_aoti_keys(output_fields)
        metadata = {k: str(v) for k, v in metadata.items()}
        inductor_configs[_AOT_METADATA_KEY] = metadata

        logger.debug(
            "Inductor Configs:\n"
            + yaml.dump(
                OmegaConf.to_yaml(inductor_configs),
                default_flow_style=False,
                default_style="|",
            )
        )
        # === export model ===
        _ = aot_export_model(
            model=model,
            device=device,
            input_fields=input_fields,
            output_fields=output_fields,
            data=data,
            batch_map=batch_map,
            output_path=str(args.output_path),
            inductor_configs=inductor_configs,
            constant_fold=args.constant_fold,
            seed=_COMPILE_SEED,
        )
        # Determine which custom-op libraries this model needs. Primary source is the
        # model metadata (populated from each module's `_nequip_custom_ops_libs`). As a
        # robust fallback — especially for C++ (`pair_*`) targets, where a missing entry
        # would silently yield a non-self-contained .pt2 — also derive the libs from the
        # requested acceleration modifiers.
        custom_ops_libs = set()
        if NEQUIP_CUSTOM_OPS_LIBS_KEY in metadata:
            custom_ops_libs |= set(metadata[NEQUIP_CUSTOM_OPS_LIBS_KEY].split())
        # known acceleration modifiers -> the importable op library they require
        _MODIFIER_OP_LIBS = {
            "enable_OpenEquivariance": "openequivariance",
        }
        modifier_libs = {
            _MODIFIER_OP_LIBS[m] for m in args.modifiers if m in _MODIFIER_OP_LIBS
        }
        if modifier_libs - custom_ops_libs:
            logger.warning(
                "Custom-op libraries "
                f"{sorted(modifier_libs - custom_ops_libs)} required by the requested "
                "modifiers were not present in the model metadata "
                f"('{NEQUIP_CUSTOM_OPS_LIBS_KEY}'); deriving them from the modifiers so "
                "the exported model stays self-contained."
            )
        custom_ops_libs |= modifier_libs

        if custom_ops_libs:
            # name-based entry for Python/ASE consumers (import before load)
            embed_custom_ops_libs(str(args.output_path), custom_ops_libs)
            # Pure-C++ targets (LAMMPS pair styles) cannot import a Python library to
            # register its custom ops, so additionally embed the actual op `.so`
            # binaries for the pair style to `dlopen` at model load. Keep the ASE
            # `.pt2` lean by only doing this for the `pair_*` targets.
            if args.target is not None and args.target.startswith("pair_"):
                so_paths = resolve_custom_op_so_paths(custom_ops_libs)
                if so_paths:
                    embed_custom_op_so_libs(str(args.output_path), so_paths)
                    logger.info(
                        f"Embedded {len(so_paths)} custom-op shared "
                        f"librar{'y' if len(so_paths) == 1 else 'ies'} into "
                        f"{args.output_path} for C++ loading: " + ", ".join(so_paths)
                    )
                else:
                    logger.warning(
                        "Could not resolve any custom-op `.so` to embed for C++ "
                        f"target '{args.target}' (libs: {sorted(custom_ops_libs)}). "
                        "The exported .pt2 will not be self-contained; the LAMMPS "
                        "pair style will need NEQUIP_OP_LIBRARIES or a "
                        "<model>.oplibs sidecar to find them."
                    )
        logger.info(f"Exported model saved to {args.output_path}")
        set_workflow_state(None)
        return


if __name__ == "__main__":
    main()
