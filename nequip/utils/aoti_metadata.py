# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import importlib
import os
import zipfile
from typing import List, Final, Optional, Set

NEQUIP_AOTI_INPUTS_KEY: Final[str] = "nequip_aoti_inputs"
NEQUIP_AOTI_OUTPUTS_KEY: Final[str] = "nequip_aoti_outputs"
NEQUIP_CUSTOM_OPS_LIBS_KEY: Final[str] = "nequip_custom_ops_libs"

_CUSTOM_OPS_LIBS_ENTRY = "nequip_custom_ops_libs.txt"

# In-zip directory (uncompressed) holding the actual custom-op `.so` binaries for
# pure-C++ consumers (LAMMPS `pair_nequip`/`pair_allegro`), which have no interpreter
# to import the Python libraries named in `_CUSTOM_OPS_LIBS_ENTRY`.
_CUSTOM_OP_SO_DIR: Final[str] = "nequip_custom_op_libs/"

# Per custom-ops library: how to find the compiled `.so` that registers its Torch ops.
#   `accessors`: (module, attribute) pairs yielding the `.so` path (a value or callable).
#   `so_basename_hint`: substring used to recognise the library's already-mapped `.so`
#                       in `/proc/self/maps` as a fallback.
_KNOWN_CUSTOM_OP_LIBS: Final[dict] = {
    "openequivariance": {
        "accessors": [("openequivariance.extlib", "torch_ext_so_path")],
        "so_basename_hint": "libtorch_tp_jit",
    },
}


def serialize_aoti_keys(keys: List[str]) -> str:
    assert all(" " not in key for key in keys), (
        f"AOTI field names cannot contain spaces: {keys}"
    )
    return " ".join(keys)


def parse_aoti_keys(serialized_keys: str) -> List[str]:
    return serialized_keys.split()


def embed_custom_ops_libs(pt2_path: str, custom_ops_libs: Set[str]) -> None:
    """Append a custom ops libs entry to an existing AOTI .pt2 zip archive.

    Called after ``aoti_compile_and_package`` to record which Python libraries must be imported before the package can be loaded.
    The entry is written as a plain space-separated text file so that ``import_custom_ops_libs`` can read it with a bare ``zipfile`` open before PyTorch's C++ loader runs.

    Args:
        pt2_path: path to the .pt2 file to modify in-place.
        custom_ops_libs: set of importable library names (e.g. ``{"openequivariance"}``).
    """
    if not custom_ops_libs:
        return
    with zipfile.ZipFile(pt2_path, "a") as zf:
        zf.writestr(_CUSTOM_OPS_LIBS_ENTRY, " ".join(sorted(custom_ops_libs)))


def import_custom_ops_libs(pt2_path: str) -> None:
    """Read the custom ops libs entry from a .pt2 archive and import each library.

    Must be called *before* ``torch._inductor.aoti_load_package`` so that custom op schemas are registered before the C++ ``AOTIModelPackageLoader`` runs.

    No-op if the entry is absent (e.g. models compiled without custom ops, or models compiled before this feature was added).

    Args:
        pt2_path: path to the .pt2 file to inspect.
    """
    with zipfile.ZipFile(pt2_path, "r") as zf:
        if _CUSTOM_OPS_LIBS_ENTRY not in zf.namelist():
            return
        for lib in zf.read(_CUSTOM_OPS_LIBS_ENTRY).decode().split():
            importlib.import_module(lib)


def _find_mapped_extension_so(basename_hint: str) -> Optional[str]:
    """Best-effort: find an already-mapped ``.so`` whose basename matches ``basename_hint``.

    Custom-op libraries are imported (and their JIT extension ``.so`` mapped into this
    process) by the time we embed, so ``/proc/self/maps`` lets us recover the binary's
    path even when the build cache dir is non-standard (e.g. a node-local
    ``TORCH_EXTENSIONS_DIR``). Linux-only; returns ``None`` if unavailable.
    """
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                path = line.rstrip("\n").rsplit(" ", 1)[-1].strip()
                if (
                    path.endswith(".so")
                    and basename_hint in os.path.basename(path)
                    and os.path.exists(path)
                ):
                    return os.path.realpath(path)
    except OSError:
        return None
    return None


def resolve_custom_op_so_paths(custom_ops_libs: Set[str]) -> List[str]:
    """Resolve importable custom-op library names to their compiled ``.so`` path(s).

    For Python consumers, registering a custom op is a side effect of importing the
    library (see :func:`import_custom_ops_libs`). Pure-C++ consumers (the LAMMPS
    ``pair_nequip``/``pair_allegro`` styles) have no interpreter, so the actual ``.so``
    must be embedded in the ``.pt2`` and ``dlopen``-ed at model load. This resolves each
    library name to the absolute path of the shared object that registers its Torch
    operators, ready for :func:`embed_custom_op_so_libs`.

    The libraries have already been imported (the model modifier loaded them) by the time
    this runs, so each compiled ``.so`` is mapped into the current process.

    Args:
        custom_ops_libs: importable library names (e.g. ``{"openequivariance"}``).

    Returns:
        De-duplicated absolute ``.so`` paths. Libraries that cannot be resolved are
        silently skipped here; the caller is responsible for warning if the resulting
        ``.pt2`` would not be self-contained.
    """
    resolved: List[str] = []
    for name in sorted(custom_ops_libs):
        try:
            importlib.import_module(name)
        except ImportError:
            continue
        spec = _KNOWN_CUSTOM_OP_LIBS.get(name, {})
        so: Optional[str] = None
        for modpath, attr in spec.get("accessors", []):
            try:
                obj = getattr(importlib.import_module(modpath), attr)
                cand = obj() if callable(obj) else obj
                if cand and os.path.exists(cand):
                    so = os.path.realpath(cand)
                    break
            except Exception:
                pass
        if so is None and spec.get("so_basename_hint"):
            so = _find_mapped_extension_so(spec["so_basename_hint"])
        if so is not None:
            resolved.append(so)
    # de-duplicate, preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for p in resolved:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def embed_custom_op_so_libs(pt2_path: str, so_paths: List[str]) -> None:
    """Embed compiled custom-op shared libraries into an AOTI ``.pt2`` zip archive.

    Unlike :func:`embed_custom_ops_libs` (which records importable *library names* for
    Python consumers), this stores the actual ``.so`` *binaries* under the
    ``nequip_custom_op_libs/`` prefix, **uncompressed (STORED)**, so a pure-C++ consumer
    can locate them with a minimal zip read and ``dlopen`` them without a Python
    interpreter. Intended only for the ``pair_nequip``/``pair_allegro`` C++ targets; the
    Python/ASE path keeps using the lighter name-based mechanism.

    Args:
        pt2_path: path to the .pt2 file to modify in-place.
        so_paths: absolute paths of the ``.so`` files to embed.
    """
    if not so_paths:
        return
    with zipfile.ZipFile(pt2_path, "a") as zf:
        existing = set(zf.namelist())
        for so in so_paths:
            arcname = _CUSTOM_OP_SO_DIR + os.path.basename(so)
            if arcname in existing:
                continue
            # STORED (no compression) so the C++ loader can copy the bytes directly.
            zf.write(so, arcname, compress_type=zipfile.ZIP_STORED)
