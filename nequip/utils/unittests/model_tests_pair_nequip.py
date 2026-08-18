# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""``pair_nequip_multirank`` compile-target test mixin.

CPU-runnable regression test of the multi-rank export path. The multirank artifact is
the single-rank artifact plus (a) the per-layer ghost-exchange custom op and (b) the
truncate-to-nlocal graph transformation. With ``nghost == 0`` (no LAMMPS comm context)
the registered exchange op is the identity and the truncation is a no-op, so the two
artifacts must agree to numerical precision — any deviation means the multirank graph
transformation itself changed the physics. This is the CPU analogue of the LAMMPS-level
multirank correctness gate; only the real cross-rank exchange (which requires LAMMPS and
``nghost > 0``) is out of scope here.
"""

import numpy as np
import pytest
import torch

from nequip.data import AtomicDataDict, from_ase, compute_neighborlist_
from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper
from nequip.model.inference_models.aotinductor import load_aotinductor_model
from nequip.utils.versions import _TORCH_GE_2_6

from .utils import resolve_saved_model_path, run_nequip_compile


class PairNequIPMultirankMixin:
    """Tests for the ``pair_nequip_multirank`` AOTInductor compile target."""

    @pytest.fixture(scope="class")
    @classmethod
    def multirank_tol(cls, model_dtype):
        """May be overridden by subclasses."""
        return {"float32": 5e-5, "float64": 1e-10}[model_dtype]

    @pytest.mark.skipif(
        not _TORCH_GE_2_6,
        reason="AOTInductor compile-and-package requires torch >= 2.6",
    )
    def test_pair_nequip_multirank_matches_single_rank(
        self, fake_model_training_session, device, multirank_tol
    ):
        config, tmpdir, env, model_dtype, model_source, structures = (
            fake_model_training_session
        )
        model_path = resolve_saved_model_path(tmpdir, model_source)

        sr_path = run_nequip_compile(
            model_path=model_path,
            tmpdir=tmpdir,
            env=env,
            mode="aotinductor",
            device=device,
            target="pair_nequip",
            output_prefix="pairnq_sr",
        )
        mr_path = run_nequip_compile(
            model_path=model_path,
            tmpdir=tmpdir,
            env=env,
            mode="aotinductor",
            device=device,
            target="pair_nequip_multirank",
            output_prefix="pairnq_mr",
        )

        # the `nequip_lammps` exchange ops must be registered before the AOTI package
        # loads (identity defaults here; LAMMPS installs the real exchange in-process)
        import nequip.nn._ghost_exchange_pair  # noqa: F401

        sr_model, sr_meta = load_aotinductor_model(sr_path, device)
        mr_model, mr_meta = load_aotinductor_model(mr_path, device)

        # metadata stamps: multirank capability marker (the C++ pair style's cross-check
        # against the declared inputs) and the exact comm-sizing feature width — the
        # single-rank artifact must carry neither
        assert mr_meta.get("pair_nequip_multirank") == "1"
        assert int(mr_meta.get("pair_nequip_feature_width", "0")) > 0
        assert sr_meta.get("pair_nequip_multirank") != "1"

        # processed metadata may hold parsed values (list / float) or raw strings
        type_names = sr_meta["type_names"]
        if isinstance(type_names, str):
            type_names = type_names.split()
        tm = ChemicalSpeciesToAtomTypeMapper(
            model_type_names=type_names,
            chemical_species_to_atom_type_map={s: s for s in type_names},
        )
        r_max = float(sr_meta["r_max"])

        for atoms in structures:
            data = AtomicDataDict.to_(
                tm(compute_neighborlist_(from_ase(atoms.copy()), r_max=r_max)),
                device,
            )
            num_atoms = data[AtomicDataDict.POSITIONS_KEY].shape[0]

            # nghost == 0: all atoms owned
            mr_data = dict(data)
            mr_data[AtomicDataDict.NUM_LOCAL_GHOST_NODES_KEY] = torch.tensor(
                [num_atoms, 0], dtype=torch.int64, device=data[
                    AtomicDataDict.POSITIONS_KEY
                ].device,
            )
            mr_data[AtomicDataDict.NUM_LOCAL_NODES_MARKER_KEY] = torch.zeros(
                num_atoms, dtype=torch.int64, device=data[
                    AtomicDataDict.POSITIONS_KEY
                ].device,
            )

            sr_out = sr_model(dict(data))
            mr_out = mr_model(mr_data)

            for field in (
                AtomicDataDict.PER_ATOM_ENERGY_KEY,
                AtomicDataDict.FORCE_KEY,
            ):
                np.testing.assert_allclose(
                    sr_out[field].detach().cpu().numpy(),
                    mr_out[field].detach().cpu().numpy(),
                    rtol=multirank_tol,
                    atol=multirank_tol,
                    err_msg=(
                        f"single-rank vs multirank mismatch for `{field}` "
                        f"(nghost=0 must be an identity transformation)"
                    ),
                )
