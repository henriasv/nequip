# Integration test for multi-head model packaging and compilation
#
# Tests the full pipeline: train multi-head → package → compile single head → verify output
import tempfile
import os

import pytest
import torch

from nequip.utils.global_state import set_global_state
from nequip.data import AtomicDataDict
from nequip.data.dataset import EMTTestDataset
from nequip.data.transforms import (
    ChemicalSpeciesToAtomTypeMapper,
    NeighborListTransform,
)
from nequip.nn import MultiHeadReadout
from nequip.utils import find_first_of_type


@pytest.fixture(autouse=True)
def setup_global_state():
    set_global_state(allow_tf32=False)


R_MAX = 5.0
TYPE_NAMES = ["Cu", "Al"]
SEED = 42


class TestMultiHeadCompile:
    """Test that nequip-compile --head works for multi-head model deployment."""

    @pytest.fixture(scope="class")
    def multihead_ckpt(self):
        """Train a minimal multi-head model and return tmpdir with checkpoint."""
        from nequip.utils.unittests.utils import _training_session

        session = _training_session("minimal_multihead_emt.yaml", "float64")
        config, tmpdir, env = next(session)
        yield tmpdir
        del session

    def test_compile_without_head_raises(self, multihead_ckpt):
        """Compiling a multi-head model without --head should raise ValueError."""
        from nequip.scripts.compile import main as compile_main

        with tempfile.TemporaryDirectory() as out_dir:
            output_path = os.path.join(out_dir, "model.nequip.pth")
            with pytest.raises(ValueError, match="multi-head model"):
                compile_main([
                    os.path.join(multihead_ckpt, "best.ckpt"),
                    output_path,
                    "--mode", "torchscript",
                    "--device", "cpu",
                ])

    def test_compile_wrong_head_raises(self, multihead_ckpt):
        """Compiling with a nonexistent head name should raise ValueError."""
        from nequip.scripts.compile import main as compile_main

        with tempfile.TemporaryDirectory() as out_dir:
            output_path = os.path.join(out_dir, "model.nequip.pth")
            with pytest.raises(ValueError, match="not found"):
                compile_main([
                    os.path.join(multihead_ckpt, "best.ckpt"),
                    output_path,
                    "--mode", "torchscript",
                    "--device", "cpu",
                    "--head", "nonexistent_head",
                ])

    def test_compile_each_head(self, multihead_ckpt):
        """Each head should compile to a loadable model matching eager inference."""
        from nequip.scripts.compile import main as compile_main
        from nequip.ase import NequIPCalculator
        from nequip.model.saved_models.load_utils import load_saved_model
        from nequip.model.utils import _EAGER_MODEL_KEY
        from nequip.train.lightning import _SOLE_MODEL_KEY
        from nequip.model.extract_head import extract_head
        from nequip.data import to_ase

        ckpt_path = os.path.join(multihead_ckpt, "best.ckpt")

        # Load the eager multi-head model
        eager_model = load_saved_model(
            ckpt_path, _EAGER_MODEL_KEY, _SOLE_MODEL_KEY
        )
        eager_model.eval()

        mhr = find_first_of_type(eager_model, MultiHeadReadout)
        assert mhr is not None
        head_names = mhr.head_names

        with tempfile.TemporaryDirectory() as out_dir:
            for head_idx, head_name in enumerate(head_names):
                output_path = os.path.join(out_dir, f"{head_name}.nequip.pth")
                compile_main([
                    ckpt_path,
                    output_path,
                    "--mode", "torchscript",
                    "--device", "cpu",
                    "--target", "ase",
                    "--head", head_name,
                ])
                assert os.path.exists(output_path)

                # Load compiled model via NequIPCalculator
                calc = NequIPCalculator.from_compiled_model(
                    output_path,
                    device="cpu",
                    chemical_species_to_atom_type_map=True,
                )

                # Get eager extracted-head predictions for comparison
                extracted = extract_head(eager_model, head_name)
                extracted.eval()

                # Create test data for this head
                transforms = [
                    ChemicalSpeciesToAtomTypeMapper(
                        model_type_names=TYPE_NAMES,
                        chemical_species_to_atom_type_map=(
                            {"Cu": "Cu"} if head_idx == 0 else {"Al": "Al"}
                        ),
                    ),
                    NeighborListTransform(r_max=R_MAX),
                ]
                test_dataset = EMTTestDataset(
                    transforms=transforms,
                    element="Cu" if head_idx == 0 else "Al",
                    num_frames=3,
                    supercell=(2, 2, 2),
                    seed=SEED + 100,
                )

                for i in range(len(test_dataset)):
                    data = test_dataset[i]

                    # Eager extracted-head inference (no HEAD_KEY needed)
                    # Note: no torch.no_grad() - forces need autograd
                    eager_out = extracted(data.copy())
                    eager_energy = eager_out[AtomicDataDict.TOTAL_ENERGY_KEY]
                    eager_forces = eager_out[AtomicDataDict.FORCE_KEY]

                    assert torch.isfinite(eager_energy).all()
                    assert torch.isfinite(eager_forces).all()

                    # Compiled model inference via ASE calculator
                    atoms = to_ase(data.copy(), chemical_symbols=TYPE_NAMES)
                    if isinstance(atoms, list):
                        atoms = atoms[0]
                    atoms.calc = calc
                    compiled_energy = atoms.get_potential_energy()
                    compiled_forces = atoms.get_forces()

                    # Compare - squeeze eager outputs to match ASE scalar/array shapes
                    torch.testing.assert_close(
                        eager_energy.squeeze(),
                        torch.tensor(compiled_energy, dtype=eager_energy.dtype),
                        atol=1e-6,
                        rtol=1e-6,
                    )
                    torch.testing.assert_close(
                        eager_forces.detach(),
                        torch.tensor(compiled_forces, dtype=eager_forces.dtype),
                        atol=1e-6,
                        rtol=1e-6,
                    )

    def test_package_multihead_model(self, multihead_ckpt):
        """Packaging a multi-head model should succeed and include head_names metadata."""
        from nequip.scripts.package import main as package_main

        ckpt_path = os.path.join(multihead_ckpt, "best.ckpt")

        with tempfile.TemporaryDirectory() as out_dir:
            pkg_path = os.path.join(out_dir, "model.nequip.zip")
            package_main(["build", ckpt_path, pkg_path])
            assert os.path.exists(pkg_path)

            # Verify metadata includes head_names
            from nequip.model.saved_models.package import (
                _get_package_metadata,
                _suppress_package_importer_exporter_warnings,
            )

            with _suppress_package_importer_exporter_warnings():
                imp = torch.package.PackageImporter(pkg_path)
                metadata = _get_package_metadata(imp)

            assert "head_names" in metadata
            assert len(metadata["head_names"]) == 2

    def test_compile_from_package_with_head(self, multihead_ckpt):
        """Compile each head from a packaged multi-head model."""
        from nequip.scripts.package import main as package_main
        from nequip.scripts.compile import main as compile_main
        from nequip.ase import NequIPCalculator

        ckpt_path = os.path.join(multihead_ckpt, "best.ckpt")

        with tempfile.TemporaryDirectory() as out_dir:
            # Package first
            pkg_path = os.path.join(out_dir, "model.nequip.zip")
            package_main(["build", ckpt_path, pkg_path])

            # Compile each head from the package
            for head_name in ("Cu_head", "Al_head"):
                compiled_path = os.path.join(out_dir, f"{head_name}.nequip.pth")
                compile_main([
                    pkg_path,
                    compiled_path,
                    "--mode", "torchscript",
                    "--device", "cpu",
                    "--target", "ase",
                    "--head", head_name,
                ])
                assert os.path.exists(compiled_path)

                # Verify the compiled model loads and produces finite predictions
                calc = NequIPCalculator.from_compiled_model(
                    compiled_path,
                    device="cpu",
                    chemical_species_to_atom_type_map=True,
                )
                assert calc is not None
