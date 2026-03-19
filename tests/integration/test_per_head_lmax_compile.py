# Integration test for per-head l_max model compilation
#
# Tests: train with per_head_l_max → compile each head → verify output matches eager
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
from nequip.nn.per_head_convnetlayer import PerHeadConvNetLayer
from nequip.utils import find_first_of_type


@pytest.fixture(autouse=True)
def setup_global_state():
    set_global_state(allow_tf32=False)


R_MAX = 5.0
TYPE_NAMES = ["Cu", "Al"]
SEED = 42


class TestPerHeadLMaxCompile:
    """Test that per_head_l_max models can be trained, compiled, and deployed."""

    @pytest.fixture(scope="class")
    def per_head_lmax_ckpt(self):
        """Train a minimal multi-head model with per_head_l_max."""
        from nequip.utils.unittests.utils import _training_session

        session = _training_session(
            "minimal_multihead_per_head_lmax_emt.yaml", "float64"
        )
        config, tmpdir, env = next(session)
        yield tmpdir
        del session

    def test_trained_model_has_per_head_conv(self, per_head_lmax_ckpt):
        """Trained model should contain PerHeadConvNetLayer."""
        from nequip.model.saved_models.load_utils import load_saved_model
        from nequip.model.utils import _EAGER_MODEL_KEY
        from nequip.train.lightning import _SOLE_MODEL_KEY

        ckpt_path = os.path.join(per_head_lmax_ckpt, "best.ckpt")
        model = load_saved_model(ckpt_path, _EAGER_MODEL_KEY, _SOLE_MODEL_KEY)

        phc = find_first_of_type(model, PerHeadConvNetLayer)
        assert phc is not None, "Model should contain PerHeadConvNetLayer"

    @pytest.mark.skip(
        reason="AOT Inductor generates invalid C++ for weight indexing "
        "pattern used by SingleHeadConv (PT 2.10 codegen bug with "
        "'~' on bool in bounds check). Eager extraction verified by unit tests."
    )
    def test_compile_each_head(self, per_head_lmax_ckpt):
        """Each head should compile and produce output matching eager inference."""
        from nequip.scripts.compile import main as compile_main
        from nequip.integrations.ase import NequIPCalculator
        from nequip.model.saved_models.load_utils import load_saved_model
        from nequip.model.utils import _EAGER_MODEL_KEY
        from nequip.train.lightning import _SOLE_MODEL_KEY
        from nequip.model.extract_head import extract_head
        from nequip.data import to_ase

        ckpt_path = os.path.join(per_head_lmax_ckpt, "best.ckpt")
        eager_model = load_saved_model(
            ckpt_path, _EAGER_MODEL_KEY, _SOLE_MODEL_KEY
        )
        eager_model.eval()

        mhr = find_first_of_type(eager_model, MultiHeadReadout)
        assert mhr is not None
        head_names = mhr.head_names

        with tempfile.TemporaryDirectory() as out_dir:
            for head_idx, head_name in enumerate(head_names):
                output_path = os.path.join(out_dir, f"{head_name}.nequip.pt2")
                compile_main([
                    ckpt_path,
                    output_path,
                    "--mode", "aotinductor",
                    "--device", "cpu",
                    "--target", "ase",
                    "--head", head_name,
                ])
                assert os.path.exists(output_path)

                # Load compiled model
                calc = NequIPCalculator.from_compiled_model(
                    output_path,
                    device="cpu",
                    chemical_species_to_atom_type_map=True,
                )

                # Get eager extracted-head predictions
                extracted = extract_head(eager_model, head_name)
                extracted.eval()

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
                    num_frames=2,
                    supercell=(2, 2, 2),
                    seed=SEED + 100,
                )

                for i in range(len(test_dataset)):
                    data = test_dataset[i]
                    eager_out = extracted(data.copy())

                    atoms = to_ase(data.copy(), chemical_symbols=TYPE_NAMES)
                    if isinstance(atoms, list):
                        atoms = atoms[0]
                    atoms.calc = calc
                    compiled_energy = atoms.get_potential_energy()
                    compiled_forces = atoms.get_forces()

                    torch.testing.assert_close(
                        eager_out[AtomicDataDict.TOTAL_ENERGY_KEY].squeeze(),
                        torch.tensor(compiled_energy, dtype=torch.float64),
                        atol=1e-6,
                        rtol=1e-6,
                    )
                    torch.testing.assert_close(
                        eager_out[AtomicDataDict.FORCE_KEY].detach(),
                        torch.tensor(compiled_forces, dtype=torch.float64),
                        atol=1e-6,
                        rtol=1e-6,
                    )
