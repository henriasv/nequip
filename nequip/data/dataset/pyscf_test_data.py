# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""PySCF-based test dataset for multi-head training.

Generates small gas-phase molecules (H2O, CH4) at multiple theory levels
(HF, MP2) to test multi-head workflows. Requires the ``pyscf`` package.
"""
from typing import Union, Callable, List, Tuple, Optional

import numpy as np
import torch

import ase
import ase.build

from .. import AtomicDataDict
from ..dict import from_dict
from .base_datasets import AtomicDataset


def _compute_pyscf(atoms: ase.Atoms, method: str = "hf", basis: str = "sto-3g"):
    """Compute energy and forces with PySCF.

    Args:
        atoms: ASE Atoms object (non-periodic)
        method: 'hf' or 'mp2'
        basis: basis set name

    Returns:
        energy (float), forces (np.ndarray of shape [n_atoms, 3])
    """
    from pyscf import gto, scf, mp, grad

    # Build PySCF mol
    atom_str = ""
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.positions):
        atom_str += f"{sym} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}; "

    mol = gto.M(atom=atom_str, basis=basis, unit="Angstrom", verbose=0)

    if method.lower() == "hf":
        mf = scf.RHF(mol).run()
        energy = float(mf.e_tot)
        g = grad.RHF(mf).run()
        forces = -np.array(g.de)  # gradient -> force
    elif method.lower() == "mp2":
        mf = scf.RHF(mol).run()
        mp2 = mp.MP2(mf).run()
        energy = float(mp2.e_tot)
        g = grad.mp2.Gradients(mp2).run()
        forces = -np.array(g.de)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # PySCF forces are in Hartree/Bohr, convert to Hartree/Angstrom
    from pyscf.data.nist import BOHR

    forces = forces / BOHR

    return energy, forces


class PySCFMultiHeadTestDataset(AtomicDataset):
    """Test dataset using PySCF for different theory levels on small molecules.

    Follows the same pattern as :class:`EMTTestDataset`. Generates random
    perturbations of small molecules and computes energies and forces
    at the specified theory level.

    **Requires** ``pyscf`` to be installed.

    In PySCF default units (Hartree for energy, Hartree/Angstrom for forces).

    Args:
        transforms (List[Callable]): list of data transforms
        molecule (str): molecule type, one of ``"H2O"`` or ``"CH4"``
        method (str): quantum chemistry method, e.g. ``"hf"`` or ``"mp2"``
        basis (str): basis set (default ``"sto-3g"``)
        head_index (int): head index to stamp on each frame via ``HEAD_KEY``
        sigma (float): standard deviation of Gaussian noise on positions
        num_frames (int): number of structures to generate
        seed (int): random seed
    """

    def __init__(
        self,
        transforms: List[Callable] = [],
        molecule: str = "H2O",
        method: str = "hf",
        basis: str = "sto-3g",
        head_index: int = 0,
        sigma: float = 0.05,
        num_frames: int = 5,
        seed: int = 123456,
    ):
        super().__init__(transforms=transforms)
        self.molecule = molecule
        self.method = method
        self.basis = basis
        self.head_index = head_index
        self.sigma = sigma
        self.num_frames = num_frames
        self.seed = seed

        # Build base molecule
        if molecule.upper() == "H2O":
            base_atoms = ase.Atoms(
                "OHH",
                positions=[
                    [0.0, 0.0, 0.1173],
                    [0.0, 0.7572, -0.4692],
                    [0.0, -0.7572, -0.4692],
                ],
            )
        elif molecule.upper() == "CH4":
            base_atoms = ase.Atoms(
                "CH4",
                positions=[
                    [0.0, 0.0, 0.0],
                    [0.6276, 0.6276, 0.6276],
                    [0.6276, -0.6276, -0.6276],
                    [-0.6276, 0.6276, -0.6276],
                    [-0.6276, -0.6276, 0.6276],
                ],
            )
        else:
            raise ValueError(f"Unsupported molecule: {molecule}. Use 'H2O' or 'CH4'.")

        orig_pos = base_atoms.positions.copy()
        rng = np.random.default_rng(self.seed)
        self.data_list = []
        for _ in range(self.num_frames):
            base_atoms.positions[:] = orig_pos
            base_atoms.positions += rng.normal(
                loc=0.0, scale=self.sigma, size=base_atoms.positions.shape
            )
            energy, forces = _compute_pyscf(
                base_atoms, method=self.method, basis=self.basis
            )
            self.data_list.append(
                from_dict(
                    {
                        "pos": base_atoms.positions.copy(),
                        "atomic_numbers": base_atoms.get_atomic_numbers(),
                        "forces": forces,
                        "total_energy": energy,
                        AtomicDataDict.HEAD_KEY: np.array([self.head_index]),
                    }
                )
            )

    def __len__(self) -> int:
        return self.num_frames

    def _get_data_list(
        self,
        indices: Union[List[int], torch.Tensor, slice],
    ) -> List[AtomicDataDict.Type]:
        if isinstance(indices, slice):
            return self.data_list[indices]
        else:
            return [self.data_list[index] for index in indices]
