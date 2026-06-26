import torch

from nequip.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin
from .model_modifier_utils import replace_submodules, model_modifier


class GhostExchangeModule(GraphModuleMixin, torch.nn.Module):
    """Base class for ghost atom exchange modules."""

    def __init__(
        self,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        irreps_in={},
    ):
        super().__init__()
        self.field = field

        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={field: irreps_in[field]},
            irreps_out={field: irreps_in[field]},
        )

    def forward(
        self,
        data: AtomicDataDict.Type,
        ghost_included: bool,
    ) -> AtomicDataDict.Type:
        raise NotImplementedError("Subclasses must implement forward method")


class NoOpGhostExchangeModule(GhostExchangeModule):
    """Base ghost exchange module that performs a no-op."""

    def forward(
        self,
        data: AtomicDataDict.Type,
        ghost_included: bool,
    ) -> AtomicDataDict.Type:
        return data

    @model_modifier(persistent=True, private=True)
    @classmethod
    def enable_LAMMPSMLIAPGhostExchange(cls, model):
        """Enable LAMMPS ML-IAP ghost exchange for inference in LAMMPS ML-IAP."""

        from ._ghost_exchange_lmp_mliap import LAMMPSMLIAPGhostExchangeModule

        def factory(old):
            new = LAMMPSMLIAPGhostExchangeModule(
                field=old.field,
                irreps_in=old.irreps_in,
            )
            return new

        return replace_submodules(model, cls, factory)

    @model_modifier(persistent=True, private=True)
    @classmethod
    def enable_PairNequIPGhostExchange(cls, model):
        """Enable native ``pair_nequip`` per-layer ghost exchange for multi-rank inference.

        Unlike :meth:`enable_LAMMPSMLIAPGhostExchange`, the exchange is a registered custom
        operator (``nequip_lammps::ghost_exchange``) rather than a Python ``autograd.Function``
        calling into a ``lmp_data`` object, so the model can be compiled with AOTInductor for
        the multi-rank ``pair_nequip`` target.
        """

        from ._ghost_exchange_pair import PairNequIPGhostExchangeModule

        def factory(old):
            new = PairNequIPGhostExchangeModule(
                field=old.field,
                irreps_in=old.irreps_in,
            )
            return new

        return replace_submodules(model, cls, factory)
