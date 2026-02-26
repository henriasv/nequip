# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import torch
import lightning
from lightning.pytorch.utilities.warnings import PossibleUserWarning
from hydra.utils import instantiate
from hydra.utils import get_method, get_class
from nequip.data import AtomicDataDict
from nequip.utils import RankedLogger

import warnings
from typing import Optional, Dict, List


logger = RankedLogger(__name__, rank_zero_only=True)


# metrics are already synced before logging, but Lightning still sends a PossibleUserWarning about setting sync_dist=True in self.logdict()
warnings.filterwarnings(
    "ignore",
    message=".*when logging on epoch level in distributed setting to accumulate the metric across.*",
    category=PossibleUserWarning,
)


_SOLE_MODEL_KEY = "sole_model"


class NequIPLightningModule(lightning.LightningModule):
    """:class:`~lightning.pytorch.core.LightningModule` for training, validating, testing and predicting with models constructed in the NequIP ecosystem.

    **Data**

    The ``NequIPLightningModule`` supports a single ``train`` dataset, but multiple ``val`` and ``test`` datasets.

    **Run Types and Metrics**

    - For ``train`` runs, users must provide ``loss`` and ``val_metrics``. The ``loss`` is computed on the training dataset to train the model, and requires each metric to have a corresponding coefficient that will be used to generate a ``weighted_sum``. This ``weighted_sum`` is the loss function that will be minimized over the course of training. ``val_metrics`` is computed on the validation dataset(s) for monitoring. Additionally, users may provide ``train_metrics`` to monitor metrics on the training dataset.
    - For ``val`` runs, users must provide ``val_metrics``.
    - For ``test`` runs, users must provide ``test_metrics``.

    **Logging Conventions**

    Logging is performed for the ``train``, ``val``, and ``test`` datasets.

    During ``train`` runs,
      * logging occurs at each batch ``step`` and at each ``epoch``,
      * there is only one training set, so no ``data_idx`` is used in the logging.

    For ``val`` and ``test`` runs,
      * logging only occurs at each validation or testing ``epoch``, i.e. one pass over the entirety of each validation/testing dataset,
      * there can be multiple validation and testing sets, so a zero-based ``data_idx`` index is used in the logging.

    Logging Format
      * ``/`` is used as a delimiter for to exploit the automatic grouping functionality of most loggers. Logged metrics will have the form ``train_{loss/metric}_{step/epoch}/{metric_name}`` and ``{val/test}{data_idx}_epoch/{metric_name}``. For example, ``train_loss_step/force_MSE``, ``train_metric_epoch/E_MAE``, ``val0_epoch/F_RMSE``, etc.
      * Note that this may have implications on how one would set the parameters for the `ModelCheckpoint <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html>`_ callback, i.e. if the name of a metric is used in the checkpoint file's name, the ``/`` will cause a directory to be created when instead a file is desired.
    """

    def __init__(
        self,
        model: Dict,
        num_datasets: Dict[str, int],
        optimizer: Optional[Dict] = None,
        lr_scheduler: Optional[Dict] = None,
        loss: Optional[Dict] = None,
        train_metrics: Optional[Dict] = None,
        val_metrics: Optional[Dict] = None,
        test_metrics: Optional[Dict] = None,
        # for caching training info
        info_dict: Optional[Dict] = None,
        # multi-head training
        per_head_loss_weights: Optional[Dict[str, float]] = None,
        cross_head_force_reg: Optional[Dict] = None,
    ):
        super().__init__()

        # save arguments to instantiate LightningModule from checkpoint automatically
        self.save_hyperparameters()

        # === instantiate model ===
        model_object = self._build_model(model)

        # === account for multiple models ===
        # contract:
        # - for multiple models, they must be in the form of a `ModuleDict` of `GraphModel`s
        # - if a single `GraphModel` is provided, we wrap it in a `ModuleDict`
        # - all models must have the same `type_names`

        # the reason for `hasattr(x, "is_graph_model")` and not just `isinstance(x, GraphModel)`
        # is to support `GraphModel` from a `nequip-package`d model (see https://pytorch.org/docs/stable/package.html#torch-package-sharp-edges)
        assert isinstance(model_object, torch.nn.ModuleDict) or hasattr(
            model_object, "is_graph_model"
        )
        if not isinstance(model_object, torch.nn.ModuleDict):
            model_object = torch.nn.ModuleDict({_SOLE_MODEL_KEY: model_object})
        self.model = model_object
        type_names_list = []
        for k, v in self.model.items():
            assert hasattr(v, "is_graph_model")
            type_names_list.append(v.type_names)
            logger.debug(f"Built Model Details ({k}):\n{str(v)}")
        assert all(
            [
                all(
                    [
                        name1 == name2
                        for (name1, name2) in zip(type_names_list[0], type_names)
                    ]
                )
                for type_names in type_names_list
            ]
        ), "If multiple models are used, they must have the same type names parameter."
        type_names = type_names_list[0]  # passed to `MetricsManager`s later

        # === optimizer and lr scheduler ===
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler

        # === instantiate MetricsManager objects ===
        # must have separate MetricsManagers for each dataloader
        # num_datasets goes in order [train, val, test, predict]
        self.num_datasets = (
            num_datasets
            if num_datasets is not None
            else {
                "train": 0,
                "val": 0,
                "test": 0,
                "predict": 0,
            }
        )

        assert self.num_datasets["train"] >= 1, (
            "at least one training dataset is required"
        )

        # multi-head: per-head loss weights (defaults to uniform)
        self.per_head_loss_weights = per_head_loss_weights

        # multi-head: cross-head force regularization
        self.cross_head_force_reg = cross_head_force_reg

        # == DDP concerns for loss ==

        # to account for loss contributions from multiple ranks later on
        # NOTE: this must be updated externally by the script that sets up the training run
        self.world_size = 1

        # == instantiate loss ==
        self.loss = instantiate(loss, type_names=type_names)
        if self.loss is not None:
            assert self.loss.do_weighted_sum, (
                "`coeff` must be set for entries of the `loss` MetricsManager for a weighted sum of metrics components to be used as the loss."
            )

            # set `dist_sync_on_step=True` for loss metrics
            # to ensure correct DDP syncing of loss function for batch steps
            for metric in self.loss.values():
                metric.dist_sync_on_step = True

        # == instantiate other metrics ==
        self.train_metrics = instantiate(train_metrics, type_names=type_names)
        # may need to instantate multiple instances to account for multiple val and test datasets
        self.val_metrics = torch.nn.ModuleList(
            [
                instantiate(val_metrics, type_names=type_names)
                for _ in range(self.num_datasets["val"])
            ]
        )
        self.test_metrics = torch.nn.ModuleList(
            [
                instantiate(test_metrics, type_names=type_names)
                for _ in range(self.num_datasets["test"])
            ]
        )

        # use "/" as delimiter for loggers to automatically categorize logged metrics
        self.logging_delimiter = "/"

        # for statefulness of the run stage
        self.register_buffer("run_stage", torch.zeros((1), dtype=torch.long))

    def _build_model(self, model_config: Dict) -> torch.nn.ModuleDict:
        """Constructs a ``torch.nn.ModuleDict[str, nequip.nn.GraphModel]`` from a pure Python dictionary.

        Subclasses that require more control over how the model is built can override this method.
        """
        # reason for following implementation instead of just `hydra.utils.instantiate(model)` is to prevent omegaconf from being a model dependency
        model_config = model_config.copy()  # make a copy because of `pop` mutation
        model_builder = get_method(model_config.pop("_target_"))
        model = model_builder(**model_config)
        return model

    def configure_optimizers(self):
        """"""
        # currently support 1 optimizer and 1 scheduler
        # potentially support N optimzier and N scheduler
        # (see https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers)
        optimizer_config = self.optimizer_config.copy()
        param_groups = optimizer_config.pop(
            "param_groups",
            {"_target_": "nequip.train.lightning._default_param_group_factory"},
        )
        param_groups = instantiate(param_groups, model=self.model)
        optimizer_class = optimizer_config.pop("_target_")
        optim = get_class(optimizer_class)(params=param_groups, **optimizer_config)

        if self.lr_scheduler_config is None:
            return optim

        def _instantiate_scheduler(scheduler_config: dict, optimizer):
            scheduler_config = dict(
                scheduler_config
            )  # just in case, because of pop mutation

            # NOTE: This assumes that nested schedulers always have a "schedulers" key
            inner_configs = scheduler_config.pop("schedulers", None)

            # Recursively instantiate inner schedulers if we use nested schedulers (e.g. ChainedScheduler, SequentialLR)
            if inner_configs is not None:
                inner_schedulers = [
                    _instantiate_scheduler(inner_config, optimizer)
                    for inner_config in inner_configs
                ]
                return instantiate(
                    scheduler_config, optimizer=optimizer, schedulers=inner_schedulers
                )

            # Base case: instantiate a regular scheduler
            return instantiate(scheduler_config, optimizer=optimizer)

        # instantiate lr scheduler object separately to pass the optimizer to it during instantiation
        lr_scheduler_config = dict(self.lr_scheduler_config.copy())
        scheduler_config = lr_scheduler_config.pop("scheduler")
        scheduler = _instantiate_scheduler(scheduler_config, optim)

        lr_scheduler = dict(instantiate(lr_scheduler_config))
        lr_scheduler.update({"scheduler": scheduler})
        return {"optimizer": optim, "lr_scheduler": lr_scheduler}

    def forward(self, inputs: AtomicDataDict.Type):
        """"""
        # enable grad for forces, stress, etc
        with torch.enable_grad():
            # multi-model subclasses will need to override this function
            return self.model[_SOLE_MODEL_KEY](inputs)

    @property
    def evaluation_model(self) -> torch.nn.Module:
        return self.model

    def process_target(
        self, batch: AtomicDataDict.Type, batch_idx: int, dataloader_idx: int = 0
    ) -> AtomicDataDict.Type:
        """"""
        # subclasses can override this function
        return batch.copy()

    def training_step(
        self, batch, batch_idx: int, dataloader_idx: int = 0
    ):
        """"""
        if isinstance(batch, dict) and self.num_datasets["train"] > 1:
            # Multi-head: CombinedLoader gives dict of batches keyed by str(index)
            total_loss = 0.0
            outputs = {}
            for head_key, head_batch in batch.items():
                target = self.process_target(head_batch, batch_idx, dataloader_idx)
                output = self(head_batch)
                outputs[head_key] = output

                # optionally compute training metrics (per-head prefixed)
                if self.train_metrics is not None:
                    with torch.no_grad():
                        train_metric_dict = self.train_metrics(
                            output,
                            target,
                            prefix=f"train_metric_step_head{head_key}{self.logging_delimiter}",
                        )
                    self.log_dict(train_metric_dict)

                # compute loss
                loss_dict = self.loss(
                    output,
                    target,
                    prefix=f"train_loss_step_head{head_key}{self.logging_delimiter}",
                )
                self.log_dict(loss_dict)

                head_loss = loss_dict[
                    f"train_loss_step_head{head_key}{self.logging_delimiter}weighted_sum"
                ]

                # apply per-head loss weight
                if self.per_head_loss_weights is not None:
                    weight = self.per_head_loss_weights.get(head_key, 1.0)
                    head_loss = head_loss * weight

                total_loss = total_loss + head_loss

            # Cross-head force regularization: penalize target heads' forces
            # toward a reference. Supports two config formats:
            #
            # Single-reference (all targets share one reference):
            #   cross_head_force_reg:
            #     reference_head: "0"    # or "zero" for L2 magnitude penalty
            #     target_heads: ["1", "2"]
            #     lambda: 0.01
            #
            # Per-target references (each target can have its own reference):
            #   cross_head_force_reg:
            #     terms:
            #       - reference_head: "0"
            #         target_head: "1"
            #         lambda: 0.01
            #       - reference_head: "zero"
            #         target_head: "2"
            #         lambda: 0.001
            if self.cross_head_force_reg is not None:
                # Normalize config to list of (ref, target, lambda) terms
                reg_terms = []
                if "terms" in self.cross_head_force_reg:
                    for term in self.cross_head_force_reg["terms"]:
                        reg_terms.append((
                            term["reference_head"],
                            term["target_head"],
                            term["lambda"],
                        ))
                else:
                    ref_key = self.cross_head_force_reg["reference_head"]
                    lam = self.cross_head_force_reg["lambda"]
                    for target_key in self.cross_head_force_reg["target_heads"]:
                        reg_terms.append((ref_key, target_key, lam))

                # Cache ref forces per reference head to avoid recomputation
                ref_forces_cache = {}

                for ref_key, target_key, lam in reg_terms:
                    use_zero_ref = ref_key == "zero"

                    # Get source batch (structures to evaluate on)
                    if use_zero_ref:
                        source_batch = None
                        for hk in batch:
                            source_batch = batch[hk]
                            break
                    else:
                        source_batch = batch.get(ref_key, None)

                    if source_batch is None:
                        continue

                    # Get reference forces (cached)
                    if not use_zero_ref and ref_key not in ref_forces_cache:
                        ref_output = outputs.get(ref_key, None)
                        if ref_output is None:
                            ref_output = self(source_batch)
                        ref_forces_cache[ref_key] = ref_output[
                            AtomicDataDict.FORCE_KEY
                        ].detach()

                    # Run target head on source batch structures
                    target_head_idx = int(target_key)
                    reg_batch = {
                        k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in source_batch.items()
                    }
                    reg_batch[AtomicDataDict.HEAD_KEY] = torch.full_like(
                        source_batch[AtomicDataDict.HEAD_KEY],
                        target_head_idx,
                    )
                    reg_output = self(reg_batch)
                    reg_forces = reg_output[AtomicDataDict.FORCE_KEY]

                    if use_zero_ref:
                        force_reg_loss = torch.mean(reg_forces**2)
                    else:
                        force_reg_loss = torch.nn.functional.mse_loss(
                            reg_forces, ref_forces_cache[ref_key]
                        )
                    total_loss = total_loss + lam * force_reg_loss

                    self.log(
                        f"train_loss_step{self.logging_delimiter}force_reg_head{target_key}",
                        force_reg_loss,
                    )

            return total_loss * self.world_size
        else:
            # Single-head: original path
            target = self.process_target(batch, batch_idx, dataloader_idx)
            output = self(batch)

            # optionally compute training metrics
            if self.train_metrics is not None:
                with torch.no_grad():
                    train_metric_dict = self.train_metrics(
                        output, target, prefix=f"train_metric_step{self.logging_delimiter}"
                    )
                self.log_dict(train_metric_dict)

            # compute loss and return
            loss_dict = self.loss(
                output, target, prefix=f"train_loss_step{self.logging_delimiter}"
            )
            self.log_dict(loss_dict)
            # In DDP training, because gradients are averaged rather than summed over nodes,
            # we get an effective factor of 1/n_rank applied to the loss. Because our loss already
            # manages correct accumulation of the metric over ranks, we want to cancel out this
            # unnecessary 1/n_rank term. If DDP is disabled, this is 1 and has no effect.
            loss = (
                loss_dict[f"train_loss_step{self.logging_delimiter}weighted_sum"]
                * self.world_size
            )
            return loss

    def on_train_epoch_end(self):
        """"""
        # optionally compute training metrics
        if self.train_metrics is not None:
            train_metric_dict = self.train_metrics.compute(
                prefix=f"train_metric_epoch{self.logging_delimiter}"
            )
            self.log_dict(train_metric_dict)
            self.train_metrics.reset()
        # loss
        loss_dict = self.loss.compute(
            prefix=f"train_loss_epoch{self.logging_delimiter}"
        )
        self.log_dict(loss_dict)
        self.loss.reset()

    def validation_step(
        self, batch: AtomicDataDict.Type, batch_idx: int, dataloader_idx: int = 0
    ):
        """"""
        target = self.process_target(batch, batch_idx, dataloader_idx)

        # === update basic val metrics ===
        output = self(batch)
        with torch.no_grad():
            metric_dict = self.val_metrics[dataloader_idx](
                output,
                target,
                prefix=f"val{dataloader_idx}_step{self.logging_delimiter}",
            )

        return metric_dict

    def on_validation_epoch_end(self):
        """"""
        # === reset basic val metrics ===
        for idx, metrics in enumerate(self.val_metrics):
            metric_dict = metrics.compute(
                prefix=f"val{idx}_epoch{self.logging_delimiter}"
            )
            self.log_dict(metric_dict)
            metrics.reset()

    def test_step(
        self, batch: AtomicDataDict.Type, batch_idx: int, dataloader_idx: int = 0
    ):
        """"""
        target = self.process_target(batch, batch_idx, dataloader_idx)

        # === update basic test metrics ===
        output = self(batch)
        with torch.no_grad():
            metric_dict = self.test_metrics[dataloader_idx](
                output,
                target,
                prefix=f"test{dataloader_idx}_step{self.logging_delimiter}",
            )
        metric_dict.update({f"test_{dataloader_idx}_output": output})

        return metric_dict

    def on_test_epoch_end(self):
        """"""
        # === reset basic test metrics ===
        for idx, metrics in enumerate(self.test_metrics):
            metric_dict = metrics.compute(
                prefix=f"test{idx}_epoch{self.logging_delimiter}"
            )
            self.log_dict(metric_dict)
            metrics.reset()


def _default_param_group_factory(model):
    return model.parameters()
