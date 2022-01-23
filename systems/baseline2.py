import math
import os
import pickle as pk
from importlib import import_module
from typing import Any

import omegaconf
import pytorch_lightning as pl
import schedulers as lr_schedulers
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from metrics import get_all_EERs
from utils import keras_decay


class System(pl.LightningModule):
    def __init__(
        self, config: omegaconf.dictconfig.DictConfig, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        _model = import_module("models.{}".format(config.model_arch))
        _model = getattr(_model, "Model")
        self.model = _model(config.model_config)
        self.configure_loss()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)

        return out

    def training_step(self, batch, batch_idx, dataloader_idx=-1):
        embd_asv_enrol, embd_asv_test, embd_cm_test, label = batch
        pred = self.model(embd_asv_enrol, embd_asv_test, embd_cm_test)
        loss = self.loss(pred, label)
        self.log(
            "trn_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=-1):
        """
        validation uses same-duration segments
        """
        embd_asv_enrol, embd_asv_test, embd_cm_test, key = batch
        pred = self.model(embd_asv_enrol, embd_asv_test, embd_cm_test)
        pred = torch.softmax(pred, dim=-1)

        return {"pred": pred, "key": key}

    def validation_epoch_end(self, outputs):
        log_dict = {}
        preds, keys = [], []
        for output in outputs:
            preds.append(output["pred"])
            keys.append(output["key"])

        preds = torch.cat(preds, dim=0)[:, 1]
        keys = torch.cat(keys, dim=0)
        sasv_eer, sv_eer, spf_eer = get_all_EERs(preds=preds, keys=keys)

        log_dict["dev_sasv_eer"] = sasv_eer
        log_dict["dev_sv_eer"] = sv_eer
        log_dict["dev_spf_eer"] = spf_eer

        self.log_dict(log_dict)

    def test_step(self, batch, batch_idx, dataloader_idx=-1):
        res_dict = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        return res_dict

    def test_epoch_end(self, outputs):
        log_dict = {}
        preds, keys = [], []
        for output in outputs:
            preds.append(output["pred"])
            keys.append(output["key"])

        preds = torch.cat(preds, dim=0)[:, 1]
        keys = torch.cat(keys, dim=0)
        sasv_eer, sv_eer, spf_eer = get_all_EERs(preds=preds, keys=keys)

        log_dict["eval_sasv_eer"] = sasv_eer
        log_dict["eval_sv_eer"] = sv_eer
        log_dict["eval_spf_eer"] = spf_eer

        self.log_dict(log_dict)


    def configure_optimizers(self):
        if self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=self.config.optim.lr,
                betas=self.config.optim.betas,
                weight_decay=self.config.optim.wd,
            )
        elif self.config.optimizer.lowe() == "sgd":
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=self.config.optim.lr,
                momentum=self.config.optim.momentum,
                weight_decay=self.config.optim.wd,
            )
        else:
            raise NotImplementedError("....")

        if self.config.optim.scheduler.lower() == "sgdr_cos_anl":
            assert (
                self.config.optim.n_epoch_per_cycle is not None
                and self.config.optim.min_lr is not None
                and self.config.optim.warmup_steps is not None
                and self.config.optim.lr_mult_after_cycle is not None
            )
            lr_scheduler = lr_schedulers.CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=len(self.train_dataloader())
                // self.config.ngpus
                * self.config.optim.n_epoch_per_cycle,
                cycle_mult=1.0,
                max_lr=self.config.optim.lr,
                min_lr=self.config.optim.min_lr,
                warmup_steps=self.config.optim.warmup_steps,
                gamma=self.config.optim.lr_mult_after_cycle,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        elif self.config.optim.scheduler.lower() == "reduce_on_plateau":
            assert (
                self.config.optim.lr is not None
                and self.config.optim.min_lr is not None
                and self.config.optim.factor is not None
                and self.config.optim.patience is not None
            )
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.config.optim.factor,
                patience=self.config.optim.patience,
                min_lr=self.config.optim.min_lr,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                    "monitor": "dev_sasv_eer",
                },
            }
        elif self.config.optim.scheduler.lower() == "keras":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: keras_decay(step)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "strict": True,
                },
            }

        else:
            raise NotImplementedError(".....")

    def setup(self, stage=None):
        """
        configures dataloaders.

        Args:
            stage: one among ["fit", "validate", "test", "predict"]
        """
        self.load_meta_information()
        self.load_embeddings()

        if stage == "fit" or stage is None:
            module = import_module("dataloaders." + self.config.dataloader)
            self.ds_func_trn = getattr(module, "get_trnset")
            self.ds_func_dev = getattr(module, "get_dev_evalset")
        elif stage == "validate":
            module = import_module("dataloaders." + self.config.dataloader)
            self.ds_func_dev = getattr(module, "get_dev_evalset")
        elif stage == "test":
            module = import_module("dataloaders." + self.config.dataloader)
            self.ds_func_eval = getattr(module, "get_dev_evalset")
        else:
            raise NotImplementedError(".....")

    def train_dataloader(self):
        self.train_ds = self.ds_func_trn(self.cm_embd_trn, self.asv_embd_trn, self.spk_meta_trn)
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.loader.n_workers,
        )

    def dev_dataloader(self):
        with open(self.config.dirs.sasv_dev_trial, "r") as f:
            sasv_dev_trial = f.readlines()
        self.dev_ds = self.ds_func_dev(
            sasv_dev_trial, self.cm_embd_dev, self.asv_embd_dev, self.spk_model_dev)
        return DataLoader(
            self.dev_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.loader.n_workers,
        )

    def eval_dataloader(self):
        with open(self.config.dirs.sasv_eval_trial, "r") as f:
            sasv_eval_trial = f.readlines()
        self.eval_ds = self.ds_func_eval(
            sasv_eval_trial, self.cm_embd_eval, self.asv_embd_eval, self.spk_model_eval)
        return DataLoader(
            self.eval_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.loader.n_workers,
        )

    def configure_loss(self):
        if self.config.loss.lower() == "bce":
            self.loss = F.binary_cross_entropy_with_logits
        if self.config.loss.lower() == "cce":
            self.loss = torch.nn.CrossEntropyLoss(
                weight=torch.FloatTensor(self.config.loss_weight)
            )
        else:
            raise NotImplementedError("!")

    def load_meta_information(self):
        with open(self.config.dirs.spk_meta + "spk_meta_trn.pk", "rb") as f:
            self.spk_meta_trn = pk.load(f)
        with open(self.config.dirs.spk_meta + "spk_meta_dev.pk", "rb") as f:
            self.spk_meta_dev = pk.load(f)
        with open(self.config.dirs.spk_meta + "spk_meta_eval.pk", "rb") as f:
            self.spk_meta_eval = pk.load(f)

    def load_embeddings(self):
        # load saved countermeasures(CM) related preparations
        with open(self.config.dirs.embedding + "cm_embd_trn.pk", "rb") as f:
            self.cm_embd_trn = pk.load(f)
        with open(self.config.dirs.embedding + "cm_embd_dev.pk", "rb") as f:
            self.cm_embd_dev = pk.load(f)
        with open(self.config.dirs.embedding + "cm_embd_eval.pk", "rb") as f:
            self.cm_embd_eval = pk.load(f)

        # load saved automatic speaker verification(ASV) related preparations
        with open(self.config.dirs.embedding + "asv_embd_trn.pk", "rb") as f:
            self.asv_embd_trn = pk.load(f)
        with open(self.config.dirs.embedding + "asv_embd_dev.pk", "rb") as f:
            self.asv_embd_dev = pk.load(f)
        with open(self.config.dirs.embedding + "asv_embd_eval.pk", "rb") as f:
            self.asv_embd_eval = pk.load(f)

        # load speaker models for development and evaluation sets
        with open(self.config.dirs.embedding + "spk_model_dev.pk", "rb") as f:
            self.spk_model_dev = pk.load(f)
        with open(self.config.dirs.embedding + "spk_model_eval.pk", "rb") as f:
            self.spk_model_eval = pk.load(f)
