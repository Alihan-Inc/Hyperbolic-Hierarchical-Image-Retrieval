import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import happier.lib as lib

# Hypll Imports 
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.manifolds.euclidean import Euclidean
from hypll.tensors import ManifoldTensor
from hypll.tensors import TangentTensor
from typing import Union


# adapted from :
# https://github.com/azgo14/classification_metric_learning/blob/master/metric_learning/modules/losses.py
class CELoss(nn.Module):

    def __init__(
        self,
        embedding_size,
        num_classes,
        temperature=None,
        hierarchy_level=None,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.temperature = temperature
        self.hierarchy_level = hierarchy_level

        self.linear = nn.Linear(embedding_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss() 


    def forward(self, embeddings, instance_targets, relevance_fn=None, **kwargs,):
        if self.hierarchy_level is not None:
            instance_targets = instance_targets[:, self.hierarchy_level]

        loss = self.loss_fn(embeddings, instance_targets)
        return loss

    def register_optimizers(self, opt, sch):
        self.opt = opt
        self.sch = sch
        lib.LOGGER.info(f"Optimizer registered for {self.__class__.__name__}")

    def update(self, scaler=None):
        if scaler is None:
            self.opt.step()
        else:
            scaler.step(self.opt)

        if self.sch["on_step"]:
            self.sch["on_step"].step()

    def on_epoch(self,):
        if self.sch["on_epoch"]:
            self.sch["on_epoch"].step()

    def on_val(self, val):
        if self.sch["on_val"]:
            self.sch["on_val"].step(val)

    def state_dict(self, *args, **kwargs):
        state = {"super": super().state_dict(*args, **kwargs)}
        state["opt"] = self.opt.state_dict()
        state["sch_on_step"] = self.sch["on_step"].state_dict() if self.sch["on_step"] else None
        state["sch_on_epoch"] = self.sch["on_epoch"].state_dict() if self.sch["on_epoch"] else None
        state["sch_on_val"] = self.sch["on_val"].state_dict() if self.sch["on_val"] else None
        return state

    def load_state_dict(self, state_dict, override=False, *args, **kwargs):
        super().load_state_dict(state_dict["super"], *args, **kwargs)
        if not override:
            self.opt.load_state_dict(state_dict["opt"])
            if self.sch["on_step"]:
                self.sch["on_step"].load_state_dict(state_dict["sch_on_step"])
            if self.sch["on_epoch"]:
                self.sch["on_epoch"].load_state_dict(state_dict["sch_on_epoch"])
            if self.sch["on_val"]:
                self.sch["on_val"].load_state_dict(state_dict["sch_on_val"])

    def __repr__(self,):
        repr = f"{self.__class__.__name__}(\n"
        repr = repr + f"    temperature={self.temperature},\n"
        repr = repr + f"    num_classes={self.num_classes},\n"
        repr = repr + f"    embedding_size={self.embedding_size},\n"
        repr = repr + f"    opt={self.opt.__class__.__name__},\n"
        repr = repr + f"    hierarchy_level={self.hierarchy_level},\n"
        repr = repr + ")"
        return repr
