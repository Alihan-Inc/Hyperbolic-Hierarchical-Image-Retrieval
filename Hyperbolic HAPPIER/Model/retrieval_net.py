import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Hypll Imports 
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.manifolds.euclidean import Euclidean
from hypll.tensors import ManifoldTensor
from hypll.tensors import TangentTensor
from typing import Union

import happier.lib as lib

from happier.models.get_pooling import get_pooling
from happier.models.get_backbone import get_backbone


def flatten(tens):
    if tens.ndim == 2:
        return tens.squeeze(1)
    if tens.ndim == 3:
        return tens.squeeze(2).squeeze(1)
    if tens.ndim == 4:
        return tens.squeeze(3).squeeze(2).squeeze(1)


class RetrievalNet(nn.Module):

    def __init__(
        self,
        backbone_name,
        embed_dim=512,
        normalize=True,
        norm_features=False,
        without_fc=False,
        with_autocast=True,
        pooling='default',
        projection_normalization_layer='none',
        pretrained=True,
        do_hyperbolic = True,
        clip_r = 1, 
        tau = 10, #1.0, TRY WITH 10
        manifold = Union[PoincareBall, Euclidean],
        **kwargs,
    ):
        super().__init__()

        norm_features = lib.str_to_bool(norm_features)
        without_fc = lib.str_to_bool(without_fc)
        with_autocast = lib.str_to_bool(with_autocast)
        do_hyperbolic = lib.str_to_bool(do_hyperbolic)

        self.embed_dim = embed_dim
        self.normalize = normalize
        self.norm_features = norm_features
        self.without_fc = without_fc
        self.with_autocast = with_autocast
        self.do_hyperbolic = do_hyperbolic
        self.clip_r = clip_r
        self.tau = tau
        self.manifold = manifold
        if with_autocast:
            lib.LOGGER.info("Using mixed precision")

        self.backbone, default_pooling, out_features = get_backbone(backbone_name, pretrained=pretrained, **kwargs)
        self.pooling = get_pooling(default_pooling, pooling)
        lib.LOGGER.info(f"Pooling is {self.pooling}")

        if self.norm_features:
            lib.LOGGER.info("Using a LayerNorm layer")
            self.standardize = nn.LayerNorm(out_features, elementwise_affine=False)
        else:
            self.standardize = nn.Identity()

        if not self.without_fc:
            self.fc = nn.Linear(out_features, embed_dim)
            lib.LOGGER.info(f"Projection head : \n{self.fc}")
        else:
            self.fc = nn.Identity()
            lib.LOGGER.info("Not using a linear projection layer")

        if self.do_hyperbolic:
            self.curvature = Curvature(value=1.0)
            self.ball = PoincareBall(c=self.curvature)
            self.prototypes = self.load_prototypes(
                prototypes_path=".....prototypes-128d-11318c.npy",
                num_classes=11318, 
                scale_factor=0.85, 
                curvature=1.0, 
                embedding=None,
                )[0]
            # self.prototypes = self.load_prototypes(
            # prototypes_path="....../entailment_cones_sop_full_128.npy",
            # num_classes=None, 
            # scale_factor=0.85, 
            # curvature=1.0, 
            # embedding=None,
            # )[0][0:11318]
             #self.prototypes = None
        else:
            self.manifold = Euclidean()


    def forward(self, X, return_before_fc=False) -> ManifoldTensor:
        with torch.cuda.amp.autocast(enabled=self.with_autocast or (not self.training)):
            X = self.backbone(X)
            X = self.pooling(X)

            X = flatten(X)
            X = self.standardize(X)
            if return_before_fc:
                return X

            X = self.fc(X)

            # If we use a Poincaré ball model, we (optionally) perform feature clipping 
            if self.clip_r is not None:
                X = self.clip_features(X)

            # and then map the features to the manifold.
            if self.do_hyperbolic:
                tangent_tensor = TangentTensor(data=X, man_dim=1, manifold=self.ball)
                X = self.ball.expmap(tangent_tensor).tensor
                if self.prototypes is not None:
                    out_feature = ManifoldTensor(data=X, man_dim=1, manifold=self.ball, requires_grad=True)
                    
                    # Ensure self.prototypes is a ManifoldTensor
                    if not isinstance(self.prototypes, ManifoldTensor):
                        self.prototypes = ManifoldTensor(data=self.prototypes, man_dim=1, manifold=self.ball)
                        
                    distance = self.ball.dist(self.prototypes, out_feature[:, None, :])
                    logits_cls = -distance * self.tau
                    return logits_cls
                else:
                    logits_cls = self.linear(X)
                    return logits_cls
                    
            # If we use a Euclidean manifold,
            # we simply normalize the features to lie on the unit sphere
            if isinstance(self.manifold, Euclidean):
                if self.normalize or (not self.training):
                    dtype = X.dtype
                    X = F.normalize(X, p=2, dim=-1).to(dtype)
                return X
    
    def clip_features(self, X: torch.Tensor) -> torch.Tensor:
        X_norm = torch.norm(X, dim=-1, keepdim=True) + 1e-5
        fac = torch.minimum(torch.ones_like(X_norm), self.clip_r / X_norm)
        return X * fac
    
    
    @staticmethod
    def load_prototypes(prototypes_path, num_classes, scale_factor=0.95, curvature=1.0,  embedding=None):
        '''
        This function loads the hyperbolic prototypes for the hierarchies
        '''
        prototypes = torch.from_numpy(np.load(prototypes_path)).float()
        prototypes = prototypes[:num_classes]
        prototypes = F.normalize(prototypes, p=2, dim=1)
        prototypes = prototypes.cuda() * scale_factor / math.sqrt(curvature)
        print(f"Using hyperbolic prototypes, {prototypes.shape} for the hierarchies")
    
        return prototypes, prototypes.shape[1]