"""Shared interface class for experiment scripts. Allowing BP, MF, and later FF to plug into the same scripts."""
from __future__ import annotations

from abc import ABC, abstractmethod
# ABC defines abstracts classes so that they cannot be instantiated directly, they need to be run through a subclass that implements the 
# abstract methods. This is useful for defining a common interface for different models (BP, MF, FF) while allowing each model 
# to have its own specific implementation of the methods.

import torch
from torch import nn


class Classifier(nn.Module, ABC):
    """A 10-way MNIST classifier that owns its own optimiser(s) and training step.

    Owning the optimiser is deliberate: BP takes one global step per batch, while
    MF/FF take one local step per layer, so the step logic cannot live in a shared loop.
    For each of the three models, the training is implemented differently, but the interface is the same.
    """

    num_classes: int

    @abstractmethod
    def setup_optim(self, lr: float, weight_decay: float = 0.0) -> None:
        """Set up the optimiser(s) for the model. BP has one global optimiser, while MF/FF have one per layer."""

    @abstractmethod
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """MF and FF take one local step per layer, while BP takes one global step per batch—these must be implemented differently."""

    @abstractmethod
    def predict(self, x: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        """Class scores [B, C]; argmax is the prediction."""

    @abstractmethod
    def features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Post-activation output of every hidden layer. To be used for goodness computation in MF and FF, and for probing in BP at 
        per-layer level"""

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
