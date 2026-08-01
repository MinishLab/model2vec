from __future__ import annotations

import logging

from torch import nn

from model2vec.train.similarity import StaticModelForSimilarity

logger = logging.getLogger(__name__)


class StaticModelForRegression(StaticModelForSimilarity):
    @staticmethod
    def _build_loss_function() -> nn.Module:
        """Construct the loss function used to train this model."""
        return nn.MSELoss()
