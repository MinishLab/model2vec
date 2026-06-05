from __future__ import annotations

import logging

from model2vec.train.lightning_modules import RegressionLightningModule
from model2vec.train.similarity import StaticModelForSimilarity

logger = logging.getLogger(__name__)


class StaticModelForRegression(StaticModelForSimilarity):
    _lightning_class = RegressionLightningModule
