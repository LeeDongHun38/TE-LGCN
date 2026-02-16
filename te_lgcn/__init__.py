"""
Topic-Enhanced LightGCN (TE-LGCN)

A graph neural network recommendation system that combines:
- Collaborative filtering through LightGCN
- Semantic initialization via Doc2Vec embeddings
- Structural expansion via LDA topic modeling
"""

__version__ = "0.1.0"
__author__ = "TE-LGCN Research Team"

from te_lgcn.models import LightGCN, TELightGCN
from te_lgcn.data import RecommendationDataset, build_graph
from te_lgcn.features import Doc2VecExtractor, LDAExtractor
from te_lgcn.training import Trainer
from te_lgcn.evaluation import evaluate_model

__all__ = [
    "LightGCN",
    "TELightGCN",
    "RecommendationDataset",
    "build_graph",
    "Doc2VecExtractor",
    "LDAExtractor",
    "Trainer",
    "evaluate_model",
]
