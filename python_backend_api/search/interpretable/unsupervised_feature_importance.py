import os, sys
from pathlib import Path
from xml.sax.handler import feature_string_interning
import pandas as pd, numpy as np
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import scipy.io
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W


def construct_affinity_matrix(features_np, labels_np):
    # construct affinity matrix
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "supervised", "y": labels_np,
                "weight_mode": "heat_kernel", "k": 2,'t': 1}
    W = construct_W.construct_W(features_np , **kwargs_W)
    return W


def unsupervised_feature_importance_from_laplacian(features_np, labels_np, col_name_lst):

    # get affinity matrix
    W = construct_affinity_matrix(features_np, labels_np)

    # obtain the scores of features
    score = lap_score.lap_score(X=features_np, W=W)

    # sort scores and get idx
    col_pos_idx = list(np.argsort(score, 0))

    feature_importance = {col_name_lst[v]:score[v] for v in col_pos_idx}
    return feature_importance