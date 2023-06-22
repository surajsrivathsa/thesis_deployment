import os, sys
from pathlib import Path
from xml.sax.handler import feature_string_interning
import pandas as pd, numpy as np
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    Ridge,
    RidgeClassifier,
    LogisticRegression,
    SGDClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate


def permutation_based_feature_importance(
    model, X, y, stddev_weight=2, feature_col_labels_lst=[]
):
    r = permutation_importance(model, X, y, n_repeats=30, random_state=5)
    feature_importance = {}
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - stddev_weight * r.importances_std[i] > 0:
            print(
                "Feat Name: {}  : {} +/- {}".format(
                    feature_col_labels_lst[i],
                    r.importances_mean[i],
                    r.importances_std[i],
                )
            )
            feature_importance[feature_col_labels_lst[i]] = [
                r.importances_mean[i],
                r.importances_std[i],
            ]

    return feature_importance


def supervised_feature_importance_using_different_models(features_np, labels_np, feature_col_labels_lst):

    ridge_clf = RidgeClassifier(random_state=5).fit(
        X=features_np, y=np.ravel(labels_np)
    )
    ridge_regression_feature_importance = permutation_based_feature_importance(
        ridge_clf, features_np, labels_np, stddev_weight=1, feature_col_labels_lst=feature_col_labels_lst
    )

    logistic_clf = LogisticRegression(random_state=5).fit(
        X=features_np, y=np.ravel(labels_np)
    )
    logistic_regression_feature_importance = permutation_based_feature_importance(
        logistic_clf, features_np, np.ravel(labels_np), stddev_weight=0, feature_col_labels_lst=feature_col_labels_lst
    )

    sgd_clf = SGDClassifier(random_state=5).fit(X=features_np, y=np.ravel(labels_np))
    sgd_feature_importance = permutation_based_feature_importance(
        sgd_clf, features_np, np.ravel(labels_np), stddev_weight=0, feature_col_labels_lst=feature_col_labels_lst
    )

    return (
        ridge_regression_feature_importance,
        logistic_regression_feature_importance,
        sgd_feature_importance,
    )

