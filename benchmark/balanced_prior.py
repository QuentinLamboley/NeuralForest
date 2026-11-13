"""Balanced forest with optional prevalence correction.

BalancedRandomForest estimates probabilities under an artificial 50:50
training prior. The prior-corrected form restores the observed training
prevalence without touching independent validation or test labels.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class BalancedPriorCorrectedRF(ClassifierMixin, BaseEstimator):
    def __init__(self, n_estimators=160, min_samples_leaf=1,
                 max_features="sqrt", random_state=42, n_jobs=1):
        self.n_estimators=n_estimators
        self.min_samples_leaf=min_samples_leaf
        self.max_features=max_features
        self.random_state=random_state
        self.n_jobs=n_jobs

    def fit(self,X,y):
        from imblearn.ensemble import BalancedRandomForestClassifier
        target=np.asarray(y)
        classes,count=np.unique(target,return_counts=True)
        if len(classes)!=2:
            raise ValueError("Prior correction requires binary classes")
        self.classes_=classes
        self.prevalence_=float(count[1]/count.sum())
        self.model_=BalancedRandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            bootstrap=False,
            replacement=False,
            sampling_strategy="all")
        self.model_.fit(X,y)
        return self

    def predict_proba(self,X):
        q=self.model_.predict_proba(X)
        if not np.array_equal(self.classes_,self.model_.classes_):
            raise RuntimeError("Class order changed during forest fitting")
        # Bayes odds update from the 1:1 training prior to the original prior.
        positive=np.clip(q[:,1],1e-9,1-1e-9)
        prevalence=np.clip(self.prevalence_,1e-9,1-1e-9)
        numerator=positive*prevalence
        denominator=numerator+(1-positive)*(1-prevalence)
        p=numerator/np.maximum(denominator,1e-14)
        return np.column_stack((1-p,p))
