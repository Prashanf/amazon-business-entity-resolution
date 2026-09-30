"""
Model Training Module
Trains a Gradient Boosted Decision Tree (LightGBM) using GroupKFold cross-validation
and optimizes decision threshold for F_0.5 / F1 score.
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

from .feature_engineering import FEATURE_COLS

def train_lgbm_model(train_pairs_df, n_splits=5, model_save_path="models/lgbm_entity_resolver_v1.txt"):
    """Trains 5-fold LightGBM ensemble and saves the booster."""
    X = train_pairs_df[FEATURE_COLS]
    y = train_pairs_df["label"].values
    groups = train_pairs_df["s1_id"].values

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = n_neg / max(1, n_pos)

    print(f"Dataset: {len(X):,} pairs | Pos: {n_pos:,} | Neg: {n_neg:,} | Imbalance: 1:{scale_pos_weight:.2f}")

    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(X), dtype=np.float32)
    models = []

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, y_train = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        fold_scale = (len(y_train) - y_train.sum()) / max(1, y_train.sum())

        clf = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=fold_scale,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        val_preds = clf.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        models.append(clf)

        auc = roc_auc_score(y_val, val_preds)
        prauc = average_precision_score(y_val, val_preds)
        print(f"Fold {fold}/{n_splits} | Val ROC-AUC: {auc:.4f} | Val PR-AUC: {prauc:.4f}")

    # Threshold Optimization
    thresholds = np.linspace(0.10, 0.90, 81)
    best_f1 = 0.0
    best_thresh = 0.50
    for th in thresholds:
        pred_b = (oof_preds >= th).astype(int)
        f = f1_score(y, pred_b, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = th

    print(f"\nOptimal Decision Threshold: {best_thresh:.2f} (F1 = {best_f1*100:.2f}%)")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    models[0].booster_.save_model(model_save_path)
    print(f"Saved primary booster to: {model_save_path}")

    return models, best_thresh
