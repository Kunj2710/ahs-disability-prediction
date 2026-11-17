"""Phase 4: is "undersampling beat SMOTE" about ranking, or about the threshold?

Two checks a reviewer asks for first:

1. Resampling inside the CV folds. Phases 1-3 resampled the training split once
   and then ran GridSearchCV on the resampled data, so tuning-time validation
   folds contained synthetic (SMOTE) or thinned (RUS) data. Here the sampler sits
   inside an imblearn Pipeline, so every validation fold is untouched real data.
   (The 30% test split was never resampled in any phase, so the reported test
   metrics were already on real households.)

2. Matched operating point. At a fixed 0.5 threshold, SMOTE models predict "Yes"
   far less often. Choosing each model's threshold on out-of-fold training
   predictions to hit the same recall shows whether SMOTE actually ranks
   households worse, or just sits at a different point on the same ROC curve.

Run from the repo root:  python src/phase4_operating_point_check.py
Writes results/phase4_operating_point.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
TARGET_RECALL = 0.736  # Yes TPR of the reported best model


def models():
    gb = GradientBoostingClassifier(learning_rate=0.01, max_depth=4, n_estimators=200, subsample=0.8, random_state=42)
    xgb = XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0, n_jobs=-1)
    return {
        "RUS + Gradient Boosting": Pipeline([("s", RandomUnderSampler(random_state=42)), ("m", gb)]),
        "SMOTE + Gradient Boosting": Pipeline([("s", SMOTE(random_state=42)), ("m", gb)]),
        "RUS + XGBoost": Pipeline([("s", RandomUnderSampler(random_state=42)), ("m", xgb)]),
        "SMOTE + XGBoost": Pipeline([("s", SMOTE(random_state=42)), ("m", xgb)]),
    }


def rates(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return pred[y == 1].mean(), 1 - pred[y == 0].mean()


def main():
    tr = pd.read_csv(ROOT / "outputs" / "initial_train.csv")
    te = pd.read_csv(ROOT / "outputs" / "initial_test.csv")
    X_tr, y_tr = tr.drop(columns="Class"), tr["Class"]
    X_te, y_te = te.drop(columns="Class"), te["Class"]
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    rows = []
    for name, pipe in models().items():
        oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        # threshold that gives TARGET_RECALL on out-of-fold training predictions
        t = np.quantile(oof[y_tr.to_numpy() == 1], 1 - TARGET_RECALL)
        p = pipe.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        yes05, no05 = rates(y_te, p > 0.5)
        yest, not_ = rates(y_te, p > t)
        rows.append({
            "Model": name,
            "CV AUC (sampler inside folds)": roc_auc_score(y_tr, oof),
            "Test AUC": roc_auc_score(y_te, p),
            "Yes TPR @0.5": yes05, "No TPR @0.5": no05, "MCC @0.5": matthews_corrcoef(y_te, p > 0.5),
            "Matched threshold": t,
            "Yes TPR @matched": yest, "No TPR @matched": not_, "MCC @matched": matthews_corrcoef(y_te, p > t),
        })
        print(f"done: {name}")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "phase4_operating_point.csv", index=False)
    print(out.round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
