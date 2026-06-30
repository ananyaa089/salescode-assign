import os
import time
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score
)

from features import extract_features, features_to_vector, FEATURE_NAMES

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REAL_DIR = os.path.join(DATA_DIR, "real")
SCREEN_DIR = os.path.join(DATA_DIR, "screen")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def load_dataset():
    X, y, paths = [], [], []
    t0 = time.time()
    for label, folder in [(0, REAL_DIR), (1, SCREEN_DIR)]:
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        for f in files:
            p = os.path.join(folder, f)
            feats = extract_features(p)
            X.append(features_to_vector(feats))
            y.append(label)
            paths.append(p)
    extract_time = time.time() - t0
    X = np.stack(X)
    y = np.array(y)
    print(f"Loaded {len(y)} images ({(y==0).sum()} real, {(y==1).sum()} screen)")
    print(f"Feature extraction: {extract_time*1000/len(y):.1f} ms/image average\n")
    return X, y, paths


def main():
    X, y, paths = load_dataset()

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_pred = cross_val_predict(clf, X, y, cv=cv, method="predict")
    cv_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    acc = accuracy_score(y, cv_pred)
    cm = confusion_matrix(y, cv_pred)
    auc = roc_auc_score(y, cv_proba)

    print("=" * 60)
    print("CROSS-VALIDATED RESULTS (5-fold, stratified, 100 images)")
    print("=" * 60)
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"ROC AUC  : {auc:.4f}")
    print("Confusion matrix (rows=true, cols=pred) [real, screen]:")
    print(cm)
    print()
    print(classification_report(y, cv_pred, target_names=["real", "screen"], digits=4))

    print("Per-fold accuracy:")
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        clf.fit(X[train_idx], y[train_idx])
        fold_acc = clf.score(X[test_idx], y[test_idx])
        print(f"  fold {i+1}: {fold_acc*100:.2f}%")

    wrong = np.where(cv_pred != y)[0]
    if len(wrong):
        print("\nMisclassified images:")
        for i in wrong:
            true_lbl = "real" if y[i] == 0 else "screen"
            pred_lbl = "real" if cv_pred[i] == 0 else "screen"
            print(f"  {os.path.basename(paths[i])}: true={true_lbl} pred={pred_lbl} score={cv_proba[i]:.3f}")
    else:
        print("\nNo misclassified images in cross-validation.")

    clf.fit(X, y)

    logreg = clf.named_steps["logisticregression"]
    print("\nFeature importance (logistic regression coefficients, standardized features):")
    for name, coef in sorted(zip(FEATURE_NAMES, logreg.coef_[0]), key=lambda t: -abs(t[1])):
        sign = "higher -> more screen-like" if coef > 0 else "higher -> more real-like"
        print(f"  {name:20s} {coef:+.3f}  ({sign})")

    t0 = time.time()
    for p in paths:
        feats = extract_features(p)
        vec = features_to_vector(feats).reshape(1, -1)
        clf.predict_proba(vec)
    total_t = time.time() - t0
    print(f"\nEnd-to-end latency: {total_t*1000/len(paths):.1f} ms/image (this CPU, single image at a time)")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved trained model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
