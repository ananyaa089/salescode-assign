import os
import time
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from features import extract_features, features_to_vector

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def main():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X, y, names = [], [], []
    t0 = time.time()
    for label, folder in [(0, "real"), (1, "screen")]:
        d = os.path.join(DATA_DIR, folder)
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            feats = extract_features(os.path.join(d, fn))
            X.append(features_to_vector(feats))
            y.append(label)
            names.append(fn)
    elapsed = time.time() - t0
    X = np.stack(X)
    y = np.array(y)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y, pred)
    cm = confusion_matrix(y, pred)

    print(f"Images evaluated : {len(y)}")
    print(f"Accuracy (on training set, NOT held-out -- see note.md for the honest")
    print(f"  cross-validated number): {acc*100:.2f}%")
    print("Confusion matrix [rows=true, cols=pred] order=[real, screen]:")
    print(cm)
    print()
    print(classification_report(y, pred, target_names=["real", "screen"], digits=4))
    print(f"Avg latency (feature extraction + inference): {elapsed*1000/len(y):.1f} ms/image")


if __name__ == "__main__":
    main()
