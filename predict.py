import sys
import os
import pickle

from features import extract_features, features_to_vector

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

_model = None


def _get_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def predict(image_path: str) -> float:
    feats = extract_features(image_path)
    vec = features_to_vector(feats).reshape(1, -1)
    model = _get_model()
    score = model.predict_proba(vec)[0, 1]
    return float(score)


if __name__ == "__main__":
    print(predict(sys.argv[1]))
