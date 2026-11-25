"""NumPy inference over the original network's exported weights."""
from functools import lru_cache
from pathlib import Path
import json
import math
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "static/studio/data"


@lru_cache(maxsize=1)
def load_decoder():
    metadata = json.loads((DATA / "model.json").read_text())
    weights = np.fromfile(DATA / "weights.bin", dtype="<f4").astype(np.float64)
    if weights.size != metadata["parameters"]:
        raise ValueError("Model weight size mismatch")
    return metadata, weights


def predict_vector(vector):
    metadata, weights = load_decoder()
    x = np.asarray(vector, dtype=np.float64)
    if x.shape[-1] != len(metadata["columns"]) or not np.isfinite(x).all():
        raise ValueError("Invalid model input")
    for layer in metadata["layers"]:
        ni, no, offset = layer["inputs"], layer["outputs"], layer["offset"]
        kernel = weights[offset:offset + ni * no].reshape(ni, no)
        bias = weights[offset + ni * no:offset + ni * no + no]
        x = x @ kernel + bias
        if layer["alpha"] is not None:
            x = np.where(x < 0, x * layer["alpha"], x)
    return x * metadata["std"] + metadata["mean"]


def predict_recipe(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    metadata, _ = load_decoder()
    fabric = payload.get("fabric")
    dyes, concentrations = payload.get("dyes"), payload.get("concentrations")
    if fabric not in metadata["fabrics"]:
        raise ValueError("Unknown fabric")
    if not isinstance(dyes, list) or not isinstance(concentrations, list) or not 1 <= len(dyes) <= 4 or len(dyes) != len(concentrations):
        raise ValueError("Provide one to four dyes and matching concentrations")
    if not all(isinstance(d, str) and d in metadata["dyes"] for d in dyes) or len(set(dyes)) != len(dyes):
        raise ValueError("Dyes must be unique model dye codes")
    if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 5 for v in concentrations):
        raise ValueError("Concentrations must be finite numbers between 0 and 5")
    vector = np.zeros(len(metadata["columns"]))
    for dye, value in zip(dyes, concentrations):
        vector[metadata["dyes"].index(dye)] = value
    vector[len(metadata["dyes"]) + metadata["fabrics"].index(fabric)] = 1
    return {"fabric": fabric, "dyes": dyes, "concentrations": concentrations,
            "predicted_lab": predict_vector(vector).tolist()}
