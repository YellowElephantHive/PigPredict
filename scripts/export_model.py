"""Export the archived Keras Dense/LeakyReLU network without loading TensorFlow.

The small metadata pickles are inspected as opcodes, never executed.
Only color, fabric and recipe fields are exported from the historical CSVs.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import pickletools
import struct
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "PigPredict_app"
DEST = ROOT / "static/studio/data"


def metadata():
    ops = pickletools.genops((APP / "resource/pickled/decoder_input_columns.pkl").read_bytes())
    columns = [arg for op, arg, _ in ops if op.name in ("BINUNICODE", "SHORT_BINUNICODE")
               and (arg.startswith("concentration_") or arg.startswith("FVF"))]
    stats = []
    for name in ("lab_mean", "lab_std"):
        ops = pickletools.genops((APP / f"resource/pickled/{name}.pkl").read_bytes())
        values = [struct.unpack("<ddd", arg) for op, arg, _ in ops
                  if op.name in ("SHORT_BINBYTES", "BINBYTES") and len(arg) == 24]
        if len(values) != 1:
            raise ValueError(f"Unexpected normalization format: {name}")
        stats.append(values[0])
    return columns, *stats


def export():
    DEST.mkdir(parents=True, exist_ok=True)
    columns, mean, std = metadata()
    dyes = [c.removeprefix("concentration_") for c in columns if c.startswith("concentration_")]
    fabrics = [c for c in columns if not c.startswith("concentration_")]
    source = APP / "resource/keras_model/keras_decoder_model"
    weights, layers, offset = [], [], 0
    with h5py.File(source) as h:
        config = json.loads(h.attrs["model_config"])["config"]
        for layer in config:
            kind, cfg = layer["class_name"], layer["config"]
            if kind == "Dense":
                if cfg["activation"] != "linear":
                    raise ValueError("Only linear Dense activations are supported")
                group = h["model_weights"][cfg["name"]][cfg["name"]]
                kernel = np.asarray(group["kernel:0"], dtype="<f4")
                bias = np.asarray(group["bias:0"], dtype="<f4")
                layers.append(dict(inputs=kernel.shape[0], outputs=kernel.shape[1], offset=offset, alpha=None))
                weights.extend([kernel.ravel(), bias.ravel()])
                offset += kernel.size + bias.size
            elif kind == "LeakyReLU":
                layers[-1]["alpha"] = cfg["alpha"]
            else:
                raise ValueError(f"Unsupported layer: {kind}")
    assert layers[0]["inputs"] == len(columns)
    binary = np.concatenate(weights).astype("<f4").tobytes()
    (DEST / "weights.bin").write_bytes(binary)
    records, seen, swatches = [], set(), {}
    for filename in ("hong_make_all_revised_3.csv", "hong_make_single_revised_3.csv"):
        with (APP / "data" / filename).open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["abort"] == "1" or row["布號"] not in fabrics or "-0" in row.get("LAB", ""):
                    continue
                try:
                    lab = [float(row[k]) for k in ("L", "a", "b")]
                    recipe = {}
                    for i in range(1, 5):
                        dye = row.get(f"染料_{i}", "")
                        if not dye:
                            continue
                        if dye not in dyes:
                            raise ValueError("Unknown dye")
                        idx, concentration = dyes.index(dye), float(row[f"濃度_{i}"])
                        recipe[idx] = recipe.get(idx, 0) + concentration
                    if not recipe or not all(math.isfinite(x) for x in [*lab, *recipe.values()]):
                        continue
                    if any(c < 0 or c > 5 for c in recipe.values()):
                        continue
                except (ValueError, KeyError):
                    continue
                pairs = sorted(recipe.items())
                fabric = fabrics.index(row["布號"])
                key = (fabric, *lab, *pairs)
                if key in seen:
                    continue
                seen.add(key)
                records.append(dict(lab=lab, fabric=fabric, dyes=[x[0] for x in pairs], concentrations=[x[1] for x in pairs]))
                if len(pairs) == 1 and abs(pairs[0][1] - 1.5) < 1e-6 and fabric == 1:
                    swatches[str(pairs[0][0])] = lab
    info = dict(version=1, source="keras_decoder_model", source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                weights_sha256=hashlib.sha256(binary).hexdigest(), columns=columns, dyes=dyes, fabrics=fabrics,
                mean=mean, std=std, layers=layers, parameters=offset, record_count=len(records),
                dye_swatches=swatches)
    (DEST / "model.json").write_text(json.dumps(info, separators=(",", ":")) + "\n")
    (DEST / "recipes.json").write_text(json.dumps(records, separators=(",", ":")) + "\n")
    print(json.dumps({"parameters": offset, "recipes": len(records), "fabrics": fabrics, "dyes": len(dyes), "bytes": len(binary)}))


if __name__ == "__main__":
    export()
