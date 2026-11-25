from pathlib import Path
import hashlib
import json
import subprocess
import h5py
import numpy as np
from PigPredict_app.runtime import predict_vector, predict_recipe

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "static/studio/data"


def test_all_exported_recipes_match_original_hdf5_and_browser():
    info = json.loads((DATA / "model.json").read_text())
    records = json.loads((DATA / "recipes.json").read_text())
    x = np.zeros((len(records), len(info["columns"])))
    for i, row in enumerate(records):
        x[i, row["dyes"]] = row["concentrations"]
        x[i, len(info["dyes"]) + row["fabric"]] = 1
    expected = x.copy()
    source = ROOT / "PigPredict_app/resource/keras_model/keras_decoder_model"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == info["source_sha256"]
    assert hashlib.sha256((DATA / "weights.bin").read_bytes()).hexdigest() == info["weights_sha256"]
    with h5py.File(source) as h:
        for layer in json.loads(h.attrs["model_config"])["config"]:
            cfg = layer["config"]
            if layer["class_name"] == "Dense":
                g = h["model_weights"][cfg["name"]][cfg["name"]]
                expected = expected @ np.array(g["kernel:0"], dtype=np.float64) + np.array(g["bias:0"], dtype=np.float64)
            elif layer["class_name"] == "LeakyReLU":
                expected = np.where(expected < 0, expected * cfg["alpha"], expected)
    expected = expected * info["std"] + info["mean"]
    np.testing.assert_allclose(predict_vector(x), expected, atol=1e-8, rtol=0)
    script = """
import fs from 'node:fs';
import {Decoder} from './static/studio/engine.js';
const m=JSON.parse(fs.readFileSync('static/studio/data/model.json'));
const b=fs.readFileSync('static/studio/data/weights.bin');
const d=new Decoder(m,b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));
const rows=JSON.parse(fs.readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify(rows.map(r=>d.predict(r.dyes,r.concentrations,r.fabric))));
"""
    result = subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps(records),
                            text=True, capture_output=True, check=True, cwd=ROOT)
    browser = np.array(json.loads(result.stdout))
    np.testing.assert_allclose(browser, expected, atol=1e-8, rtol=0)


def test_exported_catalog_contains_only_model_inputs_and_color():
    records = json.loads((DATA / "recipes.json").read_text())
    assert all(set(r) == {"lab", "fabric", "dyes", "concentrations"} for r in records)
    assert len(records) == 1417


def test_fabric_code_controls_only_its_one_hot_channel():
    payload = dict(dyes=["228-9", "261-8", "327"], concentrations=[.2, .4, .6])
    a = predict_recipe({**payload, "fabric": "FVF2184"})
    b = predict_recipe({**payload, "fabric": "FVF2429"})
    assert a["predicted_lab"] != b["predicted_lab"]
