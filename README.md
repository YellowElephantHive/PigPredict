# PigPredict — Color Formulation Studio

**[Open the interactive demo](https://yellowelephanthive.github.io/PigPredict/) · [繁體中文](README.zh-TW.md)**

PigPredict turns a target textile color into candidate dye recipes. It combines manufacturing process data, a trained neural color decoder and inverse optimization to recommend dye combinations and concentrations for a selected fabric.

The original 2018–2019 project used Keras, TensorFlow and Django. The 2026 refresh preserves its trained model weights and turns the basic form-and-table workflow into an interactive studio with live color comparison, recipe editing, CSV export and a documented engineering workflow.

## Explore the studio

1. Enter a target **L*, a*, b*** and choose a fabric, or load a measured sample.
2. Compare up to eight candidate dye combinations, ranked by **model-predicted ΔE₇₆**.
3. Adjust dye concentrations and inspect the predicted color, Lab differences and total concentration.
4. Export the current recipe, including its target and predicted Lab values.

Every page load starts in **English**. The visible **English / 中文** switch translates the interface for the current visit. The layout supports desktop and mobile screens.

The demo executes the original trained network in the browser. A Web Worker performs inverse search without blocking the interface, and GitHub Pages hosts the complete interactive experience without a model-serving backend.

## Project contributions

- Connected dye recipes, fabric types and measured CIELAB colors into a formulation workflow.
- Built a forward neural decoder to model the relationship between dye concentrations and resulting color.
- Applied inverse optimization to search for concentrations that approach a target color while keeping the learned model fixed.
- Integrated the original workflow into Django, with archived IIS integration files and training notebooks retained in the repository.
- Added a browser inference engine, bounded recipe search, stateless Python inference, cross-runtime numerical tests and bilingual documentation in the interface refresh.

## Model at a glance

| Property | Value |
| --- | --- |
| Inputs | 44 dye concentrations + 3 fabric indicators |
| Outputs | L*, a*, b*, normalized during training |
| Architecture | `47 → 128 × 7 → 32 → 16 → 3` |
| Activation | LeakyReLU, α = 0.5 |
| Trained parameters | 109,923 |
| Supported fabrics | FVF2184, FVF2429, FVF2666 |
| Browser recipe catalog | 1,417 filtered and deduplicated historical records |
| Candidate ranking | Predicted Euclidean Lab distance, ΔE₇₆ |

The original inverse solver used TensorFlow 1.x and Adam with multiple random starts. The refreshed studio uses the same trained weights with deterministic initialization and bounded damped least-squares search. Model conversion and the new inverse solver are described in [Architecture](docs/ARCHITECTURE.md).

## Run locally

### Static studio

No Python packages or Node dependencies are required to view the demo:

```bash
git clone https://github.com/YellowElephantHive/PigPredict.git
cd PigPredict
python3 -m http.server 8771 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8771/`. Serve the files over HTTP so ES modules, model assets and the Web Worker can load.

### Django application and inference API

Use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 127.0.0.1:8771
```

On Windows, activate with `.venv\Scripts\activate`. Django serves the same studio and provides an optional NumPy forward-inference endpoint. The UI itself runs inverse search in the browser.

| Route | Behavior |
| --- | --- |
| `GET /` | Interactive studio |
| `GET /forms/`, `GET /training/` | Redirect old entry points to the studio |
| `GET /api/health/` | Model availability and dimensions |
| `POST /api/predict/` | Validated recipe-to-Lab inference; requires Django CSRF token |
| `/admin/` | Standard Django administration |

Example inference body:

```json
{
  "fabric": "FVF2184",
  "dyes": ["228-9", "261-8", "327"],
  "concentrations": [0.2, 0.4, 0.6]
}
```

The response returns the supplied recipe and `predicted_lab`. Concentrations must be finite numbers from 0 to 5. The workspace does not save user inputs to a shared database. Django administrative data uses `.runtime/db.sqlite3`, which is excluded from Git.

For a hosted Django service, set `DJANGO_DEBUG=0`, `DJANGO_SECRET_KEY` and `DJANGO_ALLOWED_HOSTS`, collect static files and use a WSGI server. See the [Django deployment checklist](https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/). GitHub Pages runs the static version and does not require these server settings.

## Verify and rebuild

Node.js 22 or later is used for numerical browser-engine tests; no npm install is needed.

```bash
python -m pip install -r requirements-dev.txt
python manage.py check
python -m pytest tests -q
node --test tests/engine.test.mjs

# Regenerate model/data exports when source artifacts change.
python scripts/export_model.py

# Rebuild the static site after editing the template or assets.
python scripts/build_demo.py
```

The 14 Python tests compare all **1,417 exported recipes** against independently loaded original HDF5 weights, verify browser/NumPy agreement at `1e-8` absolute Lab tolerance, and exercise routes, validation and CSRF behavior. Four Node tests cover analytical derivatives, deterministic search across all fabrics, concentration bounds, invalid inputs and display-color conversion.

GitHub Actions runs the checks and verifies reproducible generated assets. Manual browser checks cover language switching, English on reload, recipe selection and editing, stale targets, invalid Lab values, CSV export and a 390 px mobile layout.

## Repository map

```text
templates/PigPredict_app/studio.html   Shared Django/static interface
static/studio/
  app.js                             Interaction, bilingual copy and CSV export
  engine.js                          Original decoder + bounded inverse search
  worker.js                          Background recipe optimization
  data/                              Portable model and recipe catalog
PigPredict_app/
  runtime.py                         NumPy inference using the same weights
  views.py                           Current Django views and API
  legacy_views.py                     Preserved original request workflow
  core/, preprocess/, util/          Original modeling implementation
  main/                              Original training and analysis notebooks
  resource/                          Original Keras model and normalization
scripts/                             Reproducible model/static-site exporters
tests/                               Numerical and application tests
docs/                                GitHub Pages build and engineering notes
```

GitHub Pages serves the `/docs` directory. Template and asset changes should be made in their source directories, then exported with `scripts/build_demo.py`.

- [Architecture and algorithms](docs/ARCHITECTURE.md)
- [Two-minute demo walkthrough](docs/DEMO.md)
- [Modernization and validation record](docs/MODERNIZATION.md)
- [繁體中文專案說明](README.zh-TW.md)
