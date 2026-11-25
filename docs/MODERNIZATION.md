# Interface and engineering refresh — 2026

| Area | Original application | Refreshed application |
| --- | --- | --- |
| Entry point | Development placeholder page | Bilingual color formulation studio |
| Input | Django model form | Validated Lab inputs, fabric selection and measured sample presets |
| Results | First table row displayed | Up to eight ranked candidate recipes |
| Color comparison | Numeric results | Target/predicted swatches, Lab values and ΔE₇₆ |
| Interaction | Separate form and processing pages | Concentration sliders and numeric controls with live prediction |
| Export | Server-side CSV file | Download of the current selected and edited recipe |
| Runtime | TensorFlow 1.x graph and Keras 2.2 | Original weights executed in JavaScript and NumPy |
| Inverse search | Random starts and Adam | Deterministic starts and bounded damped least squares |
| Requests | Last database form row shared across requests | Explicit recipe payloads and stateless forward inference |
| Deployment | Django / IIS integration files | Django 5.2 application and buildless GitHub Pages demo |
| Documentation | Two-line README | English/Chinese README, architecture and demo walkthrough |

The original training notebooks, HDF5 model, dataset files, core algorithms and IIS configuration remain in the repository. The previous view implementation is preserved as `PigPredict_app/legacy_views.py`; active URLs point to the refreshed application. Legacy `/forms/` and `/training/` GET requests redirect to the studio.

## Validation

- 14 Python tests: all 1,417 exported recipes checked against independently loaded HDF5 weights, NumPy/browser parity, catalog fields, fabric effects, Django routes, valid/invalid API payloads and CSRF behavior.
- 4 Node tests: analytical derivatives versus central differences, deterministic and bounded inverse search across all three fabrics, invalid-input rejection and neutral sRGB conversion.
- `manage.py check`: no issues.
- Browser checks: initial recipe generation, candidate selection, manual concentration edits, reset, CSV download, Chinese/English switching, English after reload, invalid Lab feedback and fabric changes.
- Responsive checks: desktop and a 390 px phone viewport, including Chinese result cards, with no horizontal overflow.
- Export check: a manual concentration of 0.5 is retained in the downloaded CSV; the exported predicted ΔE₇₆ recomputes from its target/predicted Lab values.

The portability check uses an absolute Lab tolerance of `1e-8` for the equivalent double-precision evaluation of the archived arrays. Original TensorFlow training and the original random optimizer are not rerun by this refresh.
