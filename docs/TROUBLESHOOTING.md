# Troubleshooting

Run commands from the repository root and activate the project virtual environment. Do not install project packages globally.

This Windows workspace defaults to Python 3.13 and also has a Python 3.14 test environment; use the isolated Python 3.12 `.venv312` described in the README for TensorFlow. Do not reuse the earlier incompatible `.venv`.

| Symptom                                        | Cause/check                                                      | Action                                                                                                                                                    |
| ---------------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Dataset directory not found`                  | `--dataset` must contain `train`, `val`, `test`                  | Extract the archive; pass the folder containing those splits                                                                                              |
| Missing class / empty folder                   | Expected class directory absent or archive extracted incorrectly | Confirm all six split/class folders and inspect their contents; do not substitute test for validation                                                     |
| Unreadable image warning                       | Corrupt/unsupported image bytes                                  | Replace from source or remove that file; loader skips it and reports count                                                                                |
| No readable files                              | Extension mismatch or corrupt images                             | Use jpg/jpeg/png/bmp/tif/tiff and verify images open normally                                                                                             |
| Inconsistent image dimensions                  | Source images vary                                               | Expected: preprocessing resizes each to configured 160×160 RGB; do not resize separately for prediction                                                   |
| TensorFlow import/version error                | Python/TensorFlow wheel mismatch                                 | Use Python 3.11/3.12 with project requirements; on Windows explicitly create with `py -3.12 -m venv .venv`; check platform support and available disk/RAM |
| Training memory error / slow CPU               | Batch/image size too large or CPU compute                        | Try `--batch-size 4`, smoke test; keep model/image size modest. A smoke test is not model evaluation                                                      |
| Training accuracy rises, validation falls      | Likely overfit, leakage, or split/distribution differences       | Inspect counts and patient overlap/provenance, use early stopping; gather valid data. Do not tune against test metrics                                    |
| Model not found                                | Training has not saved it or custom path differs                 | Run `python main.py train` and pass matching `--model` to all commands                                                                                    |
| Model load error                               | Incomplete, incompatible, or invalid Keras file                  | Check file permissions/size and TensorFlow version; train again from a clean run rather than renaming arbitrary files                                     |
| Reversed labels                                | Class ordering differs between dataset or external code          | This project fixes NORMAL=0, PNEUMONIA=1; inspect folder names and `src/config.py`                                                                        |
| TensorFlow unavailable in Python 3.13/3.14     | Requirements target supported Python range                       | Create a Python 3.11/3.12 environment (`py -3.12 -m venv .venv` on Windows) and install locally. Do not repeatedly retry unsupported wheel combinations   |
| Confusion matrix shape unexpected              | Labels absent in selected controlled samples                     | Production evaluator pins labels `[0,1]`; check test support and ensure both classes exist                                                                |
| Prediction differs from training preprocessing | Separate transforms or custom dimensions                         | Both paths call `preprocess_image`; check Settings/model input dimensions and model version                                                               |
| Streamlit can't find model                     | App started before training or wrong cwd                         | Run from repository root, train first; use the configured default model path                                                                              |
| Streamlit import/command missing               | Wrong environment or dependencies not installed                  | Activate `.venv`, install `requirements-dev.txt`, run `python -m streamlit run app.py` if needed                                                          |
| No graph generated                             | Workflow did not complete or history keys/artifact absent        | Read preceding error; training writes curves after fit, evaluation writes charts after test inference                                                     |
| Permission error saving artifacts              | Output/model folder is read-only or disk unavailable             | Choose writable `--output-dir`/`--model` path, check free disk space                                                                                      |
| Dataset download interrupted                   | Large 1.15 GB archive; unstable connection                       | Resume/re-download through dataset host UI/API; verify archive integrity before extraction; downloader is not bundled                                     |
| App shows no metrics                           | Evaluation has not run                                           | Run separate `python main.py evaluate`; the app never makes up results                                                                                    |
| Git says remote missing                        | Repository not configured                                        | Review `git remote -v`, add your own remote only if none exists. Never overwrite without permission                                                       |

## First diagnostics

1. `python --version` and `python -m pip show tensorflow` (use the venv's Python).
2. `python main.py --help` and `python main.py train --help`.
3. Inspect split/class folders and counts without moving test images.
4. Run `pytest -q` for controlled tests.
5. Inspect the first traceback and fix its root cause; don't suppress it or repeatedly rerun unchanged commands.

See the root README for installation and exact commands. Scores are not clinical probabilities; the system is educational only.
