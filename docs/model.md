# Model

The desktop app loads `models/fsl15/model.h5` with `models/fsl15/scaler.pkl` and `models/fsl15/labels.npy`. `models/fsl15/best.h5` is a different checkpoint of the same 15-class network (same byte size, different file hash) and is used only if `model.h5` is missing. `models/fsl11/` is an earlier 11-class run. The app does not load it unless you pass `--model fsl11` to the smoke command.

Both 15-class files were saved with Keras 3.13.2. This repo loads them with Keras 3.15.1 and TensorFlow 2.21.0. The scalers were pickled with scikit-learn 1.6.1, which is the version pinned in `pyproject.toml`.

## Architecture

Input shape `(30, 258)`, float32. `model.count_params()` reports **190,671** trainable parameters on `models/fsl15/model.h5`.

| Layer | Config | Output |
|---|---|---|
| Conv1D | 64 filters, kernel 3, relu, valid padding | `(28, 64)` |
| Dropout | 0.3 | |
| LSTM | 64 units, tanh, sequence out, L2 0.001 | `(28, 64)` |
| Dropout | 0.5 | |
| LSTM | 128 units, tanh, L2 0.001 | `(128,)` |
| Dropout | 0.5 | |
| Dense | 64, relu, L2 0.001 | `(64,)` |
| Dropout | 0.3 | |
| Dense | 15, softmax | `(15,)` |

The 11-class file uses the same stack with a final dense of 11.

## Landmark vector

258 floats per frame:

- pose: 33 landmarks × (x, y, z, visibility) = 132
- left hand: 21 × (x, y, z) = 63
- right hand: 21 × (x, y, z) = 63

If a part is missing, those slots are 0. If pose xyz is present without visibility, visibility is set to 1. The live window is the latest 30 frames. The LSTM runs on every second processed frame once that window is full. A gloss is recorded after it stays above 0.70 confidence for 10 inference steps, then a 12-step cooldown blocks a repeat.

## Labels

Shipped 15-class order (`models/fsl15/labels.npy`):

| Gloss | Tagalog shown in the UI |
|---|---|
| GOOD_AFTERNOON | Magandang hapon |
| NICE_TO_MEET_YOU | Ikinagagalak kitang makilala |
| YES | Oo |
| NO | Hindi |
| THANK_YOU | Salamat |
| HOW_ARE_YOU | Kumusta ka? |
| ONE | Isa |
| TWO | Dalawa |
| THREE | Tatlo |
| SIX | Anim |
| SEVEN | Pito |
| CORRECT | Tama |
| WRONG | Mali |
| IM_FINE | Okay lang ako |
| UNDERSTAND | Naiintindihan ko |

`HELLO` is in the Tagalog map because the demo script uses it. It is not an output of the 15-class model.

Legacy 11-class order: `GOOD_EVENING`, `NICE_TO_MEET_YOU`, `HELLO`, `HOW_ARE_YOU`, `THANK_YOU`, `YES`, `LONGANISA`, `TODAY`, `SUNDAY`, `YOURE_WELCOME`, `IM_FINE`.

The two scalers are both 258-feature `StandardScaler`s and they are not the same fit (the means differ). Each stays next to its own checkpoint.

## Dataset

`notebooks/train_fsl_15_conv1d_lstm.ipynb` and `notebooks/train_fsl_15_transformer_vsl.ipynb` download Kaggle `previlace/fsl-105` and keep the 15 glosses above. Landmark extraction samples 30 frames per clip with MediaPipe Holistic. The LSTM notebook is the trainer that matches the shipped architecture. The transformer notebook is a separate experiment and is not what the app loads.

Video notebooks in `notebooks/` cover capture, forcing clips to 30 frames, and landmark dumps. Expected training layout, from those notebooks:

```
videos/<gloss>/<id>.mp4
videos_30frames/<gloss>/<id>.mp4
MP_Data/<gloss>/<sequence>/<frame>.npy    # 258 floats
```

Clip counts and a train/test split are not stored in this repository.

An older script in git history (`model-processing/train_model.py` before this cleanup) trained on a different 15-word list (`ako`, `bakit`, `hi`, ...) and contained `no_sequences = 216`. That constant is not the size of the shipped 15-class set.

## Metrics

TODO: accuracy, per-class precision/recall, and a confusion matrix on a held-out split of the 15 glosses. Nothing in this tree is an evaluation set, so no accuracy is stated.

`python -m fsl.app --smoke` loads the checkpoint and runs a zero-filled 30×258 window. The printed top label is only a proof that softmax ran.

## Weights

Committed paths:

| File | Role |
|---|---|
| `models/fsl15/model.h5` | App checkpoint, 2,343,200 bytes |
| `models/fsl15/best.h5` | Different 15-class checkpoint, same size, fallback only |
| `models/fsl15/scaler.pkl` | Scaler for the 15-class model, 6,807 bytes |
| `models/fsl15/labels.npy` | 15 glosses |
| `models/fsl11/model.h5` | Legacy checkpoint, 2,340,128 bytes |
| `models/fsl11/scaler.pkl` | Scaler for the 11-class model, 6,807 bytes |
| `models/fsl11/labels.npy` | 11 glosses |

`scripts/download_weights.py` looks for GitHub Release `model-weights-v1` on `sc-lacanlale/fsl-sign-language-translator` and then on `sc-lacanlale/Codewarts`. While these files are in the clone it prints `present` and exits 0. The release does not exist yet. Upload instructions are in the module docstring of `src/fsl/weights.py`. Do not delete the committed weights until that release is uploaded.

The MediaPipe graph is not committed. `scripts/download_holistic_landmarker.py` fetches:

`https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task`

## History purge (not done)

Git history was not rewritten. Removing a file in this change only stops tracking it from the next commit. A later `git filter-repo` plus force-push, which the owner has to approve, would be required to drop the blobs below.

Removed from the tree in this cleanup, still in history:

| Path | Largest blob |
|---|---|
| `models/holistic_landmarker.task` | 13,683,609 bytes (13.05 MiB) |
| `models/NF1.h5.backup` | 2,927,224 bytes (2.79 MiB) |
| `models/group1-shard1of1.bin.backup` | 950,332 bytes (0.91 MiB) |
| `Logs/train/events.out.tfevents.1686185033.DESKTOP-H6NUQNL.8788.0.v2` | 727,333 bytes (0.69 MiB) |
| `models/model.json.backup` | 7,715 bytes |
| `.DS_Store`, `Logs/.DS_Store`, `video-processing/.DS_Store` | 8,196 + 6,148 + 6,148 bytes |

Already absent from `HEAD` before this cleanup, still in history:

| Path | Largest blob |
|---|---|
| `asl_webapp/asl_model.h5` | 8,461,012 bytes (8.07 MiB); an older blob of the same path is about 4.59 MiB |
| `models/fsl_105_model.h5` | 2,414,592 bytes (2.30 MiB) |
| `models/fsl_105_model_2.h5` | 2,414,592 bytes (2.30 MiB) |
| `apps/tanaw-mobile/package-lock.json` | 429,162 bytes |
| `apps/tanaw-mobile/assets/images/icon.png` | 393,493 bytes |
| `frontend/package-lock.json` | 232,493 bytes |

The 105-class claim in the old README referred to weights that are not in the working tree. The networks this app can load have 15 and 11 outputs.

The `.h5` files that the app still needs are listed above and stay in the tree on purpose.
