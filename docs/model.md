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

The same seven files are GitHub Release [model-weights-v1](https://github.com/sc-lacanlale/Codewarts/releases/tag/model-weights-v1). `scripts/download_weights.py` tries `sc-lacanlale/fsl-sign-language-translator` first, then `sc-lacanlale/Codewarts`. While the files are in the clone it prints `present` and exits 0. A run with the local files removed downloaded each asset from the Codewarts release, matched the committed SHA-256, and `python -m fsl.app --smoke` loaded both checkpoints (15 classes / 190,671 parameters, and 11 classes / 190,411 parameters). The printed top label on that zero window is a load check, not accuracy.

The MediaPipe graph is not committed. `scripts/download_holistic_landmarker.py` fetches:

`https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task`

## History purge (approved, not run)

Git history has not been rewritten. The owner approved a purge, and it has to run after the cleanup pull request is merged. Running it earlier changes every commit hash and breaks that pull request. Do not force-push before the merge.

The current pack is 38.59 MiB (`git count-objects -vH`). The paths below are absent from the tip and account for 39,238,194 unique uncompressed blob bytes (37.42 MiB). Those same blobs occupy 31,731,261 bytes inside the current pack (30.26 MiB). A repack after the purge should leave a pack near 8 MiB. The shipped `models/fsl15/` and `models/fsl11/` files are not in this list.

| Path | Unique blob bytes |
|---|---|
| `models/holistic_landmarker.task` | 13,683,609 (1 blob) |
| `asl_webapp/` | 13,284,161 (6 blobs) |
| `*.backup` (`models/NF1.h5.backup`, `models/group1-shard1of1.bin.backup`, `models/model.json.backup`) | 3,885,271 (3 blobs) |
| `models/fsl_105_model.h5` | 2,414,592 (1 blob) |
| `models/fsl_105_model_2.h5` | 2,414,592 (1 blob) |
| `apps/tanaw-mobile/` | 2,188,388 (169 blobs), plus an 11-byte blob also stored under `frontend/` |
| `Logs/` | 727,333 (1 TensorBoard event file) |
| `frontend/` | 555,062 (42 blobs) |
| `services/` | 64,683 (30 blobs) |
| `.DS_Store`, `Logs/.DS_Store`, `video-processing/.DS_Store` | 20,492 (3 blobs) |

The exact `git filter-repo` commands and the re-clone steps are in the pull request description. Old session notes and `.pyc` files still in history are about 0.11 MiB packed and are not part of this purge.

The 105-class claim in the old README referred to weights that are not in the working tree. The networks this app can load have 15 and 11 outputs.

The `.h5` files that the app still needs are listed above and stay in the tree on purpose. Dropping them from history is a later choice, only after `scripts/download_weights.py` has been shown to restore them from the release.
