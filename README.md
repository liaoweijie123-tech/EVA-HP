# EVA-HP: Hand-Pose Dataset and Preprocessing Pipeline

This repository contains the **EVA-HP** dataset and the preprocessing code used in the paper
*"A Method for Improving Fine-Grained Hand Motion Estimation Accuracy Based on SG-Former"*.

EVA-HP pairs **8-channel surface EMG** with **20-DOF hand-joint angles** continuously, and is
intended for research on continuous hand-kinematic estimation from sEMG.

---

## 1. Overview

| Item | Value |
|---|---|
| Subjects | **6** healthy adults (5 male, 1 female, aged 24–26), right hand only |
| Motions | **8** (index-finger flexion, sequential tapping, and six grasping motions) |
| Repetitions | **10** per motion |
| Motion duration | **4 s** per repetition; **8 s** for "sequential tapping" |
| Rest between repetitions | **3 s** |
| Recordings | **48** files (6 subjects × 8 motions) |
| sEMG | Myo armband, **8 channels**, **200 Hz**, right forearm proximal |
| Reference pose | Intel RealSense RGB + MediaPipe Hands, **21 landmarks → 20 joint angles**, ~30–35 Hz, interpolated to 200 Hz |

> **Total size of this release: ≈354 MB.**

---

## 2. Directory layout

```
EVA-HP_release/
├── README.md
├── data/
│   ├── raw/                     # 48 original recordings, one file per subject and motion
│   │   └── subject1/ … subject6/  {1..8}.h5
│   ├── split/                   # the same recordings divided into train/test parts
│   │   └── subject1/ … subject6/  {1..8}_{train,test}.hdf5
│   └── scalers/                 # per-subject normalisation statistics
│       ├── per_subject_scalers.npz
│       └── per_subject_scalers_raw.npz
└── code/
    └── process_eva_hp.py        # preprocessing: filtering, RMS, normalisation
```

---

## 3. File formats

### 3.1 Raw recordings — `data/raw/subject<N>/<M>.h5`

One HDF5 file per subject and motion, containing **two streams acquired asynchronously**
(they are *not* yet aligned):

| Path | Shape | dtype | Description |
|---|---|---|---|
| `emg/raw_data` | (N, 8) | int64 | Raw Myo ADC values, 200 Hz |
| `emg/filtered_data` | (N, 8) | float64 | Filtered sEMG as stored by the acquisition software |
| `emg/timestamps` | (N,) | float64 | Host monotonic clock, seconds |
| `hand/joint_angles` | (M, 20) | float64 | 20 joint angles, **degrees** |
| `hand/keypoints` | (M, 63) | float64 | 21 landmarks × (x, y, z); **x, y are image-normalised (0–1)**, z is MediaPipe relative depth |
| `hand/timestamps` | (M,) | float64 | Host monotonic clock, seconds |

`metadata` (group attributes): `action`, `dominant_hand`, `session_id`, `start_time`,
`subject_id`, `task_type`, `trial`.

`stats` (group attributes): `duration`, `emg_rate`, `hand_rate`, `hand_info`,
`recording_number`, `repeat_times`.

> **Note.** `hand/keypoints` stores the **2D image-normalised** landmarks, not the metric 3D world
> landmarks; the joint angles were computed online from the 3D world landmarks, which are **not**
> persisted. `hand_rate` is ~30–35 Hz while `emg_rate` is 200 Hz, hence the streams have
> different lengths (M ≠ N).

### 3.2 Split segments — `data/split/subject<N>/<M>_{train,test}.hdf5`

Each file stores one **already synchronised** segment at 200 Hz:

| Path | Shape | dtype | Description |
|---|---|---|---|
| `neuropose/timeseries` | (K,) | structured | see fields below |

The structured dataset has four fields, all of length K:

| Field | Shape | dtype | Description |
|---|---|---|---|
| `emg` | (K, 8) | float32 | sEMG, raw ADC values |
| `joint_angles` | (K, 20) | float32 | Joint angles in **degrees** |
| `skeleton` | (K, 63) | float32 | 21 landmarks × (x, y, z), image-normalised |
| `time` | (K,) | float64 | Monotonic timestamp, seconds |

### 3.3 Normalisation statistics — `data/scalers/per_subject_scalers.npz`

A NumPy archive with one entry per subject (`s1` … `s6`), each a dictionary of four arrays of
length 8 (EMG) or 20 (angles):

| Key | Shape | Description |
|---|---|---|
| `emg_min`, `emg_max` | (8,) | Per-channel min/max of the **RMS features**, fitted on the training split only |
| `angle_min`, `angle_max` | (20,) | Per-joint min/max of the **joint angles**, fitted on the training split only |

`per_subject_scalers_raw.npz` holds the same statistics computed from the unfiltered signal and
is not used in the paper.

---

## 4. Train / test split

The split is **within each recording**, along the repetition axis, and is **contiguous in time**:

* For every subject and motion, the 10 repetitions are divided into a **training part** and a
  **test part**, giving `<M>_train.hdf5` and `<M>_test.hdf5`.
* The **first 7 repetitions (≈70 % of the recording) form the training part** and the **last 3
  repetitions (≈30 %) the test part**. For example, `subject1/1.h5` has 14 170 samples
  (≈71 s), of which the first 9 707 samples (0–48.6 s) are training and the remaining 4 455
  samples (48.6–71.0 s) are test.
* The same rule is applied to all 48 recordings, including the 8-second motion
  (`subject1/6.h5`: 15 394 training samples covering 0–77.2 s, 6 864 test samples covering
  77.2–111.6 s).
* **The test repetitions are never used to fit the normalisation statistics**, which are computed
  from `*_train.hdf5` only (see `process_eva_hp.py`).

> **Note for reproduction.** The exact repetition boundaries are fixed by the files shipped in
> `data/split/`; using those files reproduces the paper's split exactly.

---

## 5. Preprocessing

`code/process_eva_hp.py` performs the following, in order.

**Stage 1 — synchronisation.** The two streams are trimmed to their common time interval and the
lower-rate angle stream is **linearly interpolated onto the 200 Hz sEMG time base**
(`utils/datasynchronizer.py` in the original project).

**Stage 2 — filtering (zero-phase, `scipy.signal.filtfilt`).**

| Signal | Filter |
|---|---|
| sEMG | 4th-order Butterworth **band-pass, 10–95 Hz** |
| sEMG | IIR **notch, 50 Hz, Q = 30** |
| Joint angles | 4th-order Butterworth **low-pass, 5 Hz** |

> **Note.** `filtfilt` is **non-causal** and is therefore suitable for offline preprocessing only.
> The real-time acquisition software uses the same filter designs implemented causally
> (`scipy.signal.lfilter` with persistent state).

**Stage 3 — RMS features.** The sEMG is converted to root-mean-square features with a
**100 ms window and a 5 ms increment** (= one sample at 200 Hz), so the feature sequence has the
same 200 Hz rate as the input and is aligned with the joint angle at the window end.

**Stage 4 — normalisation.** Per-subject min–max normalisation to [0, 1], with the statistics
fitted on the **training split only**.

**Stage 5 — model input.** The feature sequence is segmented into samples of
**500 ms window / 250 ms stride** (100 and 50 samples at 200 Hz), and the target for each sample
is the joint-angle vector at the window end.

---

## 6. How to use this release

1. **To reproduce the paper's split and preprocessing**, run `code/process_eva_hp.py` after setting
   the paths in its `CONFIG` dictionary (`split` directory, `scaler` output directory,
   `private_S3_Preprocessed` output directory). The script expects the directory layout produced by
   `data/split/`.
2. **To work directly with the released data**, load `data/split/**/*.hdf5`; they are already
   synchronised and do not require the raw files.
3. **To verify a model's normalised inputs**, use `data/scalers/per_subject_scalers.npz`.

Minimal example:

```python
import h5py, numpy as np

with h5py.File("data/split/subject1/1_train.hdf5", "r") as f:
    ts = np.array(f["neuropose/timeseries"])      # structured array
emg    = ts["emg"]            # (K, 8)  raw ADC values
angles = ts["joint_angles"]   # (K, 20) degrees
```

---

## 7. Notes and limitations

* **The joint angles are a vision-derived reference, not a calibrated ground truth.** MediaPipe
  landmark estimates were converted to angles with anatomical clipping and a One-Euro filter, but
  the reference was never validated against an optical motion-capture system or a data glove.
  Absolute angular accuracy is therefore unknown; the labels are appropriate for **relative**
  comparison between models evaluated on the same reference.
* **Occlusion.** Grasping motions involve objects partly covering the fingers, so the reference is
  least reliable exactly where the task is hardest.
* `hand/keypoints` contains **2D image-normalised** landmarks plus relative depth, not metric
  3D coordinates; abduction angles cannot be recomputed from them alone.
* **Model checkpoints and trained weights are not part of this release.**
* Subjects gave informed consent for the recording and for the use of the data for research.

---

## 8. License and citation

* **License:** `<to be completed — e.g. CC BY 4.0 for the data, MIT for the code>`
* **Persistent identifier:** `<Zenodo / Figshare DOI to be inserted>`
* **Citation:**

```bibtex
@article{liao_sgformer_semg,
  title   = {A Method for Improving Fine-Grained Hand Motion Estimation Accuracy Based on SG-Former},
  author  = {<authors>},
  journal = {Sensors},
  year    = {<year>},
  note    = {Dataset: EVA-HP}
}
```

If you use EVA-HP, please also cite the MediaPipe Hands paper for the landmark estimator and the
UmeTrack paper for the hand model used in the skinning visualisation.
