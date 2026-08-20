# ECG Arrhythmia Classification

**Multi-label prediction of cardiac conditions from 12-lead ECG, comparing handcrafted signal features against learned representations.**

Five models on ~45,000 clinical ECG recordings: three built on 22 handcrafted
signal-processing features, two on the raw waveform. The interesting result is
not which one wins overall — it is that they win on *different metrics*, and
the gap between micro and macro F1 says more about the problem than any single
score does.

---

## The problem

Each recording carries zero or more SNOMED-CT diagnosis codes, so this is
multi-label, not multi-class: a patient can be in atrial fibrillation *and*
have a bundle branch block. The label distribution is severely long-tailed —
a handful of rhythms account for most of the mass, while many conditions
appear in a few dozen records out of 45,000.

That shape drives three decisions that run through the whole project:

**Micro and macro F1 are always reported together.** Micro is dominated by the
common rhythms; macro weights every condition equally. A model can improve one
while degrading the other, so neither is quoted alone.

**The decision threshold is tuned, not left at 0.5.** With labels this
imbalanced, the default cutoff is arbitrary. Each model's operating point is
selected by sweeping cutoffs against cross-validated predictions, breaking
ties toward recall — the safer direction to err when the cost of a missed
arrhythmia exceeds the cost of a false alarm.

**Thresholds are never selected on the test set.** The sweep runs on training
folds (or, for the CNN, on a validation split carved out of the training
portion). Selecting a cutoff on test data would leak it and inflate the score.

---

## Results

The CNN is the only model the original study evaluated on the held-out test
set. The four classical models were scored by cross-validation only, so their
numbers are **not** directly comparable to the CNN's and are labelled as such.

### Held-out test set

| Model | Cutoff | F1 micro | F1 macro | Precision micro | Recall micro |
|---|---|---|---|---|---|
| **CNN_250Hz** (1D ResNet on raw signal) | 0.35 | **0.7147** | 0.2078 | 0.7478 | 0.6845 |

### Cross-validated cutoff sweeps (training folds — not test scores)

| Model | Features | Best F1 micro | at cutoff |
|---|---|---|---|
| XGB_18F | 18 handcrafted | 0.5489 | 0.30 |
| XGB_100F | 100 signal PCA components | 0.5051 | 0.25 |
| SGD_18F | 18 handcrafted | 0.1249 | 0.50 |
| SGD_13C | 13 PCA of handcrafted | 0.1129 | 0.50 |

**What this shows.** The linear models fail on this problem — one-vs-rest SGD
never gets near the tree ensembles, and PCA-compressing the features first
(SGD_13C) makes it slightly worse rather than better. Gradient boosting on 18
interpretable features beats the same algorithm on 100 PCA components of the
raw waveform, which is a useful negative result: for this task, careful feature
engineering carried more signal than an unsupervised basis over the raw trace.

The CNN reaches the highest micro F1 but its macro F1 of 0.21 exposes the real
limitation — it performs well on frequent rhythms and poorly across the long
tail. `outputs/tables/*_per_label.csv` shows which conditions are never
predicted at all.

**On reproducibility.** Re-running the pipeline will not reproduce these
figures exactly. Ray Tune's search is seeded but its trial scheduling is not
fully deterministic under parallel execution, and some CUDA kernels are
nondeterministic. Expect small variation in the third decimal place.

---

## Repository layout

```
src/ecg/
  config.py            Paths, constants, seeding, YAML config loading
  data/
    wfdb.py            WFDB .mat/.hea reading into a padded signal tensor
    labels.py          SNOMED-CT code normalisation and condition lookup
    leads.py           Lead names parsed from headers, not assumed
    resample.py        Polyphase 500 Hz to 250 Hz resampling
  features/
    waveform.py        R-peak detection, PR/QRS/QT intervals, HRV statistics
    spectral.py        Welch PSD, LF/HF band powers, spectral entropy
    complexity.py      Approximate/sample entropy, DFA, Higuchi dimension
    extract.py         Assembles the 22-feature vector per record
  preprocessing.py     Imputation, label binarisation, train/test splitting
  decomposition.py     PCA on features; Incremental PCA on raw signal
  models/
    registry.py        Maps model names to build/tune/feature-source specs
    sklearn_models.py  SGD and XGBoost specs (all four registered here)
    cnn.py             1D ResNet, defined once
  tuning/ray_search.py One Ray Tune harness for every model
  evaluation/
    metrics.py         Multi-label metrics, micro and macro
    thresholds.py      Cutoff sweeps and operating-point selection
  viz/                 One chart theme, one set of result charts

scripts/               Runnable pipeline stages, in order
configs/               Data settings and per-model frozen hyperparameters
tests/                 pytest suite (31 tests, no dataset required)
```

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then, in order:

```bash
python scripts/01_build_dataset.py --limit 500
```

```bash
python scripts/02_extract_features.py
```

```bash
python scripts/03_train.py --model xgb_18f
```

`--limit` caps how many records are read; drop it for the full run. Every
model goes through the same script:

```bash
python scripts/03_train.py --model sgd_18f
```

The raw-signal models need the resampling and PCA stage first:

```bash
python scripts/04_signal_pca.py --components 100
```

```bash
python scripts/05_train_cnn.py --epochs 8
```

```bash
python scripts/06_compare_models.py
```

### Hyperparameters are frozen by default

`configs/models/*.yaml` holds the winning configuration from each search, so a
run reproduces in minutes instead of re-searching for hours. To search again:

```bash
python scripts/03_train.py --model xgb_18f --retune --trials 50
```

That prints the configuration it found and tells you to update the YAML —
deliberately, rather than by pasting values back into the code.

### Tests

```bash
python -m pytest
```

The suite runs without the dataset: it uses synthetic two-rhythm signals to
exercise the full path and to check that the feature extractor recovers a
known heart rate.

---

## Getting the data

The **ECG Arrhythmia Dataset** from PhysioNet — about 5.3 GB uncompressed, and
not included here.

<https://physionet.org/content/ecg-arrhythmia/1.0.0/>

```bash
wget -r -N -c -np https://physionet.org/files/ecg-arrhythmia/1.0.0/
```

Place it at `./ecg_data/`, or point `ECG_DATA_ROOT` at an existing copy:

```bash
export ECG_DATA_ROOT=/Volumes/external/ecg_data
```

The pipeline expects `ecg_data/WFDBRecords/`, `ConditionNames_SNOMED-CT.csv`
and `Remaining_DX_Codes_SNOMED_Labels.csv`.

---

## Method notes and limitations

**Fiducial points are approximated, not delineated.** P, QRS and T boundaries
are located by fixed physiological offsets from each detected R peak rather
than by a proper delineation algorithm. This is fast enough for 45,000 records
and robust to noise, but the intervals should be read as population-level
descriptors, not as clinical measurements on any individual trace.

**Features come from one lead.** Handcrafted extraction uses lead II only
(index 1), where the P wave is typically clearest. The raw-signal models use
all twelve.

**250 Hz for the raw-signal models.** Halves memory for the Incremental PCA
pass and the CNN, and the diagnostic content of a surface ECG sits well below
the 125 Hz Nyquist limit this affords. Resampling is polyphase, so decimation
anti-aliases rather than folding high-frequency noise into the band of
interest.

**Two features are dropped before modelling.** `higuchi_fractal_dimension` and
`dfa_scaling_exponent` are both RR-derived and undefined for short records, so
they carry the most missingness of the 22. The rest are median-imputed.

**Not a clinical tool.** This is a methods study on a public research dataset.
Nothing here is validated for diagnostic use.

---

## Licence

MIT — see [LICENSE](LICENSE).

The PhysioNet dataset carries its own licence; consult the source before
redistributing any part of it.
