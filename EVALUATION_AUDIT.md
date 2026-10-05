# Evaluation Audit — Flood Extent Detection (flood-detection-eo)

Internal record of experimental method, decisions, and results. This document exists so every claim in the public README can be traced back to a specific, reproducible step. Written honestly: negative and neutral results are recorded with the same care as positive ones.

---

## 1. Objective

Determine whether a U-Net trained on Sentinel-2 imagery can detect flood water more accurately than a simple physics-based spectral index (NDWI), using the Sen1Floods11 hand-labeled dataset.

## 2. Dataset

- **Source:** Sen1Floods11 (Sentinel-2 hand-labeled subset), publicly hosted on Google Cloud Storage.
- **Size:** 394 usable image chips (512×512 px), spanning 11 distinct flood events across multiple continents.
- **Labels:** Per-pixel binary mask (1 = water/flood, 0 = non-water, −1 = invalid/cloud, excluded from all metrics).

## 3. Data split — leakage control

**Decision:** Split at the **flood-event level**, not the chip level. All chips from a given event were assigned entirely to train, validation, or test.

**Reason:** Chips from the same event share the same river, soil, and land cover. A random chip-level split would let the model see near-duplicate geography in both train and test, inflating apparent performance.

**Result:**
- Train: 7 events (partial regional coverage)
- Validation: subset of events, held separately
- Test: Bolivia, Ghana, USA (3 events, 137 chips) — never used for training or model selection

Verified programmatically that no event appears in more than one split.

## 4. Baseline — NDWI (physics-based)

**Formula:** NDWI = (Green − NIR) / (Green + NIR), using Sentinel-2 bands B3 and B8.

**Naive threshold (0.3, textbook value):** F1 = 0.086, IoU = 0.064 across all 394 chips. Effectively unusable — most flood water fell below this threshold, because flood water is turbid and spectrally different from clean open water.

**Threshold sweep (−0.30 to +0.55, step 0.05), evaluated across all 394 chips:** best threshold = **−0.10**, giving F1 = 0.6289, IoU = 0.5252. This became the baseline the model needed to beat.

**Interpretation:** The optimal threshold is negative, not the textbook value, because flood water's turbidity and mixing with vegetation/soil shifts its spectral signature away from clean-water assumptions.

## 5. Model — U-Net (from scratch)

- **Architecture:** Standard U-Net, 4 encoder/decoder levels, base filter width 32, ~7.77M parameters.
- **Input:** 4 channels (B2 Blue, B3 Green, B4 Red, B8 NIR), each percentile-normalized (2nd–98th percentile clip).
- **Loss:** Combined BCE + Dice, masked to exclude invalid (−1) pixels, to counter class imbalance (flood pixels are a minority in most chips).
- **Training:** 25 epochs, Adam optimizer (lr = 1e-3, ReduceLROnPlateau), batch size 4, horizontal/vertical flip augmentation, seed fixed (42) for reproducibility.
- **Model selection:** Best checkpoint chosen by validation F1 (0.8431 at threshold 0.30 — see §7).

## 6. Test-set result — first evaluation (default threshold 0.5)

Evaluated once, on the untouched 137-chip test set:

| Metric | NDWI (train+val+test, all 394 chips) | U-Net (test only) |
|---|---|---|
| F1 | 0.6289 | 0.5960 |
| IoU | 0.5252 | 0.4245 |

**Caught methodology error:** this comparison mixed two different populations — NDWI's number came from all 394 chips, U-Net's from only the 137 test chips. Not a fair comparison. Corrected in §7.

## 7. Fair, apples-to-apples comparison

Re-evaluated NDWI (same fixed threshold, −0.10) on the identical 111 test chips that actually contain flood water (26 of the 137 test chips have zero flood pixels and were evaluated separately — IoU/F1 are undefined-in-spirit for all-negative chips, see §8).

| Metric | NDWI (111 flood chips) | U-Net (111 flood chips, threshold 0.5) |
|---|---|---|
| IoU | 0.5540 | 0.3562 |
| F1 | 0.6614 | 0.4515 |

**On the same 111 chips, NDWI outperforms U-Net by a wide margin.**

## 8. Zero-flood chips — false-positive behavior

26 of 137 test chips (all from the Ghana event) contain no flood water at all.

- 6/26 predicted with zero false flood water (correct).
- 20/26 hallucinated some flood water (mean 5.85% of image area, generally mild, not catastrophic).

## 9. Threshold-tuning experiment (ruling out miscalibration)

Hypothesis: default threshold (0.5) might be suboptimal for U-Net's output distribution, the same way 0.3 was wrong for NDWI.

**Method:** Swept thresholds 0.10–0.90 on the **validation set only** (never touching test), selected best by F1, applied that single fixed threshold to test exactly once.

- Best validation threshold: 0.30 (validation F1 = 0.8431, IoU = 0.7287)
- Applied to test: F1 = 0.5240, IoU = 0.3550 — **worse** than the default-threshold test result (F1 = 0.5960).

**Conclusion:** Threshold miscalibration is not the primary cause of the gap. If it were, tuning on validation would have improved test performance. Instead, the large drop from validation (F1 0.84) to test (F1 0.52–0.60) under the *same* threshold indicates a **validation-to-test generalization gap** — the model is learning patterns specific to the events it was trained/validated on, not general flood-water spectral behavior.

## 9b. Reproducibility check (independent rerun)

The notebook was independently rerun from a clean environment (new session, same seed) after a session loss. Results:

| Metric | Run 1 | Run 2 (independent) |
|---|---|---|
| NDWI (flood chips, fixed threshold) | IoU 0.5540 / F1 0.6614 | IoU 0.5540 / F1 0.6614 (identical — deterministic) |
| U-Net (flood chips, threshold 0.5) | IoU 0.3562 / F1 0.4515 | IoU 0.3746 / F1 0.4825 |

**The core finding reproduced across two independent training runs** with different random weight initialization (same seed, but a fresh CUDA session introduces some non-determinism in GPU operations). The gap between NDWI and U-Net remained large and consistent (~0.18 IoU, ~0.18 F1) in both runs, which increases confidence that this is a real effect of limited training-event diversity, not an artifact of one unlucky training run.

## 10. Final finding

**With only 11 total flood events (and a training set drawn from roughly 7 of them), a U-Net trained end-to-end overfits to event-specific visual characteristics faster than it learns transferable flood-water spectral behavior. A simple physics-based index (NDWI) generalizes more reliably to unseen geography, because it directly encodes reflectance physics rather than learned visual patterns from a small sample of places.**

This is consistent with known behavior in small-data remote sensing: deep learning requires substantially more geographic diversity than 11 events to reliably outperform physics-based methods on out-of-region generalization.

## 11. What would change this conclusion (falsifiability)

- Training on a larger, more geographically diverse flood dataset (more than 11 events).
- Domain adaptation or transfer learning from a larger pretrained EO encoder.
- Explicit hard-negative training on dry/arid chips resembling the Ghana false-positive cases.

None of these have been tested. They are labeled as **future work**, not attempted improvements.

## 12. Reproducibility

- Seed fixed at 42 (Python, NumPy, PyTorch, CUDA).
- Event-level split verified disjoint by assertion.
- All code cells run in a single Kaggle notebook (GPU T4×2), no manual intervention between cells.
- Best model checkpoint saved (`best_unet_model.pth`); NDWI requires no trained artifact.
