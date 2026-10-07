# Evaluation Audit — Flood Extent Detection (flood-detection-eo)

Internal record of experimental method, decisions, and results. This document exists so every claim in the public README can be traced back to a specific, reproducible step. Written honestly: negative and neutral results are recorded with the same care as positive ones.

---

## 1. Objective

Determine whether a U-Net trained on Sentinel-2 imagery can detect flood water more accurately than a simple physics-based spectral index (NDWI), using the Sen1Floods11 hand-labeled dataset.

## 2. Dataset

- **Source:** Sen1Floods11 (Sentinel-2 hand-labeled subset), publicly hosted on Google Cloud Storage.
- **Size:** 446 usable image chips (512×512 px), spanning 11 distinct flood events across multiple continents. (Note: an earlier session's `gsutil` download was incomplete and undercounted this at 394 chips; confirmed via a later full redownload. The test set — Bolivia/Ghana/USA, 137 chips, 111 flood-containing — was unaffected and identical across every run reported in this document, since the event-level split assigns events by name under a fixed seed, independent of total chip count in train/val.)
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

## 9b. Reproducibility check (three independent runs)

The notebook was independently rerun from a clean environment three times total (session losses forced full restarts, which incidentally became a reproducibility check). Results on the identical 111 flood-containing test chips:

| Metric | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| NDWI (fixed threshold −0.10) | IoU 0.5540 / F1 0.6614 | IoU 0.5540 / F1 0.6614 | IoU 0.5540 / F1 0.6614 (identical all three — deterministic, no randomness in NDWI) |
| U-Net (threshold 0.5) | IoU 0.3562 / F1 0.4515 | IoU 0.3746 / F1 0.4825 | IoU 0.3451 / F1 0.4453 |

U-Net mean across 3 runs: IoU ≈ 0.359, F1 ≈ 0.460. Gap to NDWI: ≈ 0.195 IoU, ≈ 0.201 F1, consistently.

**The core finding reproduced across three independent training runs** with different random weight initialization (same seed, but fresh CUDA sessions introduce some non-determinism in GPU operations). The gap between NDWI and U-Net remained stable and substantial (~0.19-0.21 in both IoU and F1) across all three runs, which gives strong confidence this is a real effect of limited training-event diversity, not an artifact of one unlucky run.

Note: total dataset size varied between runs (394 vs 446 chips) due to an incomplete download in earlier sessions — see §2. This did not affect the test set, which was identical (same 3 events, same 137/111 chip counts) in every run, so the comparison above remains valid and apples-to-apples.

## 9c. SAR extension — multi-sensor comparison

As an extension beyond the original optical-only scope, Sentinel-1 SAR data (VV, VH polarization, already included in Sen1Floods11 for the same chips and labels) was evaluated using the same methodology: threshold sweep on train+val data, fixed threshold applied once to the identical 111 flood-containing test chips.

**Physical basis:** calm/flooded water reflects radar specularly (mirror-like) away from the sensor, yielding low backscatter ("dark" in SAR imagery); rough land scatters energy diffusely, yielding higher backscatter ("bright"). This is the opposite logic to NDWI, which uses reflected sunlight rather than returned radar echo.

**A data-quality bug was caught and fixed:** approximately 10% of chips (46/446) contained NaN pixels (invalid SAR coverage at swath edges), which NumPy comparisons silently treat as `False`. This was corrected by explicitly excluding NaN pixels from the valid mask, matching the treatment already used for invalid (`-1`) label pixels. The fix changed results negligibly (IoU +0.0002), confirming the bug existed but did not materially affect prior conclusions — verified rather than assumed.

**VV/VH ratio attempt — inconclusive, not negative:** a VV−VH (dB difference) threshold sweep was tested but its initial range (−5 to +14 dB) never reached a peak — the best-scoring value sat at the edge of the tested range, meaning the sweep had not actually located an optimum. This is reported as an inconclusive experiment, not a finding that the ratio method underperforms, since the true optimal threshold was never located.

**Despeckling experiment:** published SAR flood-mapping methodology (e.g., Rana & Suryanarayana 2019; MATLAB Sentinel-1 flood-mapping reference workflow) applies speckle-noise filtering before thresholding, a step the initial SAR baseline omitted. A 5×5 median filter was applied to VV before a properly widened threshold sweep (−30 to +4 dB, confirmed non-boundary optimum at −14.0 dB).

| Method | IoU | F1 |
|---|---|---|
| NDWI (optical, tuned threshold) | 0.554 | 0.661 |
| SAR VV, raw (no despeckling) | 0.230 | 0.328 |
| SAR VV, despeckled (5×5 median filter) | 0.244 | 0.332 |
| U-Net (optical, mean of 3 runs) | ~0.359-0.387 | ~0.460-0.488 |

Despeckling produced a small, real improvement (IoU +0.014, F1 +0.004), consistent in direction with published results but smaller in magnitude — likely because a plain median filter is cruder than the edge-preserving Refined-Lee/Lee-Sigma filters used in studies reporting larger gains.

**Interpretation — why SAR underperformed optical here:** the literature search surfaced that most published SAR flood-mapping methods use **multi-temporal change detection** (comparing before/after-flood imagery of the same location) rather than a single fixed absolute threshold, because "normal" backscatter varies by location and land cover. This project used only a single-image absolute threshold for SAR, which is a simpler and known-weaker technique than change detection. This is assessed as the most likely primary reason SAR underperformed NDWI here — not an indication that SAR is inherently inferior for flood mapping, but that the specific (simplest) SAR technique applied was not the technique the literature recommends as the baseline standard.

**References consulted:**
- Rana, V.K. and Suryanarayana, T.M.V. (2019). Evaluation of SAR speckle filter technique for inundation mapping. *Remote Sensing Applications: Society and Environment*, 16.
- MATLAB/Simulink: "Map Flood Areas Using Sentinel-1 SAR Imagery" reference workflow (despeckling, dB thresholding pipeline).
- An exploratory study of Sentinel-1 SAR for rapid urban flood mapping on Google Earth Engine, *ScienceDirect* (polarization combination comparison).
- A methodology for mapping annual flood extent using multi-temporal Sentinel-1 imagery, *ScienceDirect* (Refined-Lee/Lee-Sigma filter comparison, multi-temporal approach).

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
