# Strawberry Ripeness Classification Under LED-Induced Domain Shift

**Module**: WMG9B7-15 — Artificial Intelligence & Deep Learning  
**Institution**: University of Warwick, WMG  
**Assessment**: Individual Assignment 2025/26 (70% weighting)

---

## The Problem

Dogtooth Technologies' Gen5 strawberry-harvesting robots run 24 hours a day. During daylight they use ambient sunlight; at night they switch to onboard LEDs. The problem is that the robots are trained on daytime images, and LED illumination shifts the spectral profile of the scene — ripe red berries look meaningfully different under narrow-band LED light than under broad-spectrum solar irradiance. This causes a measurable drop in classification accuracy at night.

This project quantifies that drop, compares how a traditional ML model and a deep learning model each handle it, and tests whether augmentation-based domain adaptation can recover the lost performance.

---

## Dataset

**Pastell, M. & Liakka, V. (2022).** *Strawberry dataset for object detection.* Zenodo.  
DOI: [10.5281/zenodo.6126677](https://doi.org/10.5281/zenodo.6126677) — CC BY 4.0

- 813 field images, YOLO bounding-box annotations
- Classes used: ripe (0), unripe (1) — peduncle excluded (scoped out)
- 2,691 patches extracted at 224×224px after 10% boundary padding
- Class split: 1,960 ripe / 731 unripe (2.68:1 imbalance)

The dataset downloads automatically on first run (~1.49 GB). No manual setup required.

---

## Experiment Design

Five experiments. Each answers a specific question.

| ID | Model | Train set | Test set | Question |
|---|---|---|---|---|
| E1 | ConvNeXt-Tiny | Day patches | test_day | What's the DL performance ceiling? |
| E2 | SVM (RBF) | Day features | test_day / test_night | How does traditional ML compare, and how does it hold up at night? |
| E3 | ConvNeXt-Tiny (E1 weights) | — | test_night | How much does the DL model degrade without adaptation? |
| E4 | — | — | — | Quantify the E1→E3 gap across all metrics |
| E5 | ConvNeXt-Tiny | Day + LED-augmented | test_night + test_day | Does augmentation-based adaptation close the gap? |

Night simulation is applied in LAB colour space: L\*×1.25, b\*+20. Deterministic for test sets, stochastic (L\*×U(1.10–1.40), b\*+U(10–40)) during E5 training. This approximates the spectral shift from solar to onboard LED without requiring real night-time labelled data.

---

## Results

| Experiment | Model | Test split | Macro-F1 | 95% CI | ROC-AUC | MCC |
|---|---|---|---|---|---|---|
| E1 — DL Day Baseline | ConvNeXt-Tiny | test_day | 0.9569 | [0.944, 0.969] | 0.9811 | 0.9141 |
| E2 — SVM Day Baseline | SVM (RBF) | test_day | 0.9045 | [0.875, 0.931] | 0.9666 | 0.8103 |
| E2 — SVM Night | SVM (RBF) | test_night | 0.7476 | [0.708, 0.784] | 0.8910 | 0.5062 |
| E3 — DL Day→Night (no adapt.) | ConvNeXt-Tiny | test_night | 0.9492 | [0.937, 0.962] | 0.9744 | 0.8985 |
| E5 — DL Adapted (night) | ConvNeXt-Tiny | test_night | 0.9515 | [0.938, 0.963] | 0.9880 | 0.9030 |
| E5 — DL Adapted (day stability) | ConvNeXt-Tiny | test_day | 0.9541 | [0.942, 0.965] | 0.9837 | 0.9082 |

Key takeaways:
- The SVM drops **15.7 F1 points** when tested at night (0.905 → 0.748). MCC collapses from 0.81 to 0.51 — barely better than chance on the minority class.
- ConvNeXt-Tiny drops only **0.77 F1 points** on the same shift (0.957 → 0.949), suggesting its learned representations are significantly more domain-robust.
- E5 domain adaptation recovers the night performance to 0.952 while *improving* day performance to 0.954 — no stability trade-off.
- All DL results are means over 3 independent seeds. Bootstrap 95% CIs are non-overlapping between DL and SVM at night.

---

## Architecture Decisions

**Deep Learning — ConvNeXt-Tiny** (Liu et al., CVPR 2022)  
27.8M parameters, ImageNet-1K pretrained. Stem + Stage 1 frozen; Stages 2–4 and classifier fine-tuned. Chosen over ResNet-18 (2015 architecture) and Swin-Tiny (Grad-CAM requires token-to-spatial reshaping — fragile). Training: AdamW + layer-wise LR decay (0.85/stage) + 3-epoch linear warmup + cosine schedule, label smoothing 0.1, 3 seeds.

**Traditional ML — SVM with RBF kernel**  
259-dimensional hand-crafted feature vector: HSV histogram (96), LBP (18), HOG (81), LAB a\*/b\* histogram (64). L\* channel deliberately excluded — it's the primary nuisance variable under LED shift. Hyperparameters via 5-fold GridSearchCV. Platt scaling for calibrated probabilities.

**Why not a bigger model?**  
The dataset has ~2,000 training patches. Larger models (ViT-B, EfficientNet-L) would overfit without significant additional regularisation and offer no Grad-CAM benefit. ConvNeXt-Tiny hits 96.4% accuracy on the day test set — there's no meaningful headroom to chase.

---

## Interpretability

Three complementary methods, covering different scopes:

- **Grad-CAM** (local) — spatial attention maps per image, comparing E1 model on day patches, E1 model on night patches, and E5 model on night patches. Shows how domain shift disrupts feature localisation.
- **t-SNE of GAP embeddings** (global) — 768-dim feature vectors from all test patches projected to 2D. E1 feature space clusters by domain (day/night) rather than class; E5 feature space clusters by class regardless of domain.
- **Reliability diagrams** (calibration) — quantile-binned calibration curves for E1 and E5. ECE: 0.075 (E1) and 0.080 (E5), Brier: 0.038 for both. Both models are well-calibrated at high confidence (>0.8), which is the operationally relevant regime for autonomous picking.

---

## How to Run

**The notebook is fully self-contained. Open it in Google Colab and click `Runtime → Run All`.**

Everything else — data download, dependency installation, patch extraction, training, evaluation, and figure generation — runs automatically. Idempotency guards throughout: re-running any cell skips completed work by checking for cached output files.

If outputs already exist (e.g., trained checkpoints are in `strawberry_project/models/`), training is skipped and metrics are loaded from JSON. Full fresh run takes ~90–120 minutes on a T4 GPU.

### Requirements

| | |
|---|---|
| Platform | Google Colab (recommended) or Python 3.10+ |
| GPU | T4 or better (CPU fallback available, ~8× slower) |
| RAM | 12 GB minimum |
| Disk | 5 GB free |
| Internet | Required on first run for dataset download |

### Optional: Google Drive caching

Set `USE_DRIVE_CACHE = True` in the configuration cell to persist the dataset and checkpoints to Drive across Colab sessions. Leave as `False` for standard single-session use.

---

## Repository Structure

```
strawberry_domain_shift.ipynb   ← main submission notebook
FINAL_PROJECT_REPORT.md         ← written report (all phases + results)
strawberry_project/
    models/                     ← trained checkpoints (E1 ×3 seeds, E5 ×3 seeds, SVM)
    results/
        figures/                ← all plots at 300 DPI (27 figures)
        metrics/                ← per-experiment JSON files + master CSV
    metadata.json               ← full audit trail (hyperparams, seeds, hardware)
```

---

## References

- Liu, Z. et al. (2022). A ConvNet for the 2020s. *CVPR 2022.*
- Selvaraju, R.R. et al. (2017). Grad-CAM: Visual explanations from deep networks. *ICCV 2017.*
- Pastell, M. & Liakka, V. (2022). Strawberry dataset for object detection. *Zenodo.* DOI: 10.5281/zenodo.6126677.
- Chicco, D. & Jurman, G. (2020). The advantages of the Matthews correlation coefficient. *BMC Genomics.*
- Wang, D. et al. (2021). Tent: Fully test-time adaptation by entropy minimisation. *ICLR 2021.*
