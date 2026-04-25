# PROJECT HANDOFF PROMPT
## Strawberry Ripeness Classification Under LED-Induced Domain Shift
### University of Warwick — WMG9B7-15 AIDL Individual Assessment

---

## 1. WHO YOU ARE AND YOUR ROLE

You are continuing the work of an experienced AI/ML engineer and data scientist acting as **project lead** on a university individual assessment. You have full context of every decision made so far. Your job is to continue building the Jupyter notebook and (eventually) the critical reflection report, following the exact development procedure described in Section 5. You must behave as if you wrote every line of the existing notebook yourself.

**Deadline**: Monday 27 April 2026 by 12:00. Today is 25 April 2026. **You have approximately 2 days.**

---

## 2. ASSESSMENT CONTEXT

- **Module**: WMG9B7-15 — Artificial Intelligence & Deep Learning
- **Institution**: University of Warwick, WMG (Warwick Manufacturing Group)
- **Weight**: 70% of module grade
- **Submission**: Jupyter Notebook + Critical Reflection Report, submitted via Tabula
- **AI Policy**: AI Collaboration allowed — drafting, refining, evaluating — must adapt/extend work

### Two-Part Assessment
**Part 1 — Jupyter Notebook**
- Reproducible Python notebook with README cell
- Problem domain: Agriculture (NOT transportation/mobility)
- Must show: data preprocessing, model selection, training, evaluation
- Pre-trained architectures allowed with thoughtful configuration

**Part 2 — Critical Reflection** (max 2,800 words, +10% tolerance)
- ML vs DL comparison + justification for DL
- Key design decisions (architecture, params, evaluation)
- AI assistant usage disclosure
- Impact on individuals/orgs/environment/society
- Future trends & opportunities
- Academic + commercial references (both required)

### Learning Outcomes Being Assessed
- **LO1**: ML vs DL critical comparison (Part 2 Section 1, and Part 1 E1 vs E2 experiments)
- **LO2**: DL/AI solution evaluation + implications (ethical, ecological, societal)
- **LO3**: Emerging AI trends + research developments

### Marking Bands
- **80+**: Originality, sophisticated ML/DL comparison, exemplary implementation
- **70–79**: Critical comparison, comprehensive evaluation, ethical implications
- **60–69**: Clear comparison, interpretable evaluation, insight into implications
- **50–59**: Basic ML/DL explanation, functional pipeline, basic implications

---

## 3. THE PROBLEM

**Dogtooth Technologies (Cambridge, UK)** builds autonomous strawberry-harvesting robots. Their **Gen5 robots** operate 24 hours using onboard LED illumination at night. The robots were trained on **daytime images** but must operate under **LED lighting at night**, causing a **domain shift** — the spectral distribution of LED light differs fundamentally from solar irradiance, shifting the colour appearance of strawberries (warm reds become cooler/bluish under LEDs). This degrades ripeness classification accuracy significantly.

**This is a documented, unsolved problem**: Dogtooth public blog (gen5 launch) + Journal of Field Robotics (2024).

**Your task**: Build a controlled 5-experiment study that (1) quantifies this degradation, (2) compares DL vs traditional ML robustness to it, and (3) tests whether augmentation-based domain adaptation partially closes the gap.

**Domain**: Agriculture — specifically satisfies the "NOT transportation/mobility" constraint.

---

## 4. DATASET

### Dataset 1 (Primary — ONLY dataset used)
- **Source**: Pastell & Liakka, Luke Finland
- **Size**: 813 images, YOLO format annotations
- **Classes**: 0 = ripe, 1 = unripe, 2 = peduncle
- **License**: CC BY 4.0
- **Zenodo URL**: `https://zenodo.org/record/6126677/files/strawberries.zip`
- **Zenodo MD5**: `db8d5dcb4b8adebf1621788373fd3031`
- **Size**: 1.49 GB single zip

### Why Dataset 2 (Abdalla et al., Egypt) was excluded
D2 has different camera hardware, field conditions, and annotation schema — including it would introduce an **uncontrolled domain confound** separate from the LED shift. D2 exclusion is documented in the notebook.

### Task Scoping
- **Task**: Patch classification (NOT object detection — YOLO bounding boxes are used to crop patches, not for detection output)
- **Classes**: Binary — ripe (0) vs unripe (1) ONLY. **Peduncle (class 2) is excluded** — documented scoping decision
- **Patch size**: All patches resized to 224×224 RGB

### Data Splits (image-level stratified — CRITICAL for leakage prevention)
- **Test set**: 150 source images held out (stratified by dominant class) → never used during training
- **Train/Val**: Remaining ~663 images split 80/20 (stratified)
- Patches from the same source image go to exactly ONE split — enforced by assertion
- Split saved as CSV — **single source of truth for all experiments**

---

## 5. EXPERIMENT DESIGN (5 Experiments)

| ID | Name | Train Data | Test Set | Purpose | Primary Metric |
|---|---|---|---|---|---|
| **E1** | DL-Day Baseline | Day patches | test_day | DL performance ceiling | Macro-F1, ROC-AUC |
| **E2-Day** | SVM-Day Baseline | Day features | test_day | Traditional ML ceiling (LO1) | Macro-F1, ROC-AUC |
| **E2-Night** | SVM-Night | Day features | test_night | SVM fragility under shift | ΔF1 vs E2-Day |
| **E3** | DL-Day→Night | Day patches | test_night | Prove domain shift breaks DL | ΔF1 vs E1 |
| **E4** | Degradation | — | — | Quantify E1→E3 gap | E1−E3 all metrics |
| **E5** | DL-Adapted | Day+Night patches | test_night | Partial gap closure | ΔF1 vs E3 |

**XAI**: Grad-CAM comparing E1 model attention on day patches vs night patches vs E5 model attention — three-way comparison (Phase 9).

### Night Simulation (LED Domain Shift)
Applied in **LAB colour space**:
- **Fixed (test_night generation)**: L\*×1.25, b\*+20, a\*+0
- **Stochastic (E5 training augmentation)**: L\*×U(1.10,1.40), b\*+U(10,40), a\*+U(-10,+10)
- Physics rationale: solar → LED shifts spectral peaks from broad-spectrum to narrow blue (~450nm) + red (~630nm), compressing perceived colour separation between ripe/unripe

---

## 6. MODELS USED (CRITICAL — READ CAREFULLY)

### Two model architectures, three trained instances:

**Model 1 — SVM (Traditional ML, Experiments E2-Day and E2-Night)**
- **Architecture**: SVM with RBF kernel inside sklearn Pipeline (StandardScaler + SVC)
- **Feature vector**: 259-dimensional hand-crafted features:
  - HSV histogram: 32 bins × 3 channels = 96 features (indices 0:96)
  - LBP (uniform, radius=2, n_points=16): 18 features (indices 96:114)
  - HOG (3×3 cells, 9 orientation bins): 81 features (indices 114:195)
  - LAB a\*/b\* histogram: 32 bins × 2 channels = 64 features (indices 195:259). **L\* excluded** — primary LED nuisance variable
- **Hyperparameters**: GridSearchCV 5-fold StratifiedKFold, C=[0.01,0.1,1,10,100], gamma=['scale',0.01,0.001], scoring='f1_macro', class_weight='balanced'
- **Status**: FULLY IMPLEMENTED AND COMPLETE (Phases 5.1–5.4)
- **Global**: `SVM_MODEL` = fitted Pipeline; `E2_DAY_METRICS`, `E2_NIGHT_METRICS` dicts saved to JSON

**Model 2 — ConvNeXt-Tiny (Deep Learning, Experiments E1, E3, E5)**
- **Architecture**: ConvNeXt-Tiny (Liu et al., *A ConvNet for the 2020s*, CVPR 2022, Meta AI)
- **Pretrained weights**: ImageNet-1K (IMAGENET1K_V1) via torchvision
- **Why ConvNeXt-Tiny (not ResNet-18)**: ResNet-18 is 2015 — not enterprise-level. ConvNeXt-Tiny (CVPR 2022) implements transformer design principles (depthwise 7×7 conv, LayerNorm, GELU, inverted bottleneck) in a CNN, achieving ViT-competitive accuracy with standard Grad-CAM compatibility. 82.1% ImageNet Top-1. Safe on T4: 2.8 GB at bs=64.
- **Why not Swin-Tiny**: requires bs=32 on T4, Grad-CAM token reshaping from sequence to 2D (fragile under deadline), higher VRAM risk for 3 seeds
- **Freezing strategy**: `features[0]` (stem) + `features[1]` (Stage 1, 96ch, 56×56) FROZEN; `features[2]–[7]` + `classifier` TRAINABLE
- **Classifier head**: `Linear(768, 2)` replacing original `Linear(768, 1000)`
- **Grad-CAM target**: `model.features[-1]` (Stage 4, output [B, 768, 7, 7])
- **Global functions**: `build_convnext_tiny(num_classes, pretrained)` and `freeze_convnext_backbone(model)` — ALWAYS call both fresh per seed in Step 6.4
- **Global**: `DL_MODEL` (template instance), `GRADCAM_TARGET_LAYER = DL_MODEL.features[-1]`
- **Training**: 3 independent seeds = [42, 123, 456]. Mean ± std reported.
- **Status**: Step 6.1 COMPLETE (architecture instantiated, verified). Steps 6.2–6.5 NEXT.

---

## 7. CURRENT NOTEBOOK STATE

**File path**: `/Users/zeyadkhalil/Downloads/AIDL-individual-assignmet/strawberry_domain_shift.ipynb`

**Total cells**: 52 (50 markdown + code cells covering Phases 0–6 Step 6.1)

**Completed phases**:
- ✅ **Phase 0**: Environment setup (Steps 0.1–0.4) — dependency install, global seed, GPU verification
- ✅ **Phase 1**: Data acquisition (Steps 1.1–1.2) — Zenodo download with MD5 check, file audit
- ✅ **Phase 2**: Data pipeline (Steps 2.1–2.3) — YOLO patch extraction, class imbalance, image-level stratified split
- ✅ **Phase 3**: EDA (Steps 3.1–3.4) — visual grid, colour stats, patch size distribution, EDA narrative
- ✅ **Phase 4**: LED augmentation (Steps 4.1–4.3) — `LEDNightSimulation` class, test_night generation, visualisation
- ✅ **Phase 5**: SVM baseline (Steps 5.1–5.4) — feature extraction, SVM training, E2-Day eval, E2-Night eval
- ✅ **Phase 6 Step 6.1**: Architecture selection — ConvNeXt-Tiny instantiated and verified

**Next step**: Phase 6, **Step 6.2 — Transfer Learning Configuration** (optimizer, scheduler, loss function, detailed training config)

### Key Global Variables (all must be treated as already defined)

```python
# Paths & Config
BASE_DIR               # pathlib.Path — project root directory
DIRS                   # dict of all subdirectory paths (see below)
DEVICE                 # torch.device — cuda or cpu
GLOBAL_SEED            # 42
EXPERIMENT_SEEDS       # [42, 123, 456] — used per seed in Step 6.4
PATCH_SIZE             # 224
MIN_PATCH_PX           # 20
PATCH_PADDING          # 0.10
N_TEST_IMAGES          # 150
TRAIN_VAL_RATIO        # 0.80
RECOMMENDED_BATCH_SIZE # 64/48/32/16 based on VRAM tier (set in Step 0.4)
IS_COLAB               # bool
USE_DRIVE_CACHE        # False (default for submission)
CLASS_NAMES            # ['ripe', 'unripe']
FEATURE_DIM            # 259
metadata_path          # Path to metadata.json
METADATA               # dict — updated throughout, saved to metadata.json

# Feature matrices (from Step 5.1)
X_train, y_train       # float32 [N,259], int32 [N]
X_val, y_val           # float32 [N,259], int32 [N]
X_test_day, y_test_day     # float32 [N,259], int32 [N]
X_test_night, y_test_night # float32 [N,259], int32 [N]

# SVM (from Phase 5)
SVM_MODEL              # sklearn Pipeline(StandardScaler, SVC) — fitted
E2_DAY_METRICS         # dict — standard schema, saved to metrics/E2_day_metrics.json
E2_NIGHT_METRICS       # dict — standard schema, saved to metrics/E2_night_metrics.json

# DL (from Step 6.1)
DL_MODEL_NAME          # 'convnext_tiny'
DL_NUM_CLASSES         # 2
DL_MODEL               # nn.Module — ConvNeXt-Tiny, frozen, on DEVICE
GRADCAM_TARGET_LAYER   # DL_MODEL.features[-1]

# Functions
set_seed(seed)                        # seeds Python, numpy, torch, cuda
print_section(title)                  # prints a formatted section header
extract_features(img_rgb)             # returns float32 array [259]
build_convnext_tiny(num_classes, pretrained)  # returns fresh nn.Module
freeze_convnext_backbone(model)       # returns model with stem+stage1 frozen
_bootstrap_macro_f1(y_true, y_pred)  # returns (ci_low, ci_high) — 95% CI
```

### Key Directory Structure (DIRS dict)
```python
DIRS = {
    'base'             : BASE_DIR,
    'data'             : BASE_DIR / 'data',
    'patches'          : BASE_DIR / 'patches',
    'train_ripe'       : BASE_DIR / 'patches/train/ripe',
    'train_unripe'     : BASE_DIR / 'patches/train/unripe',
    'val_ripe'         : BASE_DIR / 'patches/val/ripe',
    'val_unripe'       : BASE_DIR / 'patches/val/unripe',
    'test_day_ripe'    : BASE_DIR / 'patches/test_day/ripe',
    'test_day_unripe'  : BASE_DIR / 'patches/test_day/unripe',
    'test_night_ripe'  : BASE_DIR / 'patches/test_night/ripe',
    'test_night_unripe': BASE_DIR / 'patches/test_night/unripe',
    'figures'          : BASE_DIR / 'figures',
    'metrics'          : BASE_DIR / 'metrics',
    'models'           : BASE_DIR / 'models',
    'model_svm'        : BASE_DIR / 'models/svm',
    'model_dl'         : BASE_DIR / 'models/dl',
    'logs'             : BASE_DIR / 'logs',
    'checkpoints'      : BASE_DIR / 'models/dl/checkpoints',
}
```

### Standard Metrics JSON Schema (used across ALL 5 experiments)
```python
METRICS_SCHEMA = {
    'experiment'     : str,           # e.g. 'E1-Day'
    'model'          : str,           # e.g. 'ConvNeXt-Tiny'
    'split'          : str,           # 'test_day' or 'test_night'
    'n_samples'      : int,
    'macro_f1'       : float,
    'macro_f1_ci_95' : [float, float],  # percentile bootstrap CI
    'auc_pr'         : float,
    'auc_roc'        : float,
    'mcc'            : float,
    'accuracy'       : float,
    'per_class'      : {
        'ripe'   : {'precision': float, 'recall': float, 'f1': float, 'support': int},
        'unripe' : {'precision': float, 'recall': float, 'f1': float, 'support': int},
    },
    'confusion_matrix' : [[int, int], [int, int]],
    # Optional keys per experiment:
    'day_night_delta'  : dict,        # E2-Night and E3
    'training_seeds'   : list,        # DL experiments
    'mean_macro_f1'    : float,       # DL experiments (mean across seeds)
    'std_macro_f1'     : float,       # DL experiments (std across seeds)
}
```

---

## 8. STRICT DEVELOPMENT PROCEDURE

**Apply to EVERY step of EVERY phase — no exceptions. This is the most critical rule.**

For each step, in this exact order:

### Step 1 — RESEARCH
Thorough, literature-grounded review of the best available approach for this specific step and project context. Must evaluate alternatives. Must be specific to our constraints (T4, 4K patches, Grad-CAM required, 3 seeds, 3-day deadline).

### Step 2 — JUSTIFY
Explain precisely why the chosen approach is selected, what alternatives were considered, and why they were rejected. Must reference specific papers or documented precedents. No vague justifications.

### Step 3 — APPROVE
Present research + justify to the project lead (user). Ask explicitly: **"Do you approve? Shall I implement Step X.Y?"** Then **WAIT** for the user's confirmation before writing any code.

### Step 4 — IMPLEMENT
Write and add the code to the notebook. All code follows these rules:
- All cells are idempotent (guarded by file-existence checks so "Run All" never re-runs completed work)
- All file paths use `DIRS` dict (never hardcoded)
- All random operations call `set_seed()` immediately before
- Variables use leading underscore (`_var`) for cell-local scope
- `print_section('Step X.Y — Title')` at the top of every code cell
- No cell executes test set more than once (test set used exactly once per experiment)
- Every figure saved at 300 DPI to `DIRS['figures']`
- Every metrics dict saved to `DIRS['metrics']` as JSON using the standard schema

### Step 5 — EXPLAIN RESULTS
Interpret all outputs, metrics, and visualisations. If the code has NOT been run on Colab yet (which is the case for all new steps), label this section clearly as **"(expected — code not yet run on Colab)"** and give realistic expected ranges based on literature and dataset characteristics.

### Step 6 — IDENTIFY IMPROVEMENTS
List 3–5 specific, actionable improvements that could be made to this step with more time or resources. Be concrete — not generic statements like "could tune more hyperparameters."

### Step 7 — ASK TO PROCEED
After completing all 6 above, say: **"Step X.Y complete. Shall I proceed to Step X.Z?"** Then **WAIT** for the user's confirmation.

### CRITICAL VIOLATIONS TO AVOID
1. **Never skip or rush the APPROVE step** — do not start implementing before the user explicitly approves
2. **Research must be thorough** — check ALL reasonable alternatives, not just the first approach that comes to mind
3. **Explain Results must be labelled** as expected/actual appropriately
4. **Always end with the "Shall I proceed?" question** — never start the next step automatically

---

## 9. NOTEBOOK CODE CONVENTIONS

### Cell structure (every code cell)
```python
# ============================================================
# STEP X.Y — TITLE IN CAPS
# ============================================================
print_section('Step X.Y — Title')

# Idempotency guard (example)
_output_path = DIRS['metrics'] / 'E1_metrics.json'
if _output_path.exists():
    print('  Already complete — loading cached result.')
    with open(_output_path) as _f:
        E1_METRICS = json.load(_f)
else:
    # ... implementation ...
    print(f'\n[OK] Step X.Y complete.')
```

### Adding cells to the notebook
The notebook is at: `/Users/zeyadkhalil/Downloads/AIDL-individual-assignmet/strawberry_domain_shift.ipynb`

To add cells, use Python to load the JSON, append to `nb['cells']`, and save:
```python
import json
with open('strawberry_domain_shift.ipynb') as f:
    nb = json.load(f)

md_cell = {"cell_type": "markdown", "metadata": {}, "source": ["### Step X.Y...\n", "..."]}
code_cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["line1\n", "line2\n"]}

nb['cells'].append(md_cell)
nb['cells'].append(code_cell)

with open('strawberry_domain_shift.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
```

**IMPORTANT**: When writing Python strings inside the JSON source list, avoid nested quotes that break the outer Python heredoc. Use escaped unicode for special characters (e.g., `\u2014` for —, `\u0394` for Δ, `\u2192` for →, `\u00d7` for ×).

### Imports already available globally (do not re-install)
```python
import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import json, time, os, shutil
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef
from sklearn.inspection import permutation_importance
from skimage.feature import local_binary_pattern, hog as skimage_hog
import albumentations as A
import joblib
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
```

### Pinned package versions (from Step 0.2)
```
albumentations>=1.3.0,<2.0.0
grad-cam>=1.4.6
scikit-image>=0.21.0
```
Everything else uses Colab pre-installed versions.

---

## 10. REMAINING WORK (what you must implement next)

### IMMEDIATE NEXT: Phase 6 — Step 6.2

**Step 6.2 — Transfer Learning Configuration**

This step defines the optimizer, learning rate scheduler, loss function, and any additional regularisation for training ConvNeXt-Tiny. This is a **configuration** step — no training yet (training is Step 6.4). The deliverable is: global constants defining all training hyperparameters, justified with literature. Expected globals to define:

```python
DL_OPTIMIZER_NAME      # 'AdamW'
DL_LR                  # learning rate (likely 1e-4 or 5e-4 for fine-tuning)
DL_WEIGHT_DECAY        # weight decay (ConvNeXt paper recommends 0.05)
DL_SCHEDULER_NAME      # 'CosineAnnealingLR'
DL_MAX_EPOCHS          # suggested 30-50
DL_EARLY_STOP_PATIENCE # suggested 10 (on val macro-F1)
DL_LABEL_SMOOTHING     # suggested 0.1 for small dataset
DL_BATCH_SIZE          # = RECOMMENDED_BATCH_SIZE
```

### Remaining phases after Step 6.2:

**Phase 6 (remaining)**:
- **Step 6.3**: Training augmentation pipeline (standard day augmentations: flip, rotate ±30°, colour jitter, random crop, ImageNet normalisation). Val/test: resize + normalise only. Albumentations-based or torchvision transforms. Visual verification on sample batch.
- **Step 6.4**: Training loop — AdamW + CosineAnnealingLR + early stopping + checkpoint saving. Run 3 seeds (EXPERIMENT_SEEDS = [42, 123, 456]). Training curves saved as CSV + plotted.
- **Step 6.5**: E1 evaluation on test_day — same metric pipeline as E2 (macro-F1 + bootstrap CI, PR-AUC, ROC-AUC, MCC, confusion matrix). Save `E1_DAY_METRICS` dict.

**Phase 7** — E3 + E4 (domain shift quantification):
- **Step 7.1**: Night test set already generated (Phase 4) — just verify patch counts match
- **Step 7.2**: Load best E1 checkpoint, evaluate on X_test_night → `E3_METRICS`
- **Step 7.3**: E4 degradation table: ΔF1 = E1−E3, per-class delta, bootstrap significance test across 3 seeds

**Phase 8** — E5 (domain adaptation):
- **Step 8.1**: DataLoader with on-the-fly LED augmentation (stochastic variant from Phase 4) applied to training patches; val remains day-only. Justify day:night ratio (1:1 default, research alternatives)
- **Step 8.2**: Retrain same ConvNeXt-Tiny architecture with same hyperparameters on adapted training set. 3 seeds.
- **Step 8.3**: Evaluate E5 on test_night → `E5_METRICS`
- **Step 8.4**: Gap closure analysis: (E4_gap − post_gap) / E4_gap × 100%; verify E5 doesn't degrade day performance

**Phase 9** — Grad-CAM XAI:
- **Step 9.1**: Setup pytorch-grad-cam with `GRADCAM_TARGET_LAYER = DL_MODEL.features[-1]`
- **Step 9.2**: Grad-CAM on 12 ripe + 12 unripe day patches using E1 model
- **Step 9.3**: Same 24 patches run through night simulation → Grad-CAM with E1 model (show confusion)
- **Step 9.4**: Same night patches → Grad-CAM with E5 model (show recovery)
- **Step 9.5**: 6×3 publication-quality comparison grid (day/E1 | night/E1 | night/E5)

**Phase 10** — Results consolidation:
- **Step 10.1**: Master results table as styled pandas DataFrame
- **Step 10.2**: 4-panel summary figure (training composition, F1 comparison, confusion matrices, gap closure)
- **Step 10.3**: Notebook cleanup — remove debug cells, verify top-to-bottom execution
- **Step 10.4**: README cell (first cell) — title, problem, dataset, how to run, hardware req, runtime estimate
- **Step 10.5**: Reproducibility audit — all seeds set, no hardcoded paths, all models saved with metadata

**Phase 11** — Critical Reflection Report:
- **Step 11.1** (~650 words): ML vs DL comparison using E1 vs E2 results
- **Step 11.2** (~750 words): Design decisions + AI tool usage disclosure
- **Step 11.3** (~550 words): Impact on individuals/orgs/environment/society
- **Step 11.4** (~550 words): Future trends — Test-Time Adaptation (Wang et al., 2021 TENT), Vision Foundation Models (DINOv2, Oquab et al. 2024), multimodal NIR sensing
- **Step 11.5**: References (academic + commercial, Harvard or IEEE format)
- **Step 11.6**: Word count audit ≤ 2,800 words

---

## 11. PROJECT STATUS ASSESSMENT

### What is working well
- The notebook architecture is enterprise-grade and methodologically sound
- All data pipeline work (Phases 0–4) is complete and idempotent
- The SVM baseline (Phase 5) is fully implemented with rigorous metrics — bootstrap CI, PR-AUC, permutation importance, day/night comparison
- The LED domain shift simulation is physically motivated and implemented cleanly
- Data leakage prevention has been rigorously enforced (image-level split + scaler inside Pipeline)
- Architecture decision (ConvNeXt-Tiny) is well-justified and documented

### What is at risk
- **Time**: 2 days remain, Phases 6–11 are not yet implemented. Phases 6–10 are the most code-heavy.
- **Training time**: 3 seeds × E1 + 3 seeds × E5 = 6 training runs on Colab T4 (~12–18 min each = ~90–108 min total). This is manageable but requires careful session management.
- **Colab session limits**: Checkpoints must be saved to Drive after each seed (Step 6.4 already plans this). If session disconnects, resume from checkpoint.
- **Phase 11 (report)**: Requires actual metric numbers from experiments to fill in. Must be written after Phase 10 is complete.

### Priority order for remaining time
1. ✅ Phase 6.2–6.5 (ConvNeXt-Tiny training + E1 evaluation) — **highest priority**
2. Phase 7 (E3 + E4) — **second priority** — reuses E1 checkpoint, fast to implement
3. Phase 8 (E5 domain adaptation) — **third priority** — requires full retraining
4. Phase 9 (Grad-CAM) — **fourth priority** — visually impressive, important for LO2
5. Phase 10 (results consolidation) — can be fast once metrics are collected
6. Phase 11 (report) — can be drafted in parallel with late phases

---

## 12. COLAB T4 OPERATIONAL NOTES

- **Hardware**: NVIDIA T4, 16 GB VRAM (~14 GB usable), 2 vCPU, ~12.7 GB RAM
- **Always verify GPU** at session start: `torch.cuda.is_available()` + `nvidia-smi`
- **ConvNeXt-Tiny VRAM**: ~2.8 GB at bs=64 — very safe, never need to reduce batch
- **3 seeds total training time** (E1): ~36–54 min on T4
- **Session timeout**: Colab disconnects after ~12h idle. Save checkpoints to Drive after each seed.
- **Zenodo dataset**: Already downloaded to Drive (via Step 1.1 idempotency guard). Do NOT re-download.
- **Idempotency guards**: Every cell checks if output files exist before running. "Run All" is always safe.
- **RECOMMENDED_BATCH_SIZE**: Set in Step 0.4 based on VRAM. Default 64 for T4 with ConvNeXt-Tiny.
- **torch.cuda.empty_cache()**: Call between experiments to free VRAM between E1 and E5 training.

---

## 13. KEY LITERATURE REFERENCES (for report + justifications)

### Architecture & Transfer Learning
- **Liu et al. (2022)** — *A ConvNet for the 2020s*, CVPR 2022 — ConvNeXt paper
- **He et al. (2016)** — *Deep Residual Learning*, CVPR 2016 — ResNet paper
- **Zeiler & Fergus (2014)** — *Visualising CNN features* — freezing justification
- **Yosinski et al. (2014)** — *How transferable are features in DNNs?* — layer transferability
- **Kornblith et al. (2019)** — *Do Better ImageNet Models Transfer Better?* — why ImageNet acc ≠ transfer acc

### Evaluation Metrics
- **Chicco & Jurman (2020)** — *The advantages of MCC over F1 and accuracy* — MCC justification
- **Davis & Goadrich (2006)** — *Relationship between PR and ROC curves* — PR-AUC justification

### Domain Shift & Adaptation
- **Ben-David et al. (2010)** — *A theory of learning from different domains* — H-divergence
- **Torralba & Efros (2011)** — *Unbiased look at dataset bias* — covariate shift
- **Wang et al. (2021)** — *TENT: Fully Test-time Adaptation* — future trend for Phase 11
- **Oquab et al. (2024)** — *DINOv2: Learning robust features* — vision foundation models

### XAI
- **Selvaraju et al. (2017)** — *Grad-CAM: Visual explanations from CNNs* — the Grad-CAM paper

### Agricultural Domain
- **Dogtooth Technologies** — Gen5 robot public blog (commercial reference)
- **Journal of Field Robotics (2024)** — documented LED domain shift
- **Pastell & Liakka (2022)** — Dataset 1 paper (Zenodo DOI: 10.5281/zenodo.6126677)
- **Doğanlar et al. (2021)** — ResNet-18 for strawberry classification on small datasets

---

## 14. WORD COUNT CONSTRAINT FOR REPORT

- **Maximum**: 2,800 words (+10% tolerance = 3,080 hard limit)
- **Includes**: in-text citations, table content, figure captions
- **Excludes**: reference list, appendix material
- **Target per section**: S1 ~650w | S2 ~750w | S3 ~550w | S4 ~550w = 2,500w total (leaves 300w buffer)

---

## 15. HOW TO CONTINUE FROM HERE

You are at the start of **Step 6.2 — Transfer Learning Configuration**.

The development procedure requires you to:
1. **RESEARCH** — what are the optimal optimizer, LR, scheduler, regularisation settings for fine-tuning ConvNeXt-Tiny on ~4,000 agricultural patches on Colab T4?
2. **JUSTIFY** — why these specific values?
3. **APPROVE** — present to the user and wait for confirmation
4. Then implement, explain, identify improvements, and ask to proceed to Step 6.3

The project plan says this step delivers: a model summary with trainable vs frozen parameter counts (already done in Step 6.1) AND all training hyperparameter globals defined and justified. Focus the implementation on defining and saving these globals as code, not on training (which is Step 6.4).

The user has been the project lead throughout. They review and approve every step before implementation. They are knowledgeable, will challenge weak justifications, and will call out procedure violations. Always follow the 7-step procedure exactly.

Good luck. The goal is a submission-ready notebook + report by 27 April 2026 12:00.
