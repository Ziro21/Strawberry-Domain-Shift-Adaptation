# Enterprise-Grade Project Plan
# Strawberry Ripeness Classification Under LED-Induced Domain Shift
# WMG9B7-15 — Artificial Intelligence & Deep Learning

---

## Project Context

**Problem**: Dogtooth Technologies Gen5 robots are trained on daytime strawberry images but operate 24h under onboard LED lighting. The LED spectral distribution causes domain shift, degrading ripeness classification accuracy at night.

**Approach**: Patch classification (not object detection) on Dataset 1 (Pastell & Liakka, Luke Finland, 813 images, YOLO annotations, CC BY 4.0). Five controlled experiments quantify the shift and test domain adaptation.

**Target Hardware**: Google Colab T4 (16 GB VRAM, ~12.7 GB RAM, 15 GB disk — Drive mounted for persistence)

**Assessment**: WMG9B7-15 Individual Assignment, 70% weight, LO1 + LO2 + LO3

---

## Development Procedure (applied to every step of every phase)

For each step, in this exact order:

1. **RESEARCH** — Thorough review of best available approach for this specific step and project context
2. **JUSTIFY** — Explain why this approach is chosen, what alternatives exist, and why they were rejected
3. **APPROVE** — Present to project lead for approval before any code is written
4. **IMPLEMENT** — Write and run the code
5. **EXPLAIN RESULTS** — Interpret outputs, metrics, visualisations
6. **IDENTIFY IMPROVEMENTS** — Flag what could be done better with more time or resources

---

## Colab T4 Operational Notes (apply throughout all phases)

- Mount Google Drive at session start — all outputs, checkpoints, and datasets saved there
- Verify GPU availability with `torch.cuda.is_available()` and `nvidia-smi` at session start
- Fixed seeds: Python `random`, `numpy`, `torch`, `torch.cuda` — set before any data operation
- T4 VRAM budget: ~14 GB usable. ResNet-18 + batch 64 patches ≈ 2 GB — safe
- Clear GPU cache between experiments: `torch.cuda.empty_cache()`
- Save model checkpoints to Drive after every training run — Colab sessions disconnect
- Estimated training time per DL experiment on T4: 10–25 minutes depending on epochs
- Dataset download: Zenodo direct wget (~1.42 GB) — save to Drive, not /tmp

---

## PHASE 0 — Environment & Infrastructure Setup

**Goal**: Reproducible, self-contained Colab notebook environment with all dependencies pinned, GPU confirmed, and project directory structure established in Drive.

### Step 0.1 — Google Drive Mounting & Directory Architecture
Establish persistent storage structure in Drive for the entire project lifetime.

**Deliverable**: Drive directory tree confirmed, paths defined as constants used throughout.

### Step 0.2 — Dependency Installation & Version Pinning
Install and pin all required libraries. Key packages: torch, torchvision, albumentations, scikit-learn, scikit-image, pytorch-grad-cam, matplotlib, seaborn, pandas, tqdm, Pillow, opencv-python-headless.

**Deliverable**: `requirements.txt` saved to Drive. All imports verified without errors.

### Step 0.3 — Global Reproducibility Setup
Fix all random seeds across Python, NumPy, PyTorch CPU and CUDA. Set `torch.backends.cudnn.deterministic = True`. Define a `set_seed(seed=42)` utility called once at notebook start and once before each experiment.

**Deliverable**: Seed function verified by running same random operation twice and confirming identical output.

### Step 0.4 — GPU Verification & Hardware Logging
Confirm T4 availability, log VRAM capacity, CUDA version, PyTorch version, and driver version. Store in a metadata dict written to Drive for reproducibility audit.

**Deliverable**: Hardware metadata JSON saved to Drive.

---

## PHASE 1 — Data Acquisition & Integrity Validation

**Goal**: Download Dataset 1 from Zenodo, verify completeness, understand the raw annotation structure before touching any data.

### Step 1.1 — Dataset Download from Zenodo
Programmatic download of Dataset 1 (DOI: 10.5281/zenodo.6126677) directly within the notebook. No manual steps. Save to Drive for session persistence.

**Deliverable**: Archive downloaded and extracted. File count verified.

### Step 1.2 — File Structure Audit
Enumerate all image files and corresponding YOLO `.txt` annotation files. Verify 1-to-1 correspondence (every image has an annotation file, every annotation file has an image). Flag any orphaned files. Log supported image formats (jpg, png).

**Deliverable**: DataFrame of all image-annotation pairs, flagged exceptions reported.

### Step 1.3 — YOLO Annotation Format Validation
Parse a sample of annotation files to confirm format: `class_id x_center y_center width height` (normalised 0–1). Confirm class IDs match expected schema (0=ripe, 1=unripe, 2=peduncle). Detect and log any malformed lines (wrong column count, values outside [0,1], negative values).

**Deliverable**: Validation report — total annotations, malformed count, class ID distribution.

### Step 1.4 — Raw Annotation Statistics
Count total bounding box annotations across all images. Count per class. Compute annotations-per-image distribution (min, max, mean, median, std). This determines effective patch dataset size before any processing.

**Deliverable**: Annotation statistics table. Histogram of annotations-per-image.

---

## PHASE 2 — Data Pipeline & Patch Extraction

**Goal**: Convert the full-image + YOLO-annotation dataset into a clean, labelled patch dataset of individual berry crops, with proper splits.

### Step 2.1 — Bounding Box Parser & Patch Cropper
Build a robust function that reads each image and its annotation file, converts normalised YOLO coordinates to absolute pixel coordinates, and crops each bounding box with a configurable padding margin (e.g. 10% of box dimension on each side to include context). Filter out patches below a minimum size threshold (e.g. 20×20 pixels — too small to be informative). Save each patch as an individual file with a structured filename encoding: `{source_image_id}_{annotation_index}_{class_id}.jpg`.

**Deliverable**: Full patch dataset saved to Drive. Count per class confirmed.

### Step 2.2 — Patch Size Analysis & Standardisation Strategy
Analyse the distribution of extracted patch sizes (height, width) across all classes. Determine the optimal fixed resize target for the classifier. Consider: too small loses texture detail, too large wastes compute. ResNet-18 input is 224×224 — determine whether to resize all patches to 224×224 directly or use a smaller intermediate (e.g. 128×128 with upscaling in the loader).

**Deliverable**: Patch size distribution plots per class. Justified resize target chosen.

### Step 2.3 — Stratified Train / Validation / Test Split
This is the most critical data operation in the project. Implement a stratified split at the **image level** (not patch level) to prevent data leakage — patches from the same source image must not appear in both train and test.

Split strategy:
- First reserve 150 source images as held-out test (stratified by per-image dominant class)
- From remaining ~663 images: 80% train / 20% validation (stratified)
- All patches from a given source image go to exactly one split

Save split assignments as a CSV to Drive. This CSV is the ground truth for all experiments — never regenerated after this point.

**Deliverable**: Split assignment CSV. Patch counts per split per class. Verification that no source image appears in two splits.

### Step 2.4 — Class Imbalance Analysis & Handling Strategy
Count ripe vs unripe patches in the training split. Compute imbalance ratio. Decide on handling strategy (class weights in loss function vs oversampling via augmentation vs neither if ratio < 2:1). Note: peduncle class (class 2) is excluded from binary classification — document this scoping decision explicitly.

**Deliverable**: Class distribution bar charts per split. Chosen imbalance handling strategy justified.

---

## PHASE 3 — Exploratory Data Analysis

**Goal**: Deeply understand the visual properties of the patch dataset before any modelling. This section directly feeds the report's justification for why the domain shift affects performance.

### Step 3.1 — Visual Sample Grid
Display a grid of randomly sampled patches per class (ripe / unripe) from the training split. Show at least 5×5 per class. Visually inspect variation in size, lighting, background, berry orientation, and ripeness appearance.

**Deliverable**: 2 sample grids (ripe / unripe), saved as figures for potential report inclusion.

### Step 3.2 — Colour Distribution Analysis (Day Baseline)
For each class, compute mean pixel values and standard deviations in RGB, HSV, and LAB colour spaces across all training patches. Plot:
- Mean colour distribution histograms per channel per class
- HSV channel distributions per class (H=hue especially relevant — ripe berries cluster at low hue/red, unripe at high hue/green)

This analysis motivates the night augmentation design: the domain shift specifically attacks the colour channels that separate ripe from unripe.

**Deliverable**: Colour distribution plots per class per colour space. Key finding: which channel most discriminates ripe from unripe.

### Step 3.3 — Texture Analysis
Compute Local Binary Pattern (LBP) histograms per class. Visualise mean LBP response per class. This determines whether texture alone (independent of colour) could separate classes — important for predicting Grad-CAM behaviour under domain shift.

**Deliverable**: LBP histograms per class. Qualitative assessment of texture discriminability.

### Step 3.4 — EDA Summary & Hypotheses
Summarise EDA findings as explicit testable hypotheses:
1. H1: Colour (hue) is the primary discriminating feature between ripe and unripe
2. H2: Texture (LBP) is a secondary discriminator, robust to lighting change
3. H3: LED augmentation will specifically degrade colour-based classification
4. H4: A colour-sensitive baseline (SVM on HSV) will degrade more than a DL model under domain shift
These hypotheses are tested by experiments E1–E5 and confirmed by Grad-CAM.

**Deliverable**: Written hypothesis table. Used directly in the report's design decisions section.

---

## PHASE 4 — LED Night Augmentation Pipeline

**Goal**: Build a physically motivated, reproducible simulation of the domain shift from solar irradiance to onboard LED illumination. This is the most scientifically novel component of the project.

### Step 4.1 — Physical Motivation & Transform Design
Research and document the spectral physics of LED illumination vs solar irradiance:
- Solar: broad continuous spectrum across 400–700nm
- Typical agricultural LED arrays: narrow peaks at ~450nm (blue) and ~630nm (red), suppressed green
- Effect on berry appearance: relative R/G/B balance shifts, warm red tones reduced, specular highlights appear on berry surface under directional LED

Design the transformation chain:
1. RGB → LAB (perceptually uniform colour space — enables targeted colour manipulation)
2. Reduce L (luminance) — night is darker
3. Shift A channel toward negative (reduce red-green warmth)
4. Shift B channel toward positive (increase blue-white component)
5. RGB → add Gaussian sensor noise (higher ISO equivalent at night)
6. Add synthetic specular highlight (Gaussian blob on random berry surface point)
7. Optionally: reduce global saturation

**Deliverable**: Physics justification written (2-3 paragraphs for report). Transform chain designed with parameter ranges.

### Step 4.2 — Night Augmentation Implementation
Implement the LED simulation as a deterministic, seeded, composable Albumentations-compatible transform class. Parameters must be configurable and bounded. Implement in pure NumPy + OpenCV for speed. Include a `validate_output()` check to ensure no out-of-range pixel values.

**Deliverable**: `LEDNightSimulation` class, tested on 10 sample patches.

### Step 4.3 — Visual Validation of Augmented Patches
Generate a visual comparison grid: for 10 ripe and 10 unripe patches, show original (day) alongside augmented (night-simulated). Visually confirm:
- Patches look plausibly darker and cooler
- Ripe berries lose their warm red appearance under simulation
- Texture is preserved (shape/edges still visible)

**Deliverable**: Day vs Night comparison grid saved as figure. Qualitative validation passed.

### Step 4.4 — Augmentation Strength Calibration
Vary augmentation strength parameters across a grid and compare colour distributions of augmented patches against published reference data for LED-lit agricultural imagery (if available). Select parameter values that maximise visual plausibility while preserving patch recognisability. Document chosen parameters with rationale.

**Deliverable**: Parameter configuration frozen and saved to Drive as a config dict. Used identically in all experiments.

---

## PHASE 5 — Experiment E2: Traditional ML Baseline (SVM)

**Goal**: Establish the traditional ML performance ceiling using hand-crafted features + SVM. This is the primary LO1 comparison point. Must be methodologically rigorous.

### Step 5.1 — Feature Extraction Pipeline Design
Design the hand-crafted feature vector:
- HSV histogram: 32 bins per channel (H, S, V) = 96 features — captures colour distribution
- Local Binary Pattern (LBP): radius=2, n_points=16, uniform method = 18 features — captures texture
- HOG (Histogram of Oriented Gradients): 3×3 cells, 9 bins = 81 features — captures shape/edges
- Total: ~195 features per patch

Resize all patches to identical size before feature extraction. Normalise features (StandardScaler fitted on training split only — scaler saved to Drive and applied to val/test without refitting).

**Deliverable**: Feature extraction function. Feature matrix shape confirmed. Scaler fitted and saved.

### Step 5.2 — SVM Architecture & Hyperparameter Tuning
Train SVM with RBF kernel. Tune C and gamma via 5-fold cross-validation on the training split using GridSearchCV. Search space: C ∈ {0.1, 1, 10, 100}, gamma ∈ {scale, auto, 0.001, 0.01}. Use `class_weight='balanced'` if class imbalance exists.

**Deliverable**: CV results grid. Best hyperparameters logged. Training time recorded.

### Step 5.3 — SVM Evaluation on Day Test Set (E2-Day)
Evaluate the tuned SVM on the held-out day test patches. Report:
- Per-class precision, recall, F1
- Macro-averaged F1
- ROC-AUC (one-vs-rest)
- Confusion matrix

**Deliverable**: Full evaluation report. Confusion matrix heatmap saved as figure.

### Step 5.4 — SVM Evaluation on Night Test Set (E2-Night)
Apply LED night augmentation to the held-out test patches (using the frozen augmentation config from Phase 4). Evaluate the same trained SVM on night-augmented patches — no retraining.

Report the same metrics as E2-Day. Compute ΔF1 (E2-Night vs E2-Day).

**Deliverable**: Night evaluation report. Direct comparison table: E2-Day vs E2-Night. This is a secondary but valuable data point confirming H4 (SVM is more fragile than DL under domain shift).

---

## PHASE 6 — Experiment E1: Deep Learning Baseline (Day)

**Goal**: Train the primary DL classifier on daytime patches and establish the DL performance ceiling under in-distribution conditions. All DL hyperparameter decisions are finalised here and frozen for E3 and E5.

### Step 6.1 — Model Architecture Selection
Research and select the optimal pretrained CNN backbone for fine-tuning on ~4,000 agricultural patch images on T4. Candidates: ResNet-18, ResNet-50, EfficientNet-B0, EfficientNet-B2, MobileNetV3-Large. Evaluate on: parameter count, T4 training speed, classification accuracy on similar agricultural tasks, ease of Grad-CAM integration.

**Deliverable**: Architecture decision with full justification. Alternative architectures documented.

### Step 6.2 — Transfer Learning Configuration
Configure the chosen model for fine-tuning:
- Load ImageNet pretrained weights
- Replace final classification layer with a new head (Linear → output 2 classes: ripe/unripe)
- Define layer freezing strategy: freeze early layers (low-level features), unfreeze from layer3 onwards
- Justify freezing strategy: early conv layers detect edges/colours universally, deeper layers are task-specific

**Deliverable**: Model summary with trainable vs frozen parameter counts.

### Step 6.3 — Training Augmentation Pipeline (Standard)
Define the training-time data augmentation applied to day patches during E1 training. This is separate from the LED simulation — these are standard augmentations for generalisation: random horizontal/vertical flip, random rotation (±30°), random colour jitter (moderate), random crop, normalisation (ImageNet mean/std). Validation and test patches receive only resize + normalise.

**Deliverable**: Training vs validation transform pipelines defined. Visual verification on sample batches.

### Step 6.4 — Training Loop Implementation
Implement full training loop:
- Loss: CrossEntropyLoss (with class weights if imbalance)
- Optimiser: AdamW with weight decay
- Scheduler: CosineAnnealingLR
- Early stopping: monitor val F1, patience=10 epochs
- Checkpoint: save best model (by val F1) to Drive every epoch it improves
- Logging: train loss, val loss, val F1, val accuracy per epoch — saved as CSV to Drive

Run 3 independent training runs with different seeds. Report mean ± std of val F1.

**Deliverable**: Training curves (loss + F1 vs epoch) for all 3 seeds. Best checkpoint saved.

### Step 6.5 — E1 Evaluation on Day Test Set
Load best checkpoint. Evaluate on held-out day test patches. Report:
- Per-class precision, recall, F1
- Macro-F1
- ROC-AUC
- Confusion matrix
- Bootstrap 95% confidence interval on macro-F1 (1000 bootstrap samples)

**Deliverable**: Full evaluation report. All figures saved. E1 result locked as the day performance ceiling.

---

## PHASE 7 — Experiments E3 & E4: Domain Shift Quantification

**Goal**: Demonstrate and quantify how much performance degrades when the E1 model (trained on day) is exposed to LED night conditions. This is the core scientific finding — the "unsolved problem" proof.

### Step 7.1 — Night Test Set Generation
Apply the frozen LED simulation pipeline to the held-out 150 test images and extract night-augmented patches. This creates a parallel night version of the test set with identical image content but simulated LED conditions. Save to Drive. Verify patch count matches day test set exactly.

**Deliverable**: Night test patch dataset saved. Day vs Night side-by-side sample confirmed identical berry content.

### Step 7.2 — Experiment E3: E1 Model on Night Test Set
Load the E1 best checkpoint (trained on day only). Evaluate directly on the night test patches — no retraining, no adaptation.

Report the same full metric set as E1. Apply bootstrap CI on macro-F1.

**Deliverable**: E3 evaluation report. Confusion matrix heatmap.

### Step 7.3 — Experiment E4: Degradation Quantification
Compute the performance gap between E1 (day) and E3 (night):
- ΔMacro-F1 = F1(E1) − F1(E3)
- ΔAccuracy = Acc(E1) − Acc(E3)
- Per-class F1 delta (ripe and unripe separately — which class degrades more?)
- Statistical significance: paired bootstrap test on F1 scores across 3 seeds

Visualise: side-by-side confusion matrices (day vs night), bar chart of F1 per class (day vs night).

**Deliverable**: E4 degradation table. Key finding: exact quantification of the unsolved problem (e.g. "macro-F1 drops from 0.91 to 0.67 under LED simulation — a 24 percentage point degradation"). This is the centrepiece result of the paper.

---

## PHASE 8 — Experiment E5: Domain Adaptation via Augmented Training

**Goal**: Test whether training the model on a mixture of day and LED-simulated patches partially closes the degradation gap from E4. This is the proposed solution component.

### Step 8.1 — Adapted Training Set Construction
Create the domain-adapted training set by combining:
- Original day training patches
- LED-simulated versions of the same training patches (night augmentation applied on-the-fly during training via the DataLoader)

Do NOT apply night augmentation to validation patches — validation remains day-only (to separate generalisation from domain shift effects cleanly).

Document the day:night ratio. Research and justify whether 1:1 or a skewed ratio (e.g. 2:1 day:night) performs better for this type of augmentation-based adaptation.

**Deliverable**: Adapted DataLoader configured. Day:night ratio justified.

### Step 8.2 — E5 Model Training (Domain Adapted)
Retrain the same architecture from Step 6.1 (same hyperparameters, same seeds) using the adapted training set. This isolates the effect of domain adaptation — the only difference from E1 is the training data distribution.

Run 3 seeds. Save best checkpoint per seed to Drive.

**Deliverable**: Training curves for E5 (compare shape vs E1 — does it converge differently?). Best checkpoint saved.

### Step 8.3 — E5 Evaluation on Night Test Set
Evaluate E5 best checkpoint on the night test patches (same set used in E3).

Report full metrics + bootstrap CI on macro-F1.

**Deliverable**: E5 evaluation report.

### Step 8.4 — Recovery Analysis
Compute the gap closure:
- Pre-adaptation gap (E4): F1(E1_day) − F1(E3_night)
- Post-adaptation gap: F1(E1_day) − F1(E5_night)
- Gap closure rate = (E4_gap − Post_gap) / E4_gap × 100%

Also verify E5 does not degrade day performance: evaluate E5 model on day test set and compare to E1.

**Deliverable**: 3-column comparison table: E1-Day / E3-Night (no adapt) / E5-Night (adapted). Gap closure percentage. This is the proof of concept for the domain adaptation approach.

---

## PHASE 9 — XAI: Grad-CAM Analysis

**Goal**: Visually explain what features the model relies on under day vs night conditions, confirming the hypotheses from Phase 3 EDA. This is the interpretability component required for LO2 Distinction.

### Step 9.1 — Grad-CAM Setup & Layer Selection
Install and configure pytorch-grad-cam. Research and select the optimal target layer for the chosen backbone. For ResNet-18: `layer4[-1]` (last residual block). For EfficientNet: `features[-1][0]`. Verify output heatmap size is appropriate for 224×224 input.

**Deliverable**: GradCAM object initialised. Test heatmap on one sample confirmed.

### Step 9.2 — Grad-CAM on Day Patches (E1 Model)
Select 12 correctly classified ripe patches and 12 correctly classified unripe patches from the day test set. Generate Grad-CAM overlays using the E1 (day-trained) model. Visualise as heatmap overlaid on original patch.

Expected finding: model attends to berry surface — distributed across both colour and texture regions.

**Deliverable**: 24-patch Grad-CAM grid for day inputs (E1 model). Interpretation written.

### Step 9.3 — Grad-CAM on Night Patches (E1 Model — Confused State)
Apply Grad-CAM using the same E1 model but on the night-augmented versions of the same 24 patches. Crucially: include examples where E1 misclassifies (a ripe berry classified as unripe under night conditions).

Expected finding: attention becomes diffuse or migrates to non-berry regions — model is confused because its colour cues are disrupted by LED shift.

**Deliverable**: 24-patch Grad-CAM grid for night inputs (E1 model). Direct visual comparison with Step 9.2.

### Step 9.4 — Grad-CAM on Night Patches (E5 Model — Adapted State)
Apply Grad-CAM using the E5 (domain-adapted) model on the same night-augmented patches. Compare attention maps to Step 9.3.

Expected finding: attention returns to berry surface shape/texture — the adapted model has learned features beyond colour, confirming H2 (texture is a more robust discriminator under domain shift).

**Deliverable**: 24-patch Grad-CAM grid for night inputs (E5 model).

### Step 9.5 — Three-Way Comparison Grid
Compose a final 3-column figure for each of 6 representative patches (3 ripe, 3 unripe):
- Column 1: E1 model on day patch
- Column 2: E1 model on night patch (pre-adaptation)
- Column 3: E5 model on night patch (post-adaptation)

This single figure is the most powerful visual in the entire notebook and should be publication-quality.

**Deliverable**: 6×3 Grad-CAM comparison figure. Caption written. Figure saved at 300 DPI.

---

## PHASE 10 — Results Consolidation & Notebook Finalisation

**Goal**: Produce a clean, reproducible, fully documented notebook ready for submission.

### Step 10.1 — Master Results Table
Compile all experiment results into a single consolidated DataFrame:

| Experiment | Training Data | Test Data | Macro-F1 (mean±std) | Ripe-F1 | Unripe-F1 | ROC-AUC | ΔF1 vs E1 |
|---|---|---|---|---|---|---|---|
| E1: DL-Day | Day | Day | | | | | — |
| E2: SVM-Day | Day (features) | Day | | | | | — |
| E2-N: SVM-Night | Day (features) | Night | | | | | |
| E3: DL-Day→Night | Day | Night | | | | | |
| E4: Degradation | — | — | — | — | — | — | E1−E3 |
| E5: DL-Adapted | Day+Night | Night | | | | | |

**Deliverable**: Master results table rendered as styled DataFrame in notebook.

### Step 10.2 — Publication-Quality Summary Figure
Compose a 4-panel summary figure:
- Panel A: Training data composition (pie charts)
- Panel B: F1 comparison bar chart across all experiments
- Panel C: Confusion matrices side-by-side (E1-Day / E3-Night / E5-Night)
- Panel D: Gap closure visualisation

**Deliverable**: 4-panel figure saved at 300 DPI.

### Step 10.3 — Notebook Cleanup & Documentation
- Remove all debug cells
- Add markdown explanatory cells before each phase and each major code block
- Ensure every figure has a numbered caption
- Ensure every variable is defined before use (notebook runs top-to-bottom without errors)
- Verify reproducibility: restart kernel, run all cells, confirm identical results

**Deliverable**: Clean notebook runs end-to-end from fresh kernel.

### Step 10.4 — README Cell
Write the required README cell (first cell of notebook, markdown) covering:
- Project title and one-sentence problem statement
- Dataset source + license + download instructions
- How to run (Runtime → Run All after Drive mount)
- Expected outputs per section
- Hardware requirement: GPU runtime (T4 or better)
- Estimated total runtime on T4
- Python and key library versions

**Deliverable**: README cell written and verified.

### Step 10.5 — Requirements & Reproducibility Audit
Final check:
- All random seeds set before every experiment
- All file paths use Drive-relative paths (no hardcoded /tmp)
- All models saved with metadata (architecture, seed, epoch, val F1)
- Split CSV is the single source of truth for all data splits

**Deliverable**: Reproducibility checklist passed. `requirements.txt` final version saved.

---

## PHASE 11 — Critical Reflection Report (Part 2)

**Goal**: Write a 2,800-word maximum critical reflection that directly addresses all four required areas and is calibrated to the Distinction/Outstanding rubric descriptors.

### Step 11.1 — Section 1: ML vs DL Comparison (target ~650 words)
Structure:
- Introduce the classification problem and both approaches (SVM + ResNet)
- Technical comparison: feature engineering vs representation learning, linear vs non-linear decision boundaries, scalability
- Problem-specific justification: why DL for this task (colour + texture + spatial relationships learned jointly; SVM HSV features collapse under spectral shift — proved by E2-Night results)
- Reference course material (LO1) + academic sources (LeCun et al., Krizhevsky et al., module slides)

**Deliverable**: Section 1 draft (~650 words). LO1 rubric criteria checked.

### Step 11.2 — Section 2: Design Decisions & AI Tool Usage (target ~750 words)
Cover:
- Dataset choice and scoping rationale (D1 only, patch classification, binary classification)
- Architecture selection (chosen backbone vs alternatives)
- Night augmentation design (physical motivation, LAB transform, parameter calibration)
- Experiment design (5-experiment structure, held-out test set, multiple seeds)
- Transfer learning strategy (layer freezing rationale)
- AI assistant usage: what was used, what was generated, how it was extended and critically evaluated
- References: Albumentations paper, pytorch-grad-cam, ImageNet transfer learning papers

**Deliverable**: Section 2 draft (~750 words). LO2 design decisions rubric checked.

### Step 11.3 — Section 3: Impact Analysis (target ~550 words)
Four impact dimensions:
- *Individuals*: Farm workers (seasonal labour displacement, UK post-Brexit agricultural workforce), robot operators, consumers (food quality)
- *Organisations*: Dogtooth Technologies (revenue from 24h uptime), competing agritech firms, strawberry growers (yield optimisation)
- *Environment*: LED energy consumption for 24h operation vs reduced food waste from optimal ripeness detection; compute cost of training (carbon footprint)
- *Society*: UK food security, precision agriculture as a sustainable farming enabler, ethical AI in autonomous systems (failure modes = real economic harm)
- Reference: existing literature on automation and agricultural labour (Sparrow & Howard, 2021 or equivalent)

**Deliverable**: Section 3 draft (~550 words). LO2 implications rubric checked.

### Step 11.4 — Section 4: Future Trends & Opportunities (target ~550 words)
Cover at minimum two concrete, recent trends with direct relevance:

**Trend 1: Test-Time Adaptation (TTA)**
Wang et al. (2021) TENT — adapt model at inference time on unlabelled target domain data. Application: Dogtooth robot continuously adapts to its specific LED hardware without relabelling. This directly solves the limitation of our augmentation-based approach (which requires predicting the shift in advance).

**Trend 2: Vision Foundation Models (ViT, DINOv2)**
Self-supervised features trained on internet-scale data encode semantic shape rather than colour statistics → inherently more robust to domain shift. Application: replace ResNet backbone with DINOv2 frozen features + lightweight head. Eliminates the colour-dependence Grad-CAM revealed.

**Optional Trend 3: Multimodal sensing (RGB + NIR)**
LED arrays emit strongly in near-infrared. NIR reflectance is well-correlated with fruit sugar content (ripeness). A dual-spectrum sensor would make the classification intrinsically LED-compatible.

Reference: Published papers for each trend (Wang et al. ICLR 2021, Oquab et al. 2024, precision agriculture NIR literature).

**Deliverable**: Section 4 draft (~550 words). LO3 rubric checked.

### Step 11.5 — References Compilation
Compile all references in a consistent academic format (Harvard or IEEE). Must include:
- Academic: LeCun et al. (1998 / 2015), Vaswani et al. (2017), Wang et al. (2021), Oquab et al. (2024), Pastell & Liakka (2022), Journal of Field Robotics (2024), Selvaraju et al. (2017 — Grad-CAM paper)
- Commercial: Dogtooth Technologies blog (Gen5, 24h operation), Ultralytics YOLO documentation

**Deliverable**: Reference list (excluded from word count). DOIs and access dates recorded.

### Step 11.6 — Word Count Audit & Final Editing
Count words (including in-text citations, tables, figures, footnotes). Ensure ≤ 2,800 words. Cut ruthlessly: prioritise precision over volume. Confirm excluded items: reference list, appendix material.

**Deliverable**: Final report at or under 2,800 words. Proofread. Submitted alongside notebook.

---

## Consolidated Experiment Summary

| ID | Name | Train | Eval | Purpose | Primary Metric |
|---|---|---|---|---|---|
| E1 | DL-Day Baseline | Day patches | Day test | DL ceiling | Macro-F1, ROC-AUC |
| E2 | SVM-Day Baseline | Day features | Day test | ML ceiling (LO1) | Macro-F1, ROC-AUC |
| E2-N | SVM-Night | Day features | Night test | ML fragility under shift | ΔF1 vs E2 |
| E3 | DL-Day → Night | Day patches | Night test | Prove domain shift breaks DL | ΔF1 vs E1 |
| E4 | Degradation | — | — | Quantify the gap | E1−E3 metrics |
| E5 | DL-Adapted | Day+Night patches | Night test | Partial gap closure | ΔF1 vs E3 |

---

## Assessment LO Coverage Map

| Phase | Steps | LO Covered |
|---|---|---|
| Phase 5 (SVM) vs Phase 6 (DL) | E2 vs E1 results | LO1 — ML vs DL comparison |
| Phase 6 (architecture), Phase 4 (augmentation), Phase 9 (Grad-CAM) | E1 eval + XAI | LO2 — implementation evaluation + interpretability |
| Phase 7+8 (domain shift quantification + adaptation) | E3+E4+E5 | LO2 — implications, design choices |
| Phase 11.3 (impact), Phase 11.4 (trends) | Report sections | LO2 + LO3 |
| Phase 11.4 (TTA, foundation models, NIR) | Report section | LO3 — emerging trends |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Colab session disconnects mid-training | High | Medium | Save checkpoint to Drive every epoch; use Colab Pro or background execution |
| Dataset download fails in Colab | Medium | Low | Cache to Drive; provide fallback manual download instructions in README |
| Class imbalance degrades minority class F1 | Medium | Medium | Class-weighted loss; monitor per-class F1 separately |
| Night augmentation too aggressive — destroys patch information | Medium | High | Visual validation in Phase 4.3; calibrate parameter ranges conservatively |
| Grad-CAM heatmaps are noisy / uninformative | Low | Medium | Try GradCAM++, ScoreCAM as alternatives; use layer at correct depth |
| E5 domain adaptation does not improve E3 | Low | Low | Still a valid scientific result — report honestly and discuss why (e.g. simulation mismatch) |
| Word count exceeds 2,800 | Medium | Medium | Draft all sections then cut; appendix for supplementary material |

---

*Plan version: 1.0 | Author: Project Lead | Target: WMG9B7-15 Individual Assessment 2526*
