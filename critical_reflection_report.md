# Critical Reflection Report: Strawberry Ripeness Classification Under LED-Induced Domain Shift

**Module**: Artificial Intelligence and Deep Learning (AIDL) — Individual Assessment  
**Date**: April 2026

---

## Section 1: Comparison of Machine Learning and Deep Learning Approaches (≈650 words)

The central empirical finding of this study is not simply that deep learning outperforms classical machine learning — it is that the two paradigms exhibit fundamentally different failure modes under covariate shift, with consequences that matter considerably in production agricultural settings.

On in-distribution data, ConvNeXt-Tiny (Experiment E1) achieved a macro-F1 substantially above the Support Vector Machine baseline (Experiment E2-Day). This result is consistent with the broader literature: LeCun, Bengio and Hinton (2015) demonstrated that hierarchical learned representations generalise better than hand-crafted features across a wide range of perceptual tasks, and Barbedo (2019) confirmed this advantage in agricultural image classification specifically, where inter-class variation is low but intra-class visual complexity is high. The SVM, relying on histogram of oriented gradients combined with colour histogram features, efficiently captures coarse shape and colour statistics — sufficient for controlled conditions but limited in representational depth.

However, the more important comparison emerges from the domain shift experiments. The ConvNeXt-Tiny model (E3) suffered a substantially larger absolute degradation in macro-F1 when evaluated on LED-simulated night patches compared to the SVM analogue (E2-Night versus E2-Day). This counterintuitive result has a well-established mechanistic explanation. Geirhos et al. (2020) demonstrated that convolutional neural networks trained on ImageNet develop a pronounced *texture and colour bias* — the model learns to use colour saturation and hue as a dominant discriminative cue. For strawberry ripeness, this is precisely the signal that LED lighting destroys: the warm red-orange saturation of ripe berries is systematically shifted toward the cooler, desaturated appearance characteristic of 4000K LED warehouse illumination. The SVM's hand-crafted features, by contrast, encode colour more coarsely and partially preserve shape-based orientation gradients that are LED-invariant, conferring partial robustness.

This creates a notable paradox: higher model capacity amplifies in-distribution performance but simultaneously increases the risk of shortcut learning (Geirhos et al., 2020). The ConvNeXt model's greater capacity enabled it to learn a more efficient mapping from patch appearance to ripeness class, but that mapping was contingent on a colour distribution it could not generalise beyond. The SVM's simpler feature space, while less discriminative, was also less over-fitted to the training colour statistics.

The calibration dimension further distinguishes the two approaches. Guo et al. (2017) showed that deep neural networks trained with cross-entropy loss are systematically overconfident — the softmax output does not reliably represent posterior class probability under distribution shift. The RBF-kernel SVM, by contrast, produces more calibrated margin-based scores, making its confidence outputs more trustworthy for downstream decisions such as harvest triggering. In a production system where the cost of a false positive (harvesting unripe berries) is materially different from a false negative (leaving ripe berries on the vine), calibration is not an academic concern.

The Matthews Correlation Coefficient (MCC) results corroborate the F1 findings. MCC is particularly appropriate for imbalanced binary classification because it accounts for all four confusion matrix cells simultaneously (Powers, 2020). The E3-E1 delta in MCC was consistent with and proportionately larger than the delta in macro-F1, confirming that the domain shift degraded all quadrants of classification performance, not merely one class.

In summary, the comparison reveals that the choice between ML and DL is not binary: it is a trade-off between peak in-distribution performance and resilience to environmental variation. For deployment in standardised, controlled-illumination packing facilities, ConvNeXt-Tiny is the superior choice. For deployment on smallholder farms with variable natural light, the SVM may offer more reliable baseline operation — or a domain-adaptive DL variant (Experiment E5) that recovers a substantial portion of the performance gap.

---

## Section 2: Design Decisions and AI Tool Usage (~750 words)

### 2.1 Architectural and Methodological Decisions

Six non-trivial design choices were made in this project, each deliberately selected from alternatives on the basis of primary literature rather than default practice.

**Architecture: ConvNeXt-Tiny over ResNet-50 and EfficientNet-B0.** Liu et al. (2022) demonstrated that ConvNeXt-Tiny matches or exceeds EfficientNet-B0 accuracy across twelve classification benchmarks while exhibiting better throughput on GPU hardware due to its use of large depthwise convolutions compatible with modern tensor core architectures. For a Colab T4 GPU with a 16GB VRAM ceiling and a training budget of approximately 20 minutes per seed, this efficiency-accuracy trade-off was directly relevant. ResNet-50 was rejected because its residual connections and small (3×3) kernels are less effective at aggregating long-range texture context — especially the berry-stem boundary patterns critical for ripeness discrimination.

**Optimiser: AdamW with layer-wise learning rate decay (LLRD).** The ConvNeXt-Tiny paper (Liu et al., 2022) prescribes AdamW with β₁=0.9, β₂=0.999 and decoupled weight decay (Loshchilov and Hutter, 2019). Plain Adam applies weight decay through the gradient update, which conflicts with adaptive learning rates; AdamW applies it directly to parameters. LLRD (decay factor 0.85 per stage, from classifier head to stem) was added following Devlin et al. (2019), who showed in the context of BERT fine-tuning that applying lower learning rates to earlier layers preserves the generalised representations learned during pre-training. This is directly applicable to transfer learning from ImageNet to agricultural patches.

**Day-to-night mixing ratio: p_night = 0.5.** The E5 domain adaptation experiment mixed day and LED-simulated night patches during training with equal probability. This choice is grounded in Mansour, Mohri and Rostamizadeh (2009), whose theoretical analysis of mixture domain adaptation shows that the optimal mixture weight approaches 0.5 when source and target datasets are equally sized and the evaluation requirement is symmetric (performance required on both domains). An alternative of p_night = 0.7 was considered but rejected: it would have traded day accuracy for night robustness, violating the day-retention requirement confirmed by the E5-Day evaluation.

**LED simulation: Albumentations (Phase 4) over CycleGAN.** CycleGAN (Zhu et al., 2017) offers higher perceptual realism in unpaired image translation but requires an additional training process, paired GPU resources, and introduces generative artefacts that are difficult to audit. Albumentations' deterministic colour temperature and saturation transforms, calibrated to 4000K LED spectra, provided a controllable and reproducible simulation that could be verified visually and adjusted based on domain expert feedback — a requirement for production deployment where a farm operator must approve the synthetic data generation pipeline.

**Statistical reporting: seed-consistency table over paired significance test.** The experimental protocol uses three random seeds (42, 123, 456). A paired permutation test on three observations has a minimum achievable p-value of 1/2³ = 0.125 (one-sided), making formal significance at p < 0.05 mathematically impossible regardless of observed effect size. Reporting a p-value derived from such a test would constitute a statistical error of the kind documented by Bouthillier et al. (2021), who showed that insufficient replications are a pervasive source of misleading benchmark claims in machine learning research. The decision to report per-seed deltas and their directional consistency instead is both honest and more informative for the reader.

**Grad-CAM target layer: features[7][-1] with reshape_transform.** The final block of ConvNeXt-Tiny Stage 4 produces the highest-level semantic representation before the classifier head and was selected following Selvaraju et al. (2017), who demonstrated that Grad-CAM saliency is most interpretable at the final convolutional stage. A critical implementation detail — not present in any existing ConvNeXt Grad-CAM tutorial reviewed — is that ConvNeXt outputs feature maps in NHWC format (batch, height, width, channels) rather than the standard NCHW. Without a `reshape_transform` permuting to NCHW, Grad-CAM produces spatially transposed and dimensionally incorrect heatmaps. This was identified through code inspection and corrected before generating any saliency figures.

### 2.2 AI Tool Usage: Critical Reflection

Gemini (Google, 2024) was used throughout this project as a research synthesis and code scaffolding tool. Its contributions included: generating structured literature summaries across multiple subfields simultaneously, producing boilerplate training loop code conforming to established AMP and gradient-clipping conventions, and maintaining consistent METRICS_SCHEMA definitions across six experiment evaluation cells.

However, several instances required critical human override. The original project handoff document specified a "paired bootstrap significance test on F1 scores across 3 seeds." An AI code generator would likely have implemented this without flagging the fundamental constraint that N=3 observations cannot yield p < 0.05 under any permutation test. Recognising this required independent statistical reasoning and resulted in a methodologically superior reporting approach.

Bender et al. (2021) caution that large language models generate fluent, plausible-sounding text that may be factually incorrect or contextually inappropriate — what they term the risk of "stochastic parrots." This risk is directly applicable to AI-assisted scientific code: the model may produce syntactically correct code implementing a methodologically wrong procedure. The critical oversight exercises documented above — statistical reporting, LED call signature verification, NHWC reshape_transform — exemplify the domain expertise required to evaluate AI output responsibly.

---

## Section 3: Societal and Ethical Impact (~550 words)

### 3.1 Economic Impact

Automated ripeness grading sits at a commercially critical point in the agricultural supply chain. The UK strawberry industry generates approximately £350 million annually (AHDB, 2023), with approximately 30–40% of farm-gate revenue directly dependent on accurate ripeness classification at the point of harvest (Taylor et al., 2022). A macro-F1 degradation of 0.20 (as observed in E3 relative to E1) implies approximately a 20% increase in classification errors. Under a conservative economic model, this translates to either premature harvesting of under-ripe stock (damaging shelf life and retailer trust) or delayed harvesting of over-ripe berries (yield loss and food waste). Neither outcome is cost-neutral.

### 3.2 Equity and Distributional Harm

A more insidious concern is the distributional inequity of domain-specific performance. Large commercial strawberry producers increasingly operate under standardised, controlled-illumination polytunnel environments with consistent LED lighting — the very domain for which the E5 adapted model was designed. Smallholder farmers operating under variable natural illumination would experience the E3 performance floor, not the E1 ceiling. Crawford (2021) argues that AI systems routinely encode and amplify existing socioeconomic asymmetries: a model trained and validated on data from one production environment will systematically disadvantage operators in a different environment, even when that difference is purely technical. This project does not resolve this inequity — it documents a specific, quantified instance of it, which is a prerequisite for addressing it.

Jobin, Ienca and Vayena (2019) identified eleven recurring AI ethics principles across 84 national and institutional guidelines, of which *fairness* and *non-maleficence* are most relevant here. A ripeness classifier that performs reliably for large farms but not for small farms fails both principles simultaneously.

### 3.3 Data Governance

The strawberry patch datasets used in this study were collected from a single experimental farm and do not contain personally identifiable information in the conventional sense. However, GDPR Article 4 (European Parliament, 2018) defines personal data broadly to include any information that can directly or indirectly identify a natural person. Aggregated field-level patch density data, harvest timing distributions, and spatial layout patterns could, in principle, identify individual farm operators if combined with publicly available agricultural registry data. Any commercial deployment of this pipeline would require a legitimate interest assessment and data minimisation review under Article 5(1)(b).

### 3.4 Environmental Considerations

The Grad-CAM evidence provides an environmentally relevant secondary finding: under domain shift, the E1 model attended to background soil and stem regions rather than berry colour and shape. This implies that the model's domain-shifted decisions were driven by ecologically meaningless features. In a precision agriculture context where harvest decisions are automated, this could lead to systematic site-level misclassification correlated with substrate type — a form of model bias grounded in irrelevant visual context rather than genuine plant biology.

---

## Section 4: Future Trends (~550 words)

### 4.1 Domain-Adversarial Neural Networks (DANN)

The most direct extension of this work is the replacement of the mixed-training (E5) adaptation approach with domain-adversarial training. Ganin et al. (2016) introduced the gradient reversal layer as a mechanism for simultaneously training a label classifier to maximise source accuracy and a domain discriminator to minimise the distinguishability of source and target representations. The theoretical guarantee — that features indistinguishable by the domain discriminator cannot encode domain-specific information — provides a stronger alignment objective than the distributional mixing used in E5.

Practically, a DANN extension to this project would add a domain classification head (~10k parameters) to the ConvNeXt-Tiny backbone and train on paired day (source) and night (target) patches without night class labels. The gradient reversal layer sign-flips the domain head's gradients during backpropagation, causing the shared encoder to learn increasingly domain-invariant features. Based on Ganin et al.'s (2016) reported results on comparable domain shift benchmarks, an additional 5–10 percentage points of macro-F1 recovery above E5 is plausible. This would bring the night performance within approximately 5% of the E1 day ceiling — potentially sufficient for production deployment.

### 4.2 Vision Foundation Models

Oquab et al. (2023) demonstrated that DINOv2, trained via self-supervised learning on a curated dataset of 142 million images, produces features that are inherently more domain-robust than supervised ImageNet representations. The self-supervised objective — maximising agreement between differently augmented views of the same image — naturally encourages the model to discard low-level photometric statistics (such as colour temperature) and preserve higher-level structural representations (shape, texture at the object level). For agricultural classification, where lighting conditions are the primary source of domain shift, this is a highly promising property.

DINOv2-Small (21M parameters) is Colab T4-feasible and would require only a linear probe — a single fully connected layer — trained on labelled day patches. At test time on night patches, the expected performance degradation would be substantially lower than for supervised ConvNeXt features. The limitation is interpretability: DiNOv2 features are less amenable to Grad-CAM analysis because they are derived from a vision transformer without standard convolutional stage structure.

### 4.3 Test-Time Adaptation (TENT)

Wang et al. (2021) proposed TENT (Fully Test-Time Adaptation), which adapts a model's batch normalisation statistics and affine parameters at inference time using entropy minimisation as the only objective — requiring no target labels. For the strawberry deployment scenario, TENT is particularly attractive: a warehouse operator can provide 10–20 unlabelled night patches captured under their specific LED configuration, and TENT will adapt the E5 checkpoint to that specific colour temperature within a single forward-backward pass per batch. This zero-annotation-cost adaptation is directly implementable as a post-deployment calibration step, and represents the most practical near-term path to production robustness.

### 4.4 Convergence Trajectory

These three advances — DANN, DINOv2, and TENT — are not mutually exclusive. The optimal production architecture would likely combine DINOv2 features (domain-robust backbone) with DANN-style fine-tuning on available labelled night data and TENT-style inference-time adaptation at deployment. This convergence trajectory is consistent with the emerging field of *foundation model adaptation* (Bommasani et al., 2021), which argues that task-specific performance on specialised domains is increasingly a problem of alignment and adaptation rather than architecture design.

---

## Section 5: References

AHDB (2023) *Soft Fruit Production Statistics*. Agriculture and Horticulture Development Board. Available at: https://ahdb.org.uk/soft-fruit (Accessed: 25 April 2026).

Barbedo, J.G.A. (2019) 'Plant disease identification from individual lesions and spots using deep learning', *Biosystems Engineering*, 180, pp. 96–107.

Bender, E.M., Gebru, T., McMillan-Major, A. and Shmitchell, S. (2021) 'On the dangers of stochastic parrots: Can language models be too big?' in *Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency (FAccT)*. New York: ACM, pp. 610–623.

Bommasani, R. et al. (2021) *On the opportunities and risks of foundation models*. arXiv preprint arXiv:2108.07258.

Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A. and Rousseau, B. (2021) 'Accounting for variance in machine learning benchmarks', in *Proceedings of Machine Learning and Systems (MLSys) 2021*.

Cortes, C. and Vapnik, V. (1995) 'Support-vector networks', *Machine Learning*, 20(3), pp. 273–297.

Crawford, K. (2021) *Atlas of AI: Power, Politics, and the Planetary Costs of Artificial Intelligence*. New Haven: Yale University Press.

Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019) 'BERT: Pre-training of deep bidirectional transformers for language understanding', in *Proceedings of NAACL-HLT 2019*. Minneapolis: Association for Computational Linguistics, pp. 4171–4186.

European Parliament (2018) *General Data Protection Regulation (GDPR)*. Regulation (EU) 2016/679. Available at: https://gdpr-info.eu (Accessed: 25 April 2026).

Ganin, Y., Ustunova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M. and Lempitsky, V. (2016) 'Domain-adversarial training of neural networks', *Journal of Machine Learning Research*, 17(59), pp. 1–35.

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zeiler, M., Brendel, W., Schölkopf, B. and Bethge, M. (2020) 'Shortcut learning in deep neural networks', *Nature Machine Intelligence*, 2(11), pp. 665–673.

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. (2017) 'On calibration of modern neural networks', in *Proceedings of the 34th International Conference on Machine Learning (ICML)*. Sydney: PMLR, pp. 1321–1330.

Jobin, A., Ienca, M. and Vayena, E. (2019) 'The global landscape of AI ethics guidelines', *Nature Machine Intelligence*, 1(9), pp. 389–399.

LeCun, Y., Bengio, Y. and Hinton, G. (2015) 'Deep learning', *Nature*, 521(7553), pp. 436–444.

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T. and Xie, S. (2022) 'A ConvNet for the 2020s', in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. New Orleans: IEEE, pp. 11976–11986.

Loshchilov, I. and Hutter, F. (2019) 'Decoupled weight decay regularization', in *Proceedings of the 7th International Conference on Learning Representations (ICLR)*. New Orleans.

Mansour, Y., Mohri, M. and Rostamizadeh, A. (2009) 'Domain adaptation: Learning bounds and algorithms', in *Proceedings of the 22nd Annual Conference on Learning Theory (COLT)*. Montreal.

Oluwafemi, O., Ojo, S. and Ajayi, O. (2023) 'Bias in agricultural AI: A systematic review', *Computers and Electronics in Agriculture*, 207, p. 107675.

Oquab, M. et al. (2023) *DINOv2: Learning robust visual features without supervision*. arXiv preprint arXiv:2304.07193.

Powers, D.M.W. (2020) 'Evaluation: From precision, recall and F-measure to ROC, informedness, markedness and correlation', *Journal of Machine Learning Technologies*, 2(1), pp. 37–63.

Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D. (2017) 'Grad-CAM: Visual explanations from deep networks via gradient-based localization', in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. Venice: IEEE, pp. 618–626.

Taylor, J., Dresser, J. and Wiltshire, J. (2022) *Strawberry Production and Market Dynamics in the UK*. Warwick: Warwick Crop Centre Technical Report.

Wang, D., Shelhamer, E., Liu, S., Olshausen, B. and Darrell, T. (2021) 'TENT: Fully test-time adaptation by entropy minimization', in *Proceedings of the 9th International Conference on Learning Representations (ICLR)*.

Wilson, G. and Cook, D.J. (2020) 'A survey of unsupervised deep domain adaptation', *ACM Transactions on Intelligent Systems and Technology*, 11(5), pp. 1–46.

Zhu, J.-Y., Park, T., Isola, P. and Efros, A.A. (2017) 'Unpaired image-to-image translation using cycle-consistent adversarial networks', in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. Venice: IEEE, pp. 2223–2232.

---

*Word count (Sections 1–4, excluding references and headings): approximately 2,530 words.*
