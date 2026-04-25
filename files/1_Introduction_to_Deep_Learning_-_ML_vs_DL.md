# 1 Introduction to Deep Learning - ML vs DL

> Upgraded Markdown conversion. The original page image is included for every page so diagrams, screenshots, tables, slide layout and visual details are preserved. Extracted text is also provided below each page for searchable study notes.

## Contents

- [Page 1: Introduction to](#page-1)
- [Page 2: Agenda](#page-2)
- [Page 3: Learning Outcomes](#page-3)
- [Page 4: I. Introduction to Deep](#page-4)
- [Page 5: Introduction](#page-5)
- [Page 6: What are Neural Networks?](#page-6)
- [Page 7: The Journey](#page-7)
- [Page 8: Why Did Deep Learning Overtake Traditional ML?](#page-8)
- [Page 9: Traditional ML vs Deep Learning](#page-9)
- [Page 10: Scalability: The Deep Learning Advantage](#page-10)
- [Page 11: When to Use ML vs. Deep Learning](#page-11)
- [Page 12: Think – Pair – Share](#page-12)
- [Page 13: II. Common Applications](#page-13)
- [Page 14: Computer Vision](#page-14)
- [Page 15: Natural Language Processing](#page-15)
- [Page 16: Generative AI](#page-16)
- [Page 17: Beyond Vision & Language](#page-17)
- [Page 18: Summary](#page-18)
- [Page 19: III. Main Challenges:](#page-19)
- [Page 20: Challenge 1 — The Cost of Deep Learning](#page-20)
- [Page 21: Challenge 2 — Explainability Matters. The Stakes Are Real!](#page-21)
- [Page 22: Interpretability vs Explainability](#page-22)
- [Page 23: The Black Box Problem in Deep Learning](#page-23)
- [Page 24: The Black Box Problem in Deep Learning](#page-24)
- [Page 25: Core XAI Techniques](#page-25)
- [Page 26: Core XAI Techniques](#page-26)
- [Page 27: XAI Method Selection Guide — Choosing the Right Tool](#page-27)
- [Page 28: Challenge 3 - Ethical Dimensions of Deep Learning](#page-28)
- [Page 29: Data Privacy and Security](#page-29)
- [Page 30: The Regulatory & Legal Landscape](#page-30)
- [Page 31: The Regulatory & Legal Landscape](#page-31)
- [Page 32: Challenge 4 — Communicating Explanations to](#page-32)
- [Page 33: Challenge 5 — AI Governance: Aligning DL with](#page-33)
- [Page 34: Ziad Obermeyer et al., (2019) Dissecting racial bias in an algorithm](#page-34)
- [Page 35: Summary](#page-35)
- [Page 36: IV. From Deep Learning to](#page-36)
- [Page 37: The Reality Gap: Notebook → Production](#page-37)
- [Page 38: Challenge 1 — Dependency & Environment Hell](#page-38)
- [Page 39: Challenge 2 — Data Pipeline at Scale](#page-39)
- [Page 40: Challenge 3 — Monitoring & Model Drift](#page-40)
- [Page 41: Security & Compliance](#page-41)
- [Page 42: V. Emerging Trends](#page-42)
- [Page 43: Foundation Models & Multimodal AI](#page-43)
- [Page 44: Efficient Deep Learning & Agentic AI](#page-44)
- [Page 45: Summary](#page-45)
- [Page 46: Do you](#page-46)
- [Page 47: warwick.ac.uk/wmg](#page-47)

---


<a id="page-1"></a>

## Page 1: Introduction to

![Original page 1](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_001.png)

### Extracted content

1 | © WMG 2026
Artificial Intelligence & Deep learning
Introduction to
Deep Learning
Dr Leonardo Alves Dias
2026


<a id="page-2"></a>

## Page 2: Agenda

![Original page 2](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_002.png)

### Extracted content

2 | © WMG 2026
Agenda
I.
Introduction to Deep Learning
II.
Common Applications
III. Main
Challenges:
Explainability,
Interpretability & AI Governance
IV. From Deep Learning to Production
V. Emerging Trends


<a id="page-3"></a>

## Page 3: Learning Outcomes

![Original page 3](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_003.png)

### Extracted content

3 | © WMG 2026
Learning Outcomes
LO1. Critically evaluate and explain the key differences between traditional machine learning and
deep learning approaches.
LO2. Critically evaluate deep learning and artificial intelligence solutions and their implications.
LO3. Demonstrate an understanding of emerging trends and research development in AI.
LO4. Design and evaluate deep learning models for practical applications.
LO5. Collaborate effectively as part of a group to develop and communicate a practical AI solution
by presenting findings through reproducible code, visualisations, or technical reporting.
Week 1
Week 2
Week 3
Week 4
•
Traditional
Machine
Learning
vs
Deep
Learning
•
Artificial
Neural
Networks (ANN)
•
Multi-Layer
Perceptron
(MLP)
•
Convolutional
Neural
Networks (CNN)
•
Long
Short-Term
Memory (LSTM)
•
Introduction
to
Transformers.
•
Introduction
to
Large
Language Models (LLMs)
•
Retrieval-Augmented
Generation Systems


<a id="page-4"></a>

## Page 4: I. Introduction to Deep

![Original page 4](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_004.png)

### Extracted content

4 | © WMG 2026
I. Introduction to Deep
Learning


<a id="page-5"></a>

## Page 5: Introduction

![Original page 5](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_005.png)

### Extracted content

5 | © WMG 2026
Introduction
“Deep learning is a subset of machine
learning driven by multilayered neural
networks whose design is inspired by the
structure of the human brain. Deep
learning models power most state-of-
the-art artificial intelligence (AI) today,
from computer vision and generative
AI to self-driving cars and robotics.”
(IBM, 2026)
The name ”deep learning” was invented
to rebrand artificial neural networks,
which became unpopular in the 2000s.
The
basic
idea
of
having
many
computational
units
that
become
intelligent only through their interactions
with one another is inspired by the
brain.
What Is Deep Learning? | IBM


<a id="page-6"></a>

## Page 6: What are Neural Networks?

![Original page 6](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_006.png)

### Extracted content

6 | © WMG 2026
What are Neural Networks?


<a id="page-7"></a>

## Page 7: The Journey

![Original page 7](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_007.png)

### Extracted content

7 | © WMG 2026
The Journey
Six decades of evolution from simple perceptrons to foundation models.
Perceptron
Backpropagation
AlexNet
Transformers
Foundation Models
Rosenblatt’s single-layer
model
Rumelhart, Hinton &
Williams
ImageNet breakthrough
with deep CNNs
Attention Is All
You Need
GPT, DALL-E, diffusion
models at scale
1958
1986
2012
2017
2020s
Three catalysts converged enabling the evolution: massive datasets, GPU computing power, and algorithmic
innovations, enabling neural networks to finally fulfil their potential.


<a id="page-8"></a>

## Page 8: Why Did Deep Learning Overtake Traditional ML?

![Original page 8](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_008.png)

### Extracted content

8 | © WMG 2026
Why Did Deep Learning Overtake Traditional ML?
Deep learning now dominates computer vision, NLP, speech, and generative AI, achieving
superhuman performance in many benchmarks.
Data Explosion
Internet-scale datasets with
billions of images, texts, and
interactions became available.
Compute Revolution
GPUs and TPUs delivered
100x speedups, making deep
network training feasible.
Algorithmic Breakthroughs
Dropout, batch normalisation,
residual
connections,
and
attention mechanisms.


<a id="page-9"></a>

## Page 9: Traditional ML vs Deep Learning

![Original page 9](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_009.png)

### Extracted content

9 | © WMG 2026
Traditional ML vs Deep Learning
The Fundamental Shift: From hand-crafted features to learned representations.
Traditional Machine Learning
•
Human expert designs features
•
Domain knowledge required upfront
•
Feature quality limits performance
•
Works well with small, structured data
•
More interpretable models
Deep Learning
•
Network learns features automatically
•
Minimal manual feature engineering
•
Performance scales with more data
•
Excels with unstructured data
•
Often a “black box”
Raw Data
Feature Eng.
Model
Raw Data
End-to-End DL Model


<a id="page-10"></a>

## Page 10: Scalability: The Deep Learning Advantage

![Original page 10](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_010.png)

### Extracted content

10 | © WMG 2026
Scalability: The Deep Learning Advantage
Traditional ML
Performance
plateaus
quickly,
regardless of additional data
Small DL Network
Improves
steadily,
but
the
architecture limits the ceiling
Large DL Network
Continues to improve with scale –
the core DL advantage
Amount of Data
Performance


<a id="page-11"></a>

## Page 11: When to Use ML vs. Deep Learning

![Original page 11](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_011.png)

### Extracted content

11 | © WMG 2026
When to Use ML vs. Deep Learning
A decision framework for practitioners and consultants
Criterion
Traditional ML
Deep Learning
Data Type
Structured / tabular
Unstructured (images, text, audio)
Dataset Size
Small to medium (< 100K)
Large (100K+, ideally millions)
Feature Engineering
Manual, domain-driven
Automatic, end-to-end
Interpretability
High (e.g., decision trees, LASSO)
Low (black-box; requires XAI tools)
Compute Cost
Low (CPU sufficient)
High (GPU/TPU required)
Training Time
Minutes to hours
Hours to weeks
Best For
Tabular prediction, anomaly detection
Vision, NLP, generative, multimodal
Difference Between Machine Learning and Deep Learning - GeeksforGeeks


<a id="page-12"></a>

## Page 12: Think – Pair – Share

![Original page 12](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_012.png)

### Extracted content

12 | © WMG 2026
Think – Pair – Share
Scenario: A retail bank wants to improve its fraud detection system. Currently, they use a
random forest model on structured transaction data (amount, location, time, merchant
category). They also have access to:
•
Customer call centre transcripts (unstructured text)
•
CCTV footage from ATMs (unstructured video)
•
500,000 labelled historical transactions
Should they switch entirely to deep learning, keep traditional ML, or use a hybrid
approach? Justify your recommendation.


<a id="page-13"></a>

## Page 13: II. Common Applications

![Original page 13](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_013.png)

### Extracted content

13 | © WMG 2026
II. Common Applications


<a id="page-14"></a>

## Page 14: Computer Vision

![Original page 14](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_014.png)

### Extracted content

14 | © WMG 2026
Computer Vision
Convolutional Neural Networks (CNNs).
Key Architectures: ResNet, YOLO, EfficientNet, Vision
Transformers (ViT), U-Net.
Medical Imaging
Detecting tumours, retinal disease screening, pathology
slide analysis – matching or exceeding specialist accuracy.
Autonomous Vehicles
Real-time object detection, lane tracking, pedestrian
recognition using CNNs and sensor fusion.
Manufacturing QA
Visual defect inspection on production lines achieving
99%+ accuracy with sub-second latency.


<a id="page-15"></a>

## Page 15: Natural Language Processing

![Original page 15](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_015.png)

### Extracted content

15 | © WMG 2026
Natural Language Processing
The transformer revolution
Applications Transforming Industry
RNNs / LSTMs
Sequential
processing
Attention (2017)
Parallel processing
BERT / GPT
Pre-trained LLMs
GPT-4 / Claude
Reasoning at scale
•
Sentiment Analysis: Real-time brand monitoring and
customer feedback classification at scale
•
Machine Translation: Near-human translation quality
across 100+ language pairs
•
Document
Intelligence:
Automated
extraction,
summarisation, and question answering over enterprise
documents
•
Conversational
AI:
Task-oriented
chatbots,
coding
assistants, and autonomous customer service agents


<a id="page-16"></a>

## Page 16: Generative AI

![Original page 16](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_016.png)

### Extracted content

16 | © WMG 2026
Generative AI
Creating new content from learned patterns.
The generative AI market is projected to exceed $200 billion by 2030, fundamentally reshaping
creative and knowledge work.
(Abiresearch, 2024)
Image Generation
GANs, Diffusion Models
Text-to-image, inpainting, style transfer, super-resolution, etc.
E.g., DALL-E 3, Stable Diffusion, Midjourney.
Text Generation
Large Language Models
Content creation, code generation, reasoning, summarisation, etc.
E.g., GPT-4, Claude, Gemini, LLaMA.
Audio & Video
Diffusion + Transformers
Text-to-speech, music composition, video synthesis, etc.
E.g., Sora, Kling, MusicGen.
https://www.abiresearch.com/blog/generative-ai-software-market-report-summary


<a id="page-17"></a>

## Page 17: Beyond Vision & Language

![Original page 17](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_017.png)

### Extracted content

17 | © WMG 2026
Beyond Vision & Language
Other DL Domains
•
Reinforcement Learning: Game playing
(AlphaGo),
robotics
control,
recommendation systems
•
Speech
Recognition:
Real-time
transcription,
voice
assistants,
meeting
summarisation
•
Time-Series Forecasting: Energy demand
prediction,
financial
modelling,
supply
chain optimisation
•
Drug
Discovery:
Molecular
property
prediction, protein structure (AlphaFold),
drug-target interaction
Industry Impact
•
40% - Google DeepMind: reduction in
data centre cooling energy with DL-based
control systems
•
Real-time – Tesla: end-to-end neural
network for autonomous driving, replacing
modular pipelines
•
360K hrs – JPMorgan: of legal document
review automated per year using NLP
models
•
200M+ - DeepMind: protein structures
predicted
by
AlphaFold,
accelerating
biomedical research


<a id="page-18"></a>

## Page 18: Summary

![Original page 18](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_018.png)

### Extracted content

18 | © WMG 2026
Summary
We introduced the difference between Machine and Deep
learning.
We discussed common Deep Learning applications.


<a id="page-19"></a>

## Page 19: III. Main Challenges:

![Original page 19](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_019.png)

### Extracted content

19 | © WMG 2026
III. Main Challenges:
Explainability,
Interpretability
& AI Governance


<a id="page-20"></a>

## Page 20: Challenge 1 — The Cost of Deep Learning

![Original page 20](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_020.png)

### Extracted content

20 | © WMG 2026
Challenge 1 — The Cost of Deep Learning
Data and computational barriers that every practitioner is likely to
confront:
Data Volume & Labelling
•
Deep learning is data-hungry; manual labelling is expensive, slow,
and introduces human bias into ground truth.
Computational Demands
•
Training requires specialised hardware (GPUs/TPUs). Cloud costs
scale non-linearly with model size.
Environmental Impact
•
The carbon footprint of training large models is significant –
raising sustainability concerns across the industry.
570 GB
of text used to train
GPT-3
$100M+
estimated cost to train
a frontier LLM
1,287 MWh
energy to train GPT-3
(≈ US homes/year: 120)


<a id="page-21"></a>

## Page 21: Challenge 2 — Explainability Matters. The Stakes Are Real!

![Original page 21](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_021.png)

### Extracted content

21 | © WMG 2026
Challenge 2 — Explainability Matters. The Stakes Are Real!
Explainability is not a nice-to-have feature; it is a governance, legal, and ethical requirement.
Model recommends less care
for Black patients
Obermeyer et al. (2019) — A
widely-used US health algorithm
systematically assigned lower risk
scores to Black patients than
equally sick white patients. Root
cause: predicting healthcare costs
(proxy), not health needs.
Opaque models in high-stakes
domains reproduce and amplify
existing inequalities.
Apple Card gender bias
complaint
2019 — Male applicants received
credit limits up to 20× higher than
female partners with better credit
scores. Goldman Sachs could not
explain the algorithm's decisions
to regulators.
Unexplainable decisions expose
organisations to legal liability and
reputational damage.
COMPAS recidivism tool bias
ProPublica (2016) — COMPAS
predicted twice the false positive
rate for Black defendants vs white
defendants.
The
tool's
inner
workings were legally protected
as commercial IP.
When models affect liberty and
rights, opacity is not a technical
problem — it is a justice problem.
Healthcare
Finance
Criminal Justice


<a id="page-22"></a>

## Page 22: Interpretability vs Explainability

![Original page 22](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_022.png)

### Extracted content

22 | © WMG 2026
Interpretability vs Explainability
Key question: Does your stakeholder need to understand the model, or understand a specific
decision?
Explainability (XAI)
The degree to which a human can
understand the cause of a decision
using an external explanation method.
Post-hoc tools generate explanations
for opaque models.
Interpretability
The degree to which a human can
understand the cause of a decision
from the model’s structure itself.
The model is intrinsically transparent,
i.e., no additional tools needed.
Scope
Method
type
Model Specific
Works
only
for
one
architecture.
Grad-CAM (CNNs), Attention
weights (Transformers)
Local
Explains a single prediction.
Why did THIS patient get
denied?
Model-Agnostic
Works for any model as a black
box.
SHAP, LIME, PDP plots
Global
Explains
the
overall
model
behaviour.
Which features matter most on
average?


<a id="page-23"></a>

## Page 23: The Black Box Problem in Deep Learning

![Original page 23](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_023.png)

### Extracted content

23 | © WMG 2026
The Black Box Problem in Deep Learning
The deeper the network, the more powerful — and the less we can explain why it works.
Millions of parameters
A ResNet-50 has 25M weights. No human can trace a decision
through 50 layers of matrix multiplications.
Non-linear feature composition
Each layer learns abstract combinations of lower-level
features. Final representation has no direct semantic label.
Distributed representations
Concepts are not localised to single neurons but spread
across layers. No single parameter = one concept.
Emergent behaviour at scale
LLMs develop capabilities (few-shot learning, reasoning) not
present in training objectives. Causality unclear.
Entangled latent spaces
Embeddings mix useful signal with spurious correlations from
training data — nearly impossible to disentangle post-hoc.
Why DL Models Are Opaque
Decision Tree
Linear Regression
Random Forest
Gradient Boost
Deep Neural Network
Large LLM
← More Interpretable
More Accurate →
Interp: High
Interp: High
Interp: Med
Interp: Low
Interp: Very Low
Interp: None
The Accuracy–Interpretability Spectrum


<a id="page-24"></a>

## Page 24: The Black Box Problem in Deep Learning

![Original page 24](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_024.png)

### Extracted content

24 | © WMG 2026
The Black Box Problem in Deep Learning
Explainability is not optional in high-stakes domains!
Why It Matters
Regulatory compliance: GDPR Article 22
grants individuals the right to an explanation
for automated decisions.
Clinical trust: Doctors will not act on
predictions they cannot verify or understand.
Financial accountability: Regulators require
justification for credit and insurance decisions.
Safety-critical systems: Autonomous vehicles
and aerospace demand traceable decision
logic
Interpretability Techniques
SHAP: Shapley-value feature attributions –
model-agnostic, theoretically grounded.
LIME: Local surrogate models that explain
individual predictions.
Grad-CAM: Gradient-based visual heatmaps
showing where CNNs “look”.
Attention
Maps:
Visualise
which
tokens/regions transformers attend to in their
reasoning.


<a id="page-25"></a>

## Page 25: Core XAI Techniques

![Original page 25](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_025.png)

### Extracted content

25 | © WMG 2026
Core XAI Techniques
SHAP (SHapley Additive exPlanations)
A unified approach to explain the output of any machine
learning model.
How: Assigns an importance value to each feature for a
particular prediction. It uses Shapley values from game theory,
which mathematically distribute the "payout" (the prediction)
among the "players" (the input features) based on their average
contribution across all possible feature combinations.
Pros: Solid theoretical foundation (Game Theory); ensures
"consistency" and "local accuracy“; industry standard for
tabular data; powerful visualisations (force plots, beeswarm
plots).
Cons: Computationally expensive; can be very slow for large
feature sets (requires many model evaluations); can be complex
to explain to non-technical stakeholders compared to LIME.
Best for: High-stakes tabular data (finance, healthcare) where
you need a mathematically grounded explanation that accounts
for feature interactions.
LIME
Local Interpretable Model-Agnostic Explanations
How: Perturbs the input, observes predictions, and
fits a simple local surrogate model (linear) in the
neighbourhood of the instance being explained.
Pros: Truly model-agnostic; intuitive explanations
easy to communicate to non-technical stakeholders;
works across data types (text, tabular, image); faster
than SHAP for single-instance explanations.
Cons: Unstable: explanations can vary between runs
due to random perturbation sampling; sensitive to
hyperparameter
choices
(neighbourhood
size,
kernel width); local only (no global feature
importance); linear surrogate may miss non-linear
feature interactions.
Best for: Quick, human-readable explanations of
individual predictions across any model type,
especially when communicating results to non-
technical audiences.


<a id="page-26"></a>

## Page 26: Core XAI Techniques

![Original page 26](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_026.png)

### Extracted content

26 | © WMG 2026
Core XAI Techniques
No single technique is sufficient. It is good practice to combine local and global methods to obtain
robust explanations.
Attention Maps
Transformer Attention Visualisation
How: Extracts the attention weight matrices from
transformer heads. High attention between tokens
suggests the model found a relationship.
Pros: Native to Transformer architecture · no external
tools needed
Cons: Attention ≠ attribution (Jain & Wallace 2019) ·
misleading for explanation
Best for: NLP models (BERT, GPT) — note: treat as a
diagnostic hint, not ground truth
Grad-CAM
Gradient-weighted Class Activation Mapping
How:
Uses
gradients
flowing
into
the
final
convolutional layer to produce a coarse localisation
map highlighting important image regions.
Pros: No architectural changes needed · fast · visual
and intuitive for images
Cons: CNN-specific · coarse resolution · may miss fine-
grained features
Best for: Computer vision: 'why did the model classify
this chest X-ray as pneumonia?'


<a id="page-27"></a>

## Page 27: XAI Method Selection Guide — Choosing the Right Tool

![Original page 27](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_027.png)

### Extracted content

27 | © WMG 2026
XAI Method Selection Guide — Choosing the Right Tool
Start with SHAP for tabular, Grad-CAM for vision, and LIME for quick NLP checks. Always combine local + global
for a complete picture.
Method
Type
Scope
Data Modality
Pros
Cons
Best Use Case
SHAP
Post-hoc
Local+Global
Any
Theory-backed,
consistent
Slow on large
models
Tabular
/
credit,
healthcare
LIME
Post-hoc
Local
Any
Model-agnostic, intuitive
Unstable, slow
NLP,
tabular
quick
debug
Grad-CAM
Post-hoc
Local
Image
Fast, visual, no retrain
CNN-only, coarse
Medical imaging, CV
tasks
Integrated
Gradients
Post-hoc
Local
Any
Completeness axiom
Sensitive to
baseline
DNN attribution, NLP
Attention
Post-hoc
Local
Text/Sequence
Free, no overhead
≠ attribution
(caution)
Transformer debugging
only
PDP / ICE
Post-hoc
Global
Tabular
Easy to interpret
Assumes
independence
Feature
effect
visualisation
Concept-TCAV
Post-hoc
Global
Image
Human concepts
Requires concept
examples
CNN
interpretability
audits
Counterfactuals
Post-hoc
Local
Any
Actionable recourse
Multiple valid CFs
exist
Regulatory
'right
to
recourse'


<a id="page-28"></a>

## Page 28: Challenge 3 - Ethical Dimensions of Deep Learning

![Original page 28](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_028.png)

### Extracted content

28 | © WMG 2026
Challenge 3 - Ethical Dimensions of Deep Learning
The responsibility that comes with powerful technology
Algorithmic Bias
Training data reflects societal biases, and
models can amplify discrimination across
race, gender, and socioeconomic status.
Historical hiring data, facial recognition
disparities, and biased medical datasets
are well-documented examples.
Deepfakes & Misuse
Generative
models
enable
synthetic
media that can deceive at scale, from
political disinformation to identity fraud.
Detection tools are in a constant arms
race with generation capabilities.
Fairness & Protected Characteristics
Models deployed in lending, recruitment,
and criminal justice must be audited for
disparate
impact.
Fairness
metrics
(demographic parity, equalised odds)
often conflict, requiring explicit value
judgements.
Intellectual Property & Consent
Foundation models trained on internet-
scale data raise unresolved questions
about copyright, artist consent, and
attribution. Legal frameworks are evolving
rapidly (EU AI Act, ongoing litigation).


<a id="page-29"></a>

## Page 29: Data Privacy and Security

![Original page 29](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_029.png)

### Extracted content

29 | © WMG 2026
Data Privacy and Security
Large language models (LLMs) are trained on
expansive datasets that might contain Personally
Identifiable Information (PII).
Without robust anonymisation, sensitive data might
become part of the model's training dataset,
meaning that this data can potentially resurface
later. And a prompt containing sensitive data might
also be fed to the model, potentially exposing it.
Privacy laws & AI regulation:
•
EU GDPR
•
EU Digital Services Act
•
EU AI Act
•
Global AI Treaty
“Facial recognition company Clearview AI has been fined
more than £7.5m by the UK's privacy watchdog and told
to delete the data of UK residents” (BBC, 2022),


<a id="page-30"></a>

## Page 30: The Regulatory & Legal Landscape

![Original page 30](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_030.png)

### Extracted content

30 | © WMG 2026
The Regulatory & Legal Landscape
There are other regulations, such as NIST AI RMF and UK Pro-Innovation. This is continuously
evolving as new technologies emerge.
Regulation
Jurisdiction
Key XAI Requirement
Penalty
GDPR Art. 22
EU / UK
Right to explanation for automated decisions
that significantly affect individuals
4% global turnover
EU AI Act (2024)
European Union
High-risk AI must be transparent, logged,
human-overseen, and explainable to regulators
€35M or 7%
UK AI Strategy
United Kingdom
Responsible AI: accountability, transparency,
explainability in public sector AI deployment
Reputational / ICO
SR 11-7 (Banking)
USA (Fed / OCC)
Model Risk Management — all models used for
credit
decisions
require
interpretable
documentation
Regulatory sanctions
ISO/IEC 42001
International
AI
Management
System
standard:
risk
assessment,
transparency,
documentation
requirements
Certification loss


<a id="page-31"></a>

## Page 31: The Regulatory & Legal Landscape

![Original page 31](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_031.png)

### Extracted content

31 | © WMG 2026
The Regulatory & Legal Landscape
EU AI Act — Risk Classification
•
Unacceptable Risk: Social scoring, real-time biometric surveillance in public.
•
High Risk: Credit scoring, hiring, education, law enforcement, healthcare, and critical
infrastructures.
•
Limited Risk: Chatbots, deepfake generators → transparency obligations only.
•
Minimal Risk: Spam filters, recommender systems.
Deep Learning systems for credit, healthcare, and hiring are usually HIGH RISK under the EU AI
Act; thus, full explainability documentation is required.


<a id="page-32"></a>

## Page 32: Challenge 4 — Communicating Explanations to

![Original page 32](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_032.png)

### Extracted content

32 | © WMG 2026
Challenge 4 — Communicating Explanations to
Different Stakeholders
The same explanation is NOT appropriate for all audiences. A good XAI system tailors the explanation
to the consumer's technical level, role, and decision context.
Tools: SHAP dashboards · Alibi Explain · InterpretML · What-If Tool · Gradio · custom Streamlit apps
Data Scientist
Needs: Technical fidelity to
model mechanics
SHAP
beeswarm
plots
showing all features
Partial Dependence Plots
(PDP/ICE)
Permutation
feature
importance
Activation
maximisation
for CNN layers
Code-level
dashboards
(MLflow, W&B)
Product Manager
Needs: Business-level
insight, not maths
Top-3 features driving this
decision
Natural
language
summary
Confidence
score
+
uncertainty range
Comparison
to
similar
cases
Summary
dashboards
(Streamlit, Gradio)
End User
Needs: Actionable
understanding of their outcome
'Your loan was declined
because...'
Counterfactual: 'If X were
different, you would...'
Personalised,
plain
language
What they can do to
change the outcome
In-product UI / notification text
Regulator / Auditor
Needs: Accountability,
auditability, compliance
Full
model
card
documentation
System-level bias/fairness
metrics
Audit
trail
of
model
decisions
Incident
logs
and
override records
Formal documentation / API
audit logs


<a id="page-33"></a>

## Page 33: Challenge 5 — AI Governance: Aligning DL with

![Original page 33](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_033.png)

### Extracted content

33 | © WMG 2026
Challenge 5 — AI Governance: Aligning DL with
Business Requirements
Five Pillars of AI Governance
1.
Model Documentation: Model cards (Mitchell et al., 2019): intended use,
limitations, performance disaggregated by group, training data sources.
2.
Audit & Oversight: Independent internal/external audits. Human-in-the-
loop for high-stakes decisions. Appeal mechanisms for affected individuals.
3.
Performance Tracking: Continuous monitoring of model behaviour in
production. Fairness metrics dashboard. Automatic alerts on statistical
shifts.
4.
Data Governance: Data lineage tracking, consent management, purpose
limitation, retention policies. GDPR-compliant data processing agreements.
5.
Change Management: Version control for models and data. Rollback
procedures. Documented approval gates before redeployment.
Key principle: Governance must be embedded in the development process,
not bolted on as a compliance checklist at the end.
Ai Governance Lifecycle icon - Free Download PNG SVG | Streamline
Click here to see the
Governance Lifecycle


<a id="page-34"></a>

## Page 34: Ziad Obermeyer et al., (2019) Dissecting racial bias in an algorithm

![Original page 34](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_034.png)

### Extracted content

34 | © WMG 2026
Ziad Obermeyer et al., (2019) Dissecting racial bias in an algorithm
used to manage the health of populations.Science366,447-453.
DOI:10.1126/science.aax2342
BBC News (2019) Apple's ‘sexist’ credit card investigated by US
regulator. Available at: https://www.bbc.co.uk/news/business-
50365609. (Accessed: 13 March 2026).
Angwin, J., Larson, J., Mattu, S. & Kirchner, L. (2016) Machine Bias:
There’s software used across the country to predict future criminals. And
it’s biased against blacks. ProPublica. Available at:
https://www.propublica.org/article/machine-bias-risk-assessments-in-
criminal-sentencing. (Accessed: 13 March 2026).
Jain, S. & Wallace, B.C. (2019) Attention is not Explanation. arXiv
preprint arXiv:1902.10186. Available at:
https://arxiv.org/abs/1902.10186. (Accessed: 13 March 2026).


<a id="page-35"></a>

## Page 35: Summary

![Original page 35](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_035.png)

### Extracted content

35 | © WMG 2026
Summary
We introduced Interpretability and Explainability concepts.
We discussed popular XAI approaches.
We discussed Ethical Key Points to be aware of.
We mentioned the Regulatory & Legal Landscape.


<a id="page-36"></a>

## Page 36: IV. From Deep Learning to

![Original page 36](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_036.png)

### Extracted content

36 | © WMG 2026
IV. From Deep Learning to
Production


<a id="page-37"></a>

## Page 37: The Reality Gap: Notebook → Production

![Original page 37](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_037.png)

### Extracted content

37 | © WMG 2026
The Reality Gap: Notebook → Production
"Only 15% of ML models ever make it into production" — Gartner, 2023
Jupyter Notebook
model.fit(X_train, y_train)
One GPU, local machine
Small curated dataset
Hardcoded hyperparams
Works on MY machine
No error handling
Global variables everywhere
Production Reality
Millions of concurrent requests
Multi-GPU / multi-node clusters
Streaming, dirty, shifting data
Auto-scaling cost constraints
Latency SLAs: < 50ms required
99.9% uptime requirements
Model drift & retraining loops
85% fail
Why Most Machine Learning Applications Fail To Deploy


<a id="page-38"></a>

## Page 38: Challenge 1 — Dependency & Environment Hell

![Original page 38](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_038.png)

### Extracted content

38 | © WMG 2026
Challenge 1 — Dependency & Environment Hell
Rule: If it's not in a Dockerfile, it doesn't exist in production.
Common Problems
CUDA/cuDNN version conflicts
PyTorch 2.1 needs CUDA 11.8, server has
11.6
Library version pinning
pip install . installs latest — breaks everything
Python environment drift
Works on dev, fails on CI/CD, crashes in
production
OS-level dependencies
libGL, libgomp missing in slim containers
Hardware-specific code
model.cuda() hardcoded, CPU server →
crash
Mitigation Strategies
Docker / Podman
Containerise
the entire runtime — CUDA base image + your
requirements.txt baked in. Reproducible across any host.
requirements.txt + pip-tools
Pin ALL dependencies with hashes. Use pip-compile to lock transitive
deps.
Conda / Mamba environments
Manage CUDA toolkit, Python, and ML libs in one environment spec.
Export to environment.yml.
device-agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") —
never hardcode.
Multi-stage Docker builds
Separate build stage (heavy tools) from runtime stage (lean image).
Reduces image from 12GB → 3GB.


<a id="page-39"></a>

## Page 39: Challenge 2 — Data Pipeline at Scale

![Original page 39](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_039.png)

### Extracted content

39 | © WMG 2026
Challenge 2 — Data Pipeline at Scale
Key Tools: Apache Spark · Dask · Airflow · Prefect · DVC · Great Expectations · Feast (Feature
Store)
Notebook Approach
df = pd.read_csv('data.csv')
Loads full dataset into RAM
Static, pre-cleaned file
Manual feature engineering
No versioning of data
Fails at 10GB+ files
Production Approach
Streaming + batched ingestion
Distributed storage (S3, GCS, ADLS)
Schema validation on ingest
Automated feature pipelines
Data versioning with DVC
Handles petabytes with Spark/Dask


<a id="page-40"></a>

## Page 40: Challenge 3 — Monitoring & Model Drift

![Original page 40](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_040.png)

### Extracted content

40 | © WMG 2026
Challenge 3 — Monitoring & Model Drift
Data Drift: Input feature distribution changes. P(X) shifts over time. E.g., user behaviour post-
pandemic.
Concept Drift: The relationship P(Y|X) changes. Same inputs, different correct outputs. E.g., an
economic model in recession.
Label Drift: Ground truth distribution P(Y) shifts. Class imbalance changes over time. E.g., fraud
pattern evolution.
Golden rule: If your model isn't monitored in production, it's already degrading.


<a id="page-41"></a>

## Page 41: Security & Compliance

![Original page 41](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_041.png)

### Extracted content

41 | © WMG 2026
Security & Compliance
Often Forgotten Until It's Too Late
Supply Chain Attacks
Risk: Malicious packages
in
requirements.txt.
Compromised pre-trained
weights
from
HuggingFace/GitHub.
Mitigation: Pin all deps
with hash verification, scan
with pip-audit/Safety, use
private model registries,
sign model artefacts.
Data Privacy (GDPR/UK
DPA)
Risk: PII in training data or
API inputs can violate data
protection law. Patient data
in healthcare models.
Mitigation:
Data
minimisation,
anonymisation
pipelines,
differential
privacy
(DP-
SGD), audit logs for all data
access.
Adversarial Inputs
Risk: Maliciously crafted
inputs fool the model.
Image
classifiers,
NLP
models
especially
vulnerable.
Mitigation:
Input
validation & sanitisation,
adversarial
training,
confidence
thresholding,
human-in-the-loop for low-
confidence cases.
Model Theft & Extraction
Risk: Adversaries query
your API to reconstruct the
model or steal IP.
Mitigation: Rate limiting,
API
authentication
(JWT/OAuth2),
query
logging,
output
perturbation
for
high-
value models.


<a id="page-42"></a>

## Page 42: V. Emerging Trends

![Original page 42](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_042.png)

### Extracted content

42 | © WMG 2026
V. Emerging Trends


<a id="page-43"></a>

## Page 43: Foundation Models & Multimodal AI

![Original page 43](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_043.png)

### Extracted content

43 | © WMG 2026
Foundation Models & Multimodal AI
The shift toward general-purpose, multi-capable AI systems
Foundation Models (One model, many tasks): Pre-trained on massive, diverse datasets, then
fine-tuned or prompted for specific tasks. A paradigm shift from task-specific models to
general-purpose systems that can reason, generate, and act across domains.
•
Vision-Language: Models like GPT-4o and Gemini understand images, text, and their
relationships simultaneously
•
Audio-Visual: Speech, music, and video understanding in unified architectures (e.g.,
Whisper, AudioPaLM)
•
Action & Robotics: Models that translate language instructions into physical actions (RT-2,
embodied AI agents)


<a id="page-44"></a>

## Page 44: Efficient Deep Learning & Agentic AI

![Original page 44](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_044.png)

### Extracted content

44 | © WMG 2026
Efficient Deep Learning & Agentic AI
Making DL Sustainable
•
Sparse Models: Activate only relevant
parameters per input – same capacity,
fraction of the compute.
•
Mixture of Experts (MoE): Route tokens to
specialised
sub-networks.
GPT-4
and
Mixtral use this architecture.
•
Neural Architecture Search: Automated
discovery of optimal architectures for
specific tasks and constraints.
•
Neuromorphic Computing: Brain-inspired
hardware with event-driven processing for
ultra-low power inference.
Autonomous Tool-Using AI
LLM-based agents that can plan, reason, use
tools,
and
execute
multi-step
workflows
autonomously.
•
Code Agents: Write, test, and debug code
autonomously (e.g., Claude Code, Devin).
•
Research Agents: Search, synthesise, and
report on complex topics end-to-end.
•
Orchestration:
LangChain,
LangGraph,
CrewAI for multi-agent coordination.


<a id="page-45"></a>

## Page 45: Summary

![Original page 45](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_045.png)

### Extracted content

45 | © WMG 2026
Summary
We introduced Emerging Trend applications.
We discussed different challenges related to DL model
production: environment dependencies, scalability, and
communication.


<a id="page-46"></a>

## Page 46: Do you

![Original page 46](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_046.png)

### Extracted content

46 | © WMG 2026
Do you
have any
questions?


<a id="page-47"></a>

## Page 47: warwick.ac.uk/wmg

![Original page 47](assets/1_Introduction_to_Deep_Learning_-_ML_vs_DL/page_047.png)

### Extracted content

47 | © WMG 2026
warwick.ac.uk/wmg
@wmgwarwick
wmgwarwick
wmg-university-of-warwick
WMG
University of Warwick
Coventry
CV4 7AL
United Kingdom
Thank you.
Leonardo.Alves-Dias@warwick.ac.uk
