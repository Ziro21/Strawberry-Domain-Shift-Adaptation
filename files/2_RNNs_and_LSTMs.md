# 2 RNNs and LSTMs

> Upgraded Markdown conversion. The original page image is included for every page so diagrams, screenshots, tables, slide layout and visual details are preserved. Extracted text is also provided below each page for searchable study notes.

## Contents

- [Page 1: Introduction to](#page-1)
- [Page 2: Agenda](#page-2)
- [Page 3: Learning Outcomes](#page-3)
- [Page 4: I. Recurrent Neural Networks](#page-4)
- [Page 5: Introduction](#page-5)
- [Page 6: A Recurrent Neuron](#page-6)
- [Page 7: The Fundamental Problem: Data with Memory](#page-7)
- [Page 8: Applications Across Domains](#page-8)
- [Page 9: Applications Across Domains](#page-9)
- [Page 10: RNNs: The Architecture](#page-10)
- [Page 11: RNN Unrolling](#page-11)
- [Page 12: The Architecture](#page-12)
- [Page 13: Detecting Names](#page-13)
- [Page 14: Tokenisation](#page-14)
- [Page 15: One-Hot Encoding](#page-15)
- [Page 16: The Architecture](#page-16)
- [Page 17: The Architecture](#page-17)
- [Page 18: The Architecture](#page-18)
- [Page 19: Forward Propagation](#page-19)
- [Page 20: Worked Example](#page-20)
- [Page 21: Step t = 1 — Input: 'H', Target: 'E'](#page-21)
- [Page 22: Step t = 1 — Input: 'H', Target: 'E'](#page-22)
- [Page 23: Gate Recurrent Unit (GRU)](#page-23)
- [Page 24: Backward Propagation](#page-24)
- [Page 25: Backpropagation Through Time (BPTT)](#page-25)
- [Page 26: Types of RNNs](#page-26)
- [Page 27: Types of RNNs](#page-27)
- [Page 28: Types of RNNs](#page-28)
- [Page 29: Types of RNNs](#page-29)
- [Page 30: Types of RNNs](#page-30)
- [Page 31: Language Modelling](#page-31)
- [Page 32: Language Modelling](#page-32)
- [Page 33: Sampling a Sequence From a Trained RNN](#page-33)
- [Page 34: Long-term dependencies](#page-34)
- [Page 35: Summary](#page-35)
- [Page 36: II. Long Short-Term Memory](#page-36)
- [Page 37: Introduction](#page-37)
- [Page 38: LSTM Applications: Natural Language Processing](#page-38)
- [Page 39: LSTM Applications: Speech & Audio](#page-39)
- [Page 40: LSTMs vs Transformers: Are LSTMs Dead?](#page-40)
- [Page 41: The xLSTM Revival (2024)](#page-41)
- [Page 42: LSTM: The Architecture](#page-42)
- [Page 43: The LSTM Cell: Big Picture](#page-43)
- [Page 44: LSTM Unit](#page-44)
- [Page 45: Gate 1: The Forget Gate](#page-45)
- [Page 46: Gate 2: The Input Gate + Update Gate](#page-46)
- [Page 47: Cell State Update](#page-47)
- [Page 48: Gate 3: The Output Gate](#page-48)
- [Page 49: Worked Example](#page-49)
- [Page 50: Worked Example](#page-50)
- [Page 51: Worked Example](#page-51)
- [Page 52: Summary](#page-52)
- [Page 53: Do you](#page-53)
- [Page 54: warwick.ac.uk/wmg](#page-54)

---


<a id="page-1"></a>

## Page 1: Introduction to

![Original page 1](assets/2_RNNs_and_LSTMs/page_001.png)

### Extracted content

1 | © WMG 2026
Artificial Intelligence & Deep learning
Introduction to
RNNs and LSTMs
Dr Leonardo Alves Dias
2026


<a id="page-2"></a>

## Page 2: Agenda

![Original page 2](assets/2_RNNs_and_LSTMs/page_002.png)

### Extracted content

2 | © WMG 2026
Agenda
I.
Recurrent Neural Networks (RNNs)
II.
Long Short-Term Memory (LSTM)


<a id="page-3"></a>

## Page 3: Learning Outcomes

![Original page 3](assets/2_RNNs_and_LSTMs/page_003.png)

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
Networks (ANNs)
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

## Page 4: I. Recurrent Neural Networks

![Original page 4](assets/2_RNNs_and_LSTMs/page_004.png)

### Extracted content

4 | © WMG 2026
I. Recurrent Neural Networks
(RNNs)


<a id="page-5"></a>

## Page 5: Introduction

![Original page 5](assets/2_RNNs_and_LSTMs/page_005.png)

### Extracted content

5 | © WMG 2026
Introduction
Recurrent Neural Networks (RNNs) are a class of neural networks designed to process
sequential data by retaining information from previous steps. They are especially effective for
tasks where context and order matter.
RNNs work by “remembering” past information: Imagine reading a sentence and trying to
predict the next word; you don’t rely only on the current word but also remember the words
that came before. An RNN works similarly: it considers all earlier words to choose the most
likely next word.
(GeeksforGeeks, 2026)
Introduction to Recurrent Neural Networks - GeeksforGeeks


<a id="page-6"></a>

## Page 6: A Recurrent Neuron

![Original page 6](assets/2_RNNs_and_LSTMs/page_006.png)

### Extracted content

6 | © WMG 2026
A Recurrent Neuron
The “memory” aspect of an RNN is achieved with a Recurrent Neuron.
It contains a hidden state (h) that maintains information about previous inputs in
a sequence by feeding back their hidden state.
Therefore, capturing dependencies across time.


<a id="page-7"></a>

## Page 7: The Fundamental Problem: Data with Memory

![Original page 7](assets/2_RNNs_and_LSTMs/page_007.png)

### Extracted content

7 | © WMG 2026
The Fundamental Problem: Data with Memory
Limitations of a FeedForward Network:
•
Fixed-size input only: A standard MLP takes a
vector of fixed length. A sentence of 5 words
and a sentence of 50 words cannot share the
same network without padding.
•
No notion of order: The model treats “Dog
bites man” and “Man bites dog” identically if
the words are encoded as a bag of words.
Word order — which carries almost all the
meaning — is invisible.
•
No temporal persistence: Each input is
processed
from
scratch.
There
is
no
mechanism to carry information from time-step
t to time-step t+1. The network has no working
memory.
What a sequence model should provide:
•
Variable-length input: Process sequences of
arbitrary length T without restructuring the
network. The same computation cell is applied
at each time step, regardless of where in the
sequence we are.
•
Order sensitivity: The output at time t must
depend on all previous inputs x₁, x₂, …, xₜ₋₁ in
the correct order — not just their aggregate.
The position in a sequence must be implicit in
the computation.
•
Persistent hidden state: A vector hₜ (the
hidden state) is updated at every time step and
passed forward. It acts as a fixed-size 'memory'
of everything the network has seen so far,
compressed into a learned representation.


<a id="page-8"></a>

## Page 8: Applications Across Domains

![Original page 8](assets/2_RNNs_and_LSTMs/page_008.png)

### Extracted content

8 | © WMG 2026
Applications Across Domains
Natural Language Processing
Examples:
Sentiment
analysis,
named-entity recognition, machine
translation, text classification.
Every word depends on prior
context. LSTMs were state-of-the-
art for NLP from ~2015–2018 and
remain embedded in Transformers.
Sutskever
et
al.
(2014)
—
Seq2Seq
Time-Series Forecasting
Examples:
Financial
markets,
electricity
demand,
weather
prediction,
server
load,
supply
chain.
Measurements
taken
at
t
are
causally related to t−1, t−2, …, t−k.
An RNN learns which past window
is relevant.
Salinas et al. (2020) — DeepAR
Speech & Audio
Examples:
Automatic
speech
recognition
(ASR),
speaker
diarisation, music generation, text-
to-speech.
Speech is a temporal acoustic
signal.
Phonemes
unfold
over
milliseconds; words over seconds.
Graves et al. (2013) — CTC + RNN


<a id="page-9"></a>

## Page 9: Applications Across Domains

![Original page 9](assets/2_RNNs_and_LSTMs/page_009.png)

### Extracted content

9 | © WMG 2026
Applications Across Domains
Why learn RNNs in 2025? Because Transformers ARE built on the same core insight, and
understanding RNNs is prerequisite knowledge for understanding self-attention.
Biomedical & Clinical Signals
Examples:
ECG
arrhythmia
detection, ICU early warning, EEG
seizure detection, patient trajectory
modelling.
Clinical
measurements
are
inherently sequential. An RNN can
detect
sudden
changes
in
a
patient's vitals.
Rajpurkar
et
al.
(2017)
—
Cardiologist-level ECG
Generative Sequence Models
Examples: Code completion, music
composition,
script
generation,
molecular sequence design (drug
discovery).
Character-level or token-level RNNs
can
generate
plausible
continuations of any sequence.
Karpathy (2015) — Unreasonable
Effectiveness
Robotics & Control
Examples:
Gesture
recognition,
trajectory prediction, reinforcement
learning with recurrent policies,
autonomous navigation.
A robot perceiving the world
through a camera or lidar stream
must maintain a model of what has
happened to predict what will
happen next.
Wierstra et al. (2007) — Recurrent
Policy Gradients


<a id="page-10"></a>

## Page 10: RNNs: The Architecture

![Original page 10](assets/2_RNNs_and_LSTMs/page_010.png)

### Extracted content

10 | © WMG 2026
RNNs: The Architecture


<a id="page-11"></a>

## Page 11: RNN Unrolling

![Original page 11](assets/2_RNNs_and_LSTMs/page_011.png)

### Extracted content

11 | © WMG 2026
RNN Unrolling
RNN unfolding or unrolling is the process of expanding the recurrent structure over time steps.
During unrolling, each step of the sequence is represented as a separate layer in a series illustrating
how information flows across each time step.
This unrolling enables backpropagation through time (BPTT), a learning process in which errors are
propagated across time steps to adjust the network’s weights, enhancing the RNN’s ability to learn
dependencies in sequential data.


<a id="page-12"></a>

## Page 12: The Architecture

![Original page 12](assets/2_RNNs_and_LSTMs/page_012.png)

### Extracted content

12 | © WMG 2026
The Architecture
ℎ= 𝜎(𝑈∙𝑋+ 𝑊∙ℎ𝑡−1 + 𝐵)
where,
h represents the current hidden state.
𝑈and 𝑊are weight matrices.
𝐵is the bias.


<a id="page-13"></a>

## Page 13: Detecting Names

![Original page 13](assets/2_RNNs_and_LSTMs/page_013.png)

### Extracted content

13 | © WMG 2026
Detecting Names
𝑋(𝑖)<𝑡>
𝑇𝑥
(𝑖)
𝑦(𝑖)<𝑡>
𝑇𝑦
(𝑖)


<a id="page-14"></a>

## Page 14: Tokenisation

![Original page 14](assets/2_RNNs_and_LSTMs/page_014.png)

### Extracted content

14 | © WMG 2026
Tokenisation
A Star Wars dictionary with
10000 words.


<a id="page-15"></a>

## Page 15: One-Hot Encoding

![Original page 15](assets/2_RNNs_and_LSTMs/page_015.png)

### Extracted content

15 | © WMG 2026
One-Hot Encoding


<a id="page-16"></a>

## Page 16: The Architecture

![Original page 16](assets/2_RNNs_and_LSTMs/page_016.png)

### Extracted content

16 | © WMG 2026
The Architecture


<a id="page-17"></a>

## Page 17: The Architecture

![Original page 17](assets/2_RNNs_and_LSTMs/page_017.png)

### Extracted content

17 | © WMG 2026
The Architecture


<a id="page-18"></a>

## Page 18: The Architecture

![Original page 18](assets/2_RNNs_and_LSTMs/page_018.png)

### Extracted content

18 | © WMG 2026
The Architecture


<a id="page-19"></a>

## Page 19: Forward Propagation

![Original page 19](assets/2_RNNs_and_LSTMs/page_019.png)

### Extracted content

19 | © WMG 2026
Forward Propagation
𝑎<1> = 𝑓(𝑊𝑎𝑎𝑎<0> + 𝑊𝑎𝑥𝑥<1> + 𝑏𝑎)
ො𝑦<1> = 𝑓(𝑊𝑦𝑎𝑎<1> + 𝑏𝑦)
𝑎<𝑡> = 𝑓𝑊𝑎𝑎𝑎<𝑡−1> + 𝑊𝑎𝑥𝑥<𝑡> + 𝑏𝑎
ො𝑦<𝑡> = 𝑓(𝑊𝑦𝑎𝑎<𝑡> + 𝑏𝑦)


<a id="page-20"></a>

## Page 20: Worked Example

![Original page 20](assets/2_RNNs_and_LSTMs/page_020.png)

### Extracted content

20 | © WMG 2026
Worked Example
Goal: Train a character-level language model to
predict the NEXT character.
Input: "H" → "E" → "L" (first 3 chars of HELLO)
Targets: Predict "E" after H, "L" after E, "L" after L
1. Data Preparation: One-hot Encoding —
Vocabulary V = 4 characters.
H
1
0
0
0
E
0
1
0
0
L
0
0
1
0
O
0
0
0
1
2. Network Dimensions:
input_dim (d) = 4.
hidden_states (nₕ) = 3.
Vocabulary (V) = 4.
3. Weight Matrices (initial values — random small
weights)
•
Wₓₕ (3 × 4) maps input (d=4) → hidden (nₕ=3)
•
Wₕₕ (3 × 3) maps hidden → hidden (memory)
•
Wₕᵧ (4 × 3) maps hidden → output (V=4)
Wₓₕ =
Wₕₕ =
Wₕᵧ =
h₀ = [0, 0, 0]
bₕ = [0, 0, 0]
bᵧ = [0, 0, 0, 0]


<a id="page-21"></a>

## Page 21: Step t = 1 — Input: 'H', Target: 'E'

![Original page 21](assets/2_RNNs_and_LSTMs/page_021.png)

### Extracted content

21 | © WMG 2026
Step t = 1 — Input: 'H', Target: 'E'
RNN Cell
t=1
‘H’
h₁ = tanh(Wₓₕx₁ + Wₕₕh₀ + bₕ)
ℎ0 = 0, 0, 0, 0
ℎ1 = [0.997
0.1974
0.2913]
x1
𝑊𝑥ℎ= [ 0.1, 0.2, 0.3 ,
0.2, −0.1, 0.1 ,
−0.1, 0.1, 0.2 ,
0.3, 0.2, −0.1 ]
× 𝑥1 =
1
0
0
0
+
𝑊ℎℎ= [ 0.5, 0.0, 0.1 ,
0.0, 0.4, 0.1 ,
0.1, 0.1, 0.3 ]
× ℎ0 =
0
0
0
0
+ 𝑏0 =
0
0
0
0
=
ℎ1 = 𝑡𝑎𝑛ℎ( 0.1, 0.2, 0.3 +
0, 0, 0, 0 +
0, 0, 0, 0 )
ℎ1 = 𝑡𝑎𝑛ℎ0.1, 0.2, 0.3
= [0.997, 0.1974,0.2913]
𝑊𝑥ℎ


<a id="page-22"></a>

## Page 22: Step t = 1 — Input: 'H', Target: 'E'

![Original page 22](assets/2_RNNs_and_LSTMs/page_022.png)

### Extracted content

22 | © WMG 2026
Step t = 1 — Input: 'H', Target: 'E'
RNN Cell
t=1
‘H’
ො𝑦<𝑡> = 𝑓(𝑊𝑦𝑥ℎ1 + 𝑏𝑦)
ℎ0 = 0, 0, 0, 0
ℎ1 = [0.997
0.1974
0.2913]
x1
𝑊𝑥𝑦= [ 0.2, 0.1, 0.3 ,
0.1, 0.3, 0.1 ,
0.3, 0.2, 0.1 ,
0.1, 0.1, 0.4 ]
× ℎ1 =
0.997
0.1974
0.2913
+
𝑧=
0.1271
0.0983
0.0985
0.1462
ො𝑦1
𝑊𝑥ℎ
ො𝑦1 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥𝑧=
0.2523
0.2452
0.2452
0.2572
𝑏1 =
0
0
0
0
=


<a id="page-23"></a>

## Page 23: Gate Recurrent Unit (GRU)

![Original page 23](assets/2_RNNs_and_LSTMs/page_023.png)

### Extracted content

23 | © WMG 2026
Gate Recurrent Unit (GRU)
Simplifying equations:
𝑎<𝑡> = 𝑓𝑊𝑎𝑎𝑎<𝑡−1> + 𝑊𝑎𝑥𝑥<𝑡> + 𝑏𝑎
𝑎<𝑡> = 𝑓𝑊𝑎[𝑎<𝑡−1>, 𝑥<𝑡>] + 𝑏𝑎
ො𝑦<𝑡> = 𝑓(𝑊𝑦𝑎𝑎<𝑡> + 𝑏𝑦)


<a id="page-24"></a>

## Page 24: Backward Propagation

![Original page 24](assets/2_RNNs_and_LSTMs/page_024.png)

### Extracted content

24 | © WMG 2026
Backward Propagation
Loss function
𝐿<𝑡> ො𝑦<𝑡>, 𝑦<𝑡> = −𝑦<𝑡> log ො𝑦<𝑡> −1 −𝑦<𝑡> log(1 −ො𝑦<𝑡>)
𝐿ො𝑦, 𝑦= ෍
𝑡=1
𝑇𝑥
𝐿<𝑡> ො𝑦<𝑡>, 𝑦<𝑡>
The gradients to adjust the weights are obtained
from the derivative of the loss function with
respect to the weight matrix W (𝑊𝑎𝑎, 𝑊𝑎𝑥, 𝑊𝑦𝑥, 𝑏𝑎).
This training process is called Backpropagation
Through Time (BPTT).


<a id="page-25"></a>

## Page 25: Backpropagation Through Time (BPTT)

![Original page 25](assets/2_RNNs_and_LSTMs/page_025.png)

### Extracted content

25 | © WMG 2026
Backpropagation Through Time (BPTT)
The Core Instability
Vanishing Gradient
Cause: Eigenvalues of Wₕₕ < 1. When multiplied
T times, the gradient shrinks exponentially, that is
≈ 0 after 30–50 steps.
Effect: Weights at early time steps receive zero
gradient. The network cannot learn long-range
dependencies, e.g., subject-verb agreement over
20+ tokens.
Example: 0.9^50 = 0.005. If the largest
eigenvalue of Wₕₕ is 0.9, after 50 steps the
gradient is 200× smaller.
Exploding Gradient
Cause: Eigenvalues of Wₕₕ > 1. The gradient
grows exponentially, which can lead to NaN
weights or numerical instability.
Effect: Loss suddenly spikes mid-training. Model
weights
become
±∞.
Training
diverges
completely.
Fix: Gradient Clipping — cap gradient norm at
threshold τ: f ← f × (τ / max(||f||, τ)). Typical τ = 1.0–
5.0.
Implemented
in
PyTorch
as
torch.nn.utils.clip_grad_norm_(). Fast, effective,
universally used.


<a id="page-26"></a>

## Page 26: Types of RNNs

![Original page 26](assets/2_RNNs_and_LSTMs/page_026.png)

### Extracted content

26 | © WMG 2026
Types of RNNs
Many to One
One to One
One to Many
Many to Many
Types of Recurrent Neural Networks (RNN) in Tensorflow - GeeksforGeeks


<a id="page-27"></a>

## Page 27: Types of RNNs

![Original page 27](assets/2_RNNs_and_LSTMs/page_027.png)

### Extracted content

27 | © WMG 2026
Types of RNNs


<a id="page-28"></a>

## Page 28: Types of RNNs

![Original page 28](assets/2_RNNs_and_LSTMs/page_028.png)

### Extracted content

28 | © WMG 2026
Types of RNNs


<a id="page-29"></a>

## Page 29: Types of RNNs

![Original page 29](assets/2_RNNs_and_LSTMs/page_029.png)

### Extracted content

29 | © WMG 2026
Types of RNNs
Binary Classification


<a id="page-30"></a>

## Page 30: Types of RNNs

![Original page 30](assets/2_RNNs_and_LSTMs/page_030.png)

### Extracted content

30 | © WMG 2026
Types of RNNs


<a id="page-31"></a>

## Page 31: Language Modelling

![Original page 31](assets/2_RNNs_and_LSTMs/page_031.png)

### Extracted content

31 | © WMG 2026
Language Modelling


<a id="page-32"></a>

## Page 32: Language Modelling

![Original page 32](assets/2_RNNs_and_LSTMs/page_032.png)

### Extracted content

32 | © WMG 2026
Language Modelling


<a id="page-33"></a>

## Page 33: Sampling a Sequence From a Trained RNN

![Original page 33](assets/2_RNNs_and_LSTMs/page_033.png)

### Extracted content

33 | © WMG 2026
Sampling a Sequence From a Trained RNN
Sampling a new random sequence of words from trained RNN


<a id="page-34"></a>

## Page 34: Long-term dependencies

![Original page 34](assets/2_RNNs_and_LSTMs/page_034.png)

### Extracted content

34 | © WMG 2026
Long-term dependencies


<a id="page-35"></a>

## Page 35: Summary

![Original page 35](assets/2_RNNs_and_LSTMs/page_035.png)

### Extracted content

35 | © WMG 2026
Summary
We Introduced the concepts of RNNs
We analysed the RNN architecture


<a id="page-36"></a>

## Page 36: II. Long Short-Term Memory

![Original page 36](assets/2_RNNs_and_LSTMs/page_036.png)

### Extracted content

36 | © WMG 2026
II. Long Short-Term Memory
(LSTM)


<a id="page-37"></a>

## Page 37: Introduction

![Original page 37](assets/2_RNNs_and_LSTMs/page_037.png)

### Extracted content

37 | © WMG 2026
Introduction
Long Short-Term Memory (LSTM) is a recurrent architecture specifically designed to capture long-
range dependencies in sequential data by controlling the flow of information through gates.
This makes it useful for:
•
Handles Long Term Dependencies: Remembers information for longer sequences.
•
Memory Cell: Stores and updates important information over time.
•
Better than RNN: Overcomes short-term memory limitations.
•
Applications: Used in language translation, speech recognition and time series forecasting.


<a id="page-38"></a>

## Page 38: LSTM Applications: Natural Language Processing

![Original page 38](assets/2_RNNs_and_LSTMs/page_038.png)

### Extracted content

38 | © WMG 2026
LSTM Applications: Natural Language Processing
LSTMs process language as a sequence of tokens, capturing context and dependencies across
words.
•
Sentiment Analysis: Classify reviews/tweets as positive, negative, or neutral. The LSTM reads
the full sentence to resolve ambiguity: "not bad at all" → positive.
•
Named Entity Recognition: Bidirectional LSTMs tag entities in text (person, location,
organisation). Context from both directions improves accuracy.
•
Machine Translation: Encoder-decoder LSTMs powered Google Translate (2016–2019). The
encoder compresses a source sentence; the decoder generates the target language.
•
Text Generation: Character-level or word-level LSTMs generate coherent text, scripts, and
code. Foundation for early language models before GPT.


<a id="page-39"></a>

## Page 39: LSTM Applications: Speech & Audio

![Original page 39](assets/2_RNNs_and_LSTMs/page_039.png)

### Extracted content

39 | © WMG 2026
LSTM Applications: Speech & Audio
Audio is a dense temporal signal — LSTMs process spectral features frame-by-frame.
Key references: Graves et al. (2013) Deep RNNs for ASR · Kalchbrenner et al. (2018) WaveRNN
· Hannun et al. (2014) DeepSpeech
Speech Recognition (ASR)
•
CTC-based models: LSTM + Connectionist
Temporal Classification for end-to-end
ASR
•
DeepSpeech
(Baidu):
Stacked
bidirectional LSTMs achieved near-human
accuracy
•
On-device ASR: LSTMs power offline
speech recognition on mobile devices
due to compact size
•
Still used in production at scale for voice
assistants and transcription services
Music & Audio Generation
•
Melody generation: LSTMs learn musical
structure (key, rhythm, phrasing) and
compose new sequences
•
Audio synthesis: WaveRNN uses recurrent
cells for high-quality neural vocoding
•
Music classification: Genre and mood
tagging from audio spectrograms
•
Sound
event
detection:
Identify
environmental sounds (sirens, alarms) in
continuous audio streams


<a id="page-40"></a>

## Page 40: LSTMs vs Transformers: Are LSTMs Dead?

![Original page 40](assets/2_RNNs_and_LSTMs/page_040.png)

### Extracted content

40 | © WMG 2026
LSTMs vs Transformers: Are LSTMs Dead?
When to choose an LSTM over a Transformer:
•
Small datasets.
•
Streaming/real-time inference.
•
Edge/mobile deployment.
•
Memory-constrained environment.
Criterion
LSTM
Transformer
Training parallelism
Sequential (slow)
Fully parallel (fast)
Long-range dependencies
Good (with gates)
Excellent (attention)
Data efficiency
Strong with small data
Needs large datasets
Inference (streaming)
Natural fit
Requires modifications
Memory footprint
Compact
O(n²) attention cost
Interpretability
Gate activations
Attention weights
Edge / mobile deployment
Lightweight
Often too large


<a id="page-41"></a>

## Page 41: The xLSTM Revival (2024)

![Original page 41](assets/2_RNNs_and_LSTMs/page_041.png)

### Extracted content

41 | © WMG 2026
The xLSTM Revival (2024)
LSTMs are making a comeback with modern enhancements that reduce the gap with Transformers.
Result: xLSTM matches or exceeds Transformer performance on language modelling benchmarks
while retaining the LSTM's strengths.
Scalar LSTM (sLSTM)
Exponential gating: replaces sigmoid gates
with exponential activation, which can be
sharper and provide more precise memory
control.
New
memory
mixing:
introduces
normalisation and stabilisation techniques
for numerical stability at scale.
Matrix LSTM (mLSTM)
Matrix-valued cell state: replaces vector cell
state with a matrix — dramatically increased
storage capacity.
Fully parallelisable: unlike traditional LSTMs,
mLSTM can be trained in parallel — matching
Transformer throughput.
[2405.04517] xLSTM: Extended Long Short-Term Memory


<a id="page-42"></a>

## Page 42: LSTM: The Architecture

![Original page 42](assets/2_RNNs_and_LSTMs/page_042.png)

### Extracted content

42 | © WMG 2026
LSTM: The Architecture


<a id="page-43"></a>

## Page 43: The LSTM Cell: Big Picture

![Original page 43](assets/2_RNNs_and_LSTMs/page_043.png)

### Extracted content

43 | © WMG 2026
The LSTM Cell: Big Picture
Four components work together to control what the cell remembers, forgets,
and outputs.


<a id="page-44"></a>

## Page 44: LSTM Unit

![Original page 44](assets/2_RNNs_and_LSTMs/page_044.png)

### Extracted content

44 | © WMG 2026
LSTM Unit
𝑓<𝑡> = 𝜎(𝑊𝑓 𝑎<𝑡−1>, 𝑥<𝑡> + 𝑏𝑓)
𝑖<𝑡> = 𝜎(𝑊𝑢 𝑎<𝑡−1>, 𝑥<𝑡> + 𝑏𝑢)
ǁ𝑐<𝑡> = tanh(𝑊𝑐𝑎<𝑡−1>, 𝑥<𝑡> + 𝑏𝑐)
𝑐<𝑡> = 𝑖<𝑡> ∗ǁ𝑐<𝑡> + 𝑓<𝑡> ∗𝑐<𝑡−1>
𝑜<𝑡> = 𝜎(𝑊𝑜 𝑎<𝑡−1>, 𝑥<𝑡> + 𝑏𝑜)
𝑎<𝑡> = 𝑜<𝑡> ∗tanh 𝑐<𝑡>
𝑐<𝑡−1>
𝑎<𝑡−1>
𝑐<𝑡>
𝑥<𝑡>
forget gate
update gate
tanh
output gate
⨁
𝑐<𝑡>
𝑎<𝑡>
𝑎<𝑡>
𝑎<𝑡>
tanh
softmax
ǁ𝑐<𝑡>
𝑜<𝑡>
𝑓<𝑡>
𝑖<𝑡>
𝑦<𝑡>
----
*
*
*
𝑐<0>
𝑎<0>
𝑐<1>
𝑥<1>
⨁
𝑎<1>
𝑎<1>
softmax
𝑦<1>
----
*
𝑐<1>
𝑎<1>
𝑥<2>
⨁
𝑎<2>
𝑎<2>
softmax
𝑦<2>
𝑐<2>
----
*
𝑐<2>
𝑎<2>
𝑥<3>
⨁
𝑎<3>
𝑎<3>
softmax
𝑦<3>
𝑐<3>
----
*


<a id="page-45"></a>

## Page 45: Gate 1: The Forget Gate

![Original page 45](assets/2_RNNs_and_LSTMs/page_045.png)

### Extracted content

45 | © WMG 2026
Gate 1: The Forget Gate
𝑓<𝑡> = 𝜎(𝑊𝑓 𝑎<𝑡−1>, 𝑥<𝑡> + 𝑏𝑓)
σ = sigmoid → output ∈ (0, 1)
0 = completely forget 1 = completely keep
How it works
1. Concatenate hₜ₋₁ and xₜ into a single vector
2. Multiply by weight matrix Wf and add bias bf
3. Apply sigmoid activation σ
4. Output fₜ: a vector of values between 0 and 1
5. Element-wise multiply fₜ ⊙ Cₜ₋₁ to selectively erase
The Forget Gate is the first stop for new information. Its primary purpose is to decide what
information is no longer useful and should be discarded from the cell state.
Intuitive Example: Language Model
The cell has stored: subject = "he"
New input: "She went to the store"
The forget gate learns to output ≈ 0
for the old subject pronoun, erasing it so
the new subject can be stored.


<a id="page-46"></a>

## Page 46: Gate 2: The Input Gate + Update Gate

![Original page 46](assets/2_RNNs_and_LSTMs/page_046.png)

### Extracted content

46 | © WMG 2026
Gate 2: The Input Gate + Update Gate
𝑖<𝑡> = 𝜎(𝑊𝑢 𝑎<𝑡−1>, 𝑥<𝑡> + 𝑏𝑢)
ǁ𝑐<𝑡> = tanh(𝑊𝑐𝑎<𝑡−1>, 𝑥<𝑡> + 𝑏𝑐)
Two-part process:
1.
The update gate (sigmoid) decides WHICH dimensions of the cell state to update. Output
∈ (0, 1) — a filter.
2.
The candidate (tanh) creates a vector of CANDIDATE values that COULD be added. Output
∈ (-1, 1) — new content.
Combined: iₜ ⊙ C̃ₜ = element-wise product enables only selected candidate values get added
to the cell state.
The Input Gate decides what new information is worth adding to the cell state. This is a two-
part process: 1. A sigmoid layer decides which values will be updated, and 2. A tanh layer
creates a vector of new candidate values ( ǁ𝑐<𝑡>) that could be added to the state.


<a id="page-47"></a>

## Page 47: Cell State Update

![Original page 47](assets/2_RNNs_and_LSTMs/page_047.png)

### Extracted content

47 | © WMG 2026
Cell State Update
𝑐<𝑡> = 𝑖<𝑡> ∗ǁ𝑐<𝑡> + 𝑓<𝑡> ∗𝑐<𝑡−1>
This additive update is why gradients can flow unchanged through time — solving the vanishing
gradient problem.
Step 1: Selectively Forget
𝑓<𝑡> ∗𝑐<𝑡−1>
Multiply each element of the
old cell state by the forget
gate output.
fₜ ≈ 0 → erase that dimension
fₜ ≈ 1 → keep that dimension
Step 2: Selectively Add
𝑖<𝑡> ∗ǁ𝑐<𝑡>
The input gate filters the
candidate values.
iₜ ≈ 0 → don't add this
dimension
iₜ ≈ 1 → add the candidate
value
Result: New Cₜ
𝑐<𝑡>
The updated cell state: old
memory
(partially
erased)
plus
new
information
(selectively added).


<a id="page-48"></a>

## Page 48: Gate 3: The Output Gate

![Original page 48](assets/2_RNNs_and_LSTMs/page_048.png)

### Extracted content

48 | © WMG 2026
Gate 3: The Output Gate
𝑜<𝑡> = 𝜎(𝑊𝑜 𝑎<𝑡−1>, 𝑥<𝑡> + 𝑏𝑜)
𝑎<𝑡> = 𝑜<𝑡> ∗tanh 𝑐<𝑡>
The output gate is a three-step process:
1.
The output gate oₜ (sigmoid) decides which parts of the cell state to output. Like the other
gates, it's a learned filter based on hₜ₋₁ and xₜ.
2.
The cell state Cₜ is passed through tanh (squashing to [-1, 1]) to normalise its values.
3.
The tanh output is multiplied element-wise by oₜ — producing hₜ, the hidden state that goes
to the next timestep AND can be used for predictions.
Key distinction: The cell state Cₜ (full memory) ≠ the hidden state hₜ (filtered output). The LSTM
keeps more than it shows.
The Output Gate determines what the next hidden state should be. While the Cell State contains
all the long-term information, the Hidden State (𝑎<𝑡>) is what the network actually "outputs" at
that specific moment.


<a id="page-49"></a>

## Page 49: Worked Example

![Original page 49](assets/2_RNNs_and_LSTMs/page_049.png)

### Extracted content

49 | © WMG 2026
Worked Example
Setup & Forget Gate — using simple numbers to trace the computation.
Setup (tiny 2D example)
Input size: 2, Hidden size: 2
xₜ = [1.0, 0.5] (current input)
hₜ₋₁ = [0.2, -0.1] (prev hidden state)
Cₜ₋₁ = [0.8, -0.3] (prev cell state)
[hₜ₋₁, xₜ] = [0.2, -0.1, 1.0, 0.5]
Forget Gate Calculation
Wf = [[0.3, 0.1, 0.5, 0.2],
[0.1, 0.4, 0.3, 0.1]]
bf = [0.1, -0.1]
Wf · [hₜ₋₁, xₜ] + bf
= [0.3(0.2)+0.1(-0.1)+0.5(1.0)+0.2(0.5)+0.1,
0.1(0.2)+0.4(-0.1)+0.3(1.0)+0.1(0.5)-0.1]
= [0.75, 0.27]
fₜ = σ([0.75, 0.27]) = [0.679, 0.567]
Interpretation & Apply to Cell State
fₜ = [0.679, 0.567] means: keep 67.9% of dimension 1, and 56.7% of dimension 2.
fₜ ⊙ Cₜ₋₁ = [0.679, 0.567] ⊙ [0.8, -0.3] = [0.543, -0.170]
The cell state after forgetting: some of the old information has been partially erased.


<a id="page-50"></a>

## Page 50: Worked Example

![Original page 50](assets/2_RNNs_and_LSTMs/page_050.png)

### Extracted content

50 | © WMG 2026
Worked Example
Setup & Forget Gate — using simple numbers to trace the computation.
Input Gate
Wi · [hₜ₋₁, xₜ] + bi = [0.62, 0.45]
iₜ = σ([0.62, 0.45]) = [0.650, 0.611]
Candidate Values
Wc · [hₜ₋₁, xₜ] + bc = [0.88, -0.52]
C̃ₜ = tanh([0.88, -0.52]) = [0.707, -0.480]
Cell State Update
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
= [0.543, -0.170] + [0.460, -0.293]
= [1.003, -0.463]
New information to add: iₜ ⊙ C̃ₜ = [0.650, 0.611] ⊙ [0.707, -0.480] = [0.460, -0.293]


<a id="page-51"></a>

## Page 51: Worked Example

![Original page 51](assets/2_RNNs_and_LSTMs/page_051.png)

### Extracted content

51 | © WMG 2026
Worked Example
Setup & Forget Gate — using simple numbers to trace the computation.
Complete Forward Pass Summary
xₜ = [1.0, 0.5] hₜ₋₁ = [0.2, -0.1] Cₜ₋₁ = [0.8, -0.3]
fₜ = [0.679, 0.567]
iₜ = [0.650, 0.611]
C̃ₜ = [0.707, -0.480]
Cₜ = [1.003, -0.463]
oₜ = [0.634, 0.574]
hₜ = [0.484, -0.248]
Output Gate & Hidden State
Wo · [hₜ₋₁, xₜ] + bo = [0.55, 0.30] → oₜ = σ([0.55, 0.30]) = [0.634, 0.574]
tanh(Cₜ) = tanh([1.003, -0.463]) = [0.763, -0.432]
hₜ = oₜ ⊙ tanh(Cₜ) = [0.634, 0.574] ⊙ [0.763, -0.432] = [0.484, -0.248]


<a id="page-52"></a>

## Page 52: Summary

![Original page 52](assets/2_RNNs_and_LSTMs/page_052.png)

### Extracted content

52 | © WMG 2026
Summary
We Introduced the concepts of LSTM
We analysed the LSTM architecture


<a id="page-53"></a>

## Page 53: Do you

![Original page 53](assets/2_RNNs_and_LSTMs/page_053.png)

### Extracted content

53 | © WMG 2026
Do you
have any
questions?


<a id="page-54"></a>

## Page 54: warwick.ac.uk/wmg

![Original page 54](assets/2_RNNs_and_LSTMs/page_054.png)

### Extracted content

54 | © WMG 2026
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
