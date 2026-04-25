# 1 Introduction to Transformers

> Upgraded Markdown conversion. The original page image is included for every page so diagrams, screenshots, tables, slide layout and visual details are preserved. Extracted text is also provided below each page for searchable study notes.

## Contents

- [Page 1: Transformers](#page-1)
- [Page 2: Agenda](#page-2)
- [Page 3: Learning Outcomes](#page-3)
- [Page 4: Introduction to](#page-4)
- [Page 5: Why transformers are so popular](#page-5)
- [Page 6: Transformers vs RNNs](#page-6)
- [Page 7: TITLE TO GO HERE](#page-7)
- [Page 8: The big picture: what happens to a sentence](#page-8)
- [Page 9: Tokenisation](#page-9)
- [Page 10: Input embeddings](#page-10)
- [Page 11: Embeddings capture meaning](#page-11)
- [Page 12: Positional encoding: why we need it](#page-12)
- [Page 13: Positional encoding](#page-13)
- [Page 14: The Encoder: understanding the input](#page-14)
- [Page 15: The Decoder: generating the output](#page-15)
- [Page 16: Summary](#page-16)
- [Page 17: II. Attention mechanisms](#page-17)
- [Page 18: Self-attention: the intuition](#page-18)
- [Page 19: Query, Key, Value: the database analogy](#page-19)
- [Page 20: Scaled dot-product attention](#page-20)
- [Page 21: Why divide by √d_k ?](#page-21)
- [Page 22: Attention is a matrix operation](#page-22)
- [Page 23: Multi-head attention](#page-23)
- [Page 24: Multi-head attention: visual](#page-24)
- [Page 25: What do different heads actually learn?](#page-25)
- [Page 26: Why multiple heads, not one big attention?](#page-26)
- [Page 27: Masked self-attention (decoder)](#page-27)
- [Page 28: Masked self-attention (decoder)](#page-28)
- [Page 29: Masked self-attention (decoder)](#page-29)
- [Page 30: Cross-attention (encoder-decoder)](#page-30)
- [Page 31: Cross-attention (encoder-decoder)](#page-31)
- [Page 32: Cross-attention: where it fits in the decoder](#page-32)
- [Page 33: Attention types: summary comparison](#page-33)
- [Page 34: TITLE TO GO HERE](#page-34)
- [Page 35: TITLE TO GO HERE](#page-35)
- [Page 36: TITLE TO GO HERE](#page-36)
- [Page 37: TITLE TO GO HERE](#page-37)
- [Page 38: Summary](#page-38)
- [Page 39: III. Hands-on Lab](#page-39)
- [Page 40: Decoding strategies: how to pick the next token](#page-40)
- [Page 41: Temperature: controlling randomness](#page-41)
- [Page 42: IV. Emerging Technologies &](#page-42)
- [Page 43: Scaling: Bigger = Smarter?](#page-43)
- [Page 44: Making Transformers Smaller & Faster](#page-44)
- [Page 45: Transformers Beyond Text](#page-45)
- [Page 46: Recent Achievements Powered by Transformers](#page-46)
- [Page 47: warwick.ac.uk/wmg](#page-47)

---


<a id="page-1"></a>

## Page 1: Transformers

![Original page 1](assets/1_Introduction_to_Transformers/page_001.png)

### Extracted content

1 | © WMG 2026
Artificial Intelligence & Deep Learning
Transformers
Gina Souvalioti
2026


<a id="page-2"></a>

## Page 2: Agenda

![Original page 2](assets/1_Introduction_to_Transformers/page_002.png)

### Extracted content

2 | © WMG 2026
Agenda
I.
Introduction to Transformers
II.
The Attention Mechanism (Self,
Multi-Head, Masked, Cross)
III. Hands-on Lab
IV. Emerging technologies and
Challenges


<a id="page-3"></a>

## Page 3: Learning Outcomes

![Original page 3](assets/1_Introduction_to_Transformers/page_003.png)

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

## Page 4: Introduction to

![Original page 4](assets/1_Introduction_to_Transformers/page_004.png)

### Extracted content

4 | © WMG 2026
I.
Introduction to
Transformers


<a id="page-5"></a>

## Page 5: Why transformers are so popular

![Original page 5](assets/1_Introduction_to_Transformers/page_005.png)

### Extracted content

5 | © WMG 2026
Why transformers are so popular
•
Power every major AI product: ChatGPT, Claude, Gemini, Copilot
•
One architecture, everything — text, images, audio, code, video,
protein folding
•
Scale and keep getting better — more data + compute = better
performance. No other architecture scales this consistently
•
One model, thousands of uses — a single pre-trained transformer
adapts to customer service, medical diagnosis, legal analysis, code
generation
•
Speed of adoption — research paper in 2017 → inside every major tech
product by 2023
•
Industry demand — architecture behind a multi-trillion dollar
technology wave


<a id="page-6"></a>

## Page 6: Transformers vs RNNs

![Original page 6](assets/1_Introduction_to_Transformers/page_006.png)

### Extracted content

6 | © WMG 2026
Transformers vs RNNs
RNN / LSTM
Transformer
How it reads
Word by word, left to right
Entire sequence at once
Long-range context
Fades with distance
Direct attention between any two tokens
Parallelisation
Each step waits for previous
All tokens processed simultaneously
Training speed
Slow on modern GPUs
Scales efficiently on GPUs/TPUs
Scalability
Performance plateaus
Keeps improving with data + compute
Dominant since
1990s – 2017
2017 – present


<a id="page-7"></a>

## Page 7: TITLE TO GO HERE

![Original page 7](assets/1_Introduction_to_Transformers/page_007.png)

### Extracted content

TITLE TO GO HERE
7 | © WMG 2026
Attention is all you need!
•
Published by Vaswani et al. at Google Brain,
NeurIPS 2017
•
Introduced the Transformer architecture —
replaced recurrence entirely with attention
•
Over 140,000 citations — one of the most cited
CS papers ever
•
Directly led to GPT-1 (2018), BERT (2018), and
every modern LLM
•
Key insight: self-attention alone is sufficient for
sequence modelling


<a id="page-8"></a>

## Page 8: The big picture: what happens to a sentence

![Original page 8](assets/1_Introduction_to_Transformers/page_008.png)

### Extracted content

8 | © WMG 2026
The big picture: what happens to a sentence
Raw
Text
Tokeni-
sation
Token
IDs
Embed-
ding
+Pos.
Encoding
Encoder
Blocks
Decoder
Blocks
Linear +
Softmax
Output
Token
We will zoom into each of these blocks in this lecture


<a id="page-9"></a>

## Page 9: Tokenisation

![Original page 9](assets/1_Introduction_to_Transformers/page_009.png)

### Extracted content

9 | © WMG 2026
Tokenisation
•
Splits raw text into smaller units called tokens
(words, sub-words, or characters)
•
Each token is mapped to a unique integer (token ID)
from the model's fixed vocabulary
•
Unknown words are split into sub-word tokens:
"unbelievable" → ["un", "believe", "able"]
•
These token IDs are what the model receives — it
never sees raw text
•
Common algorithms: BPE (GPT), WordPiece (BERT),
SentencePiece (T5)
Character level
"T h i s i s t o k e n i z i n g ."
Word level
"This" "is" "tokenizing" "."
Sub-word level (BPE)
"This" "is" "token" "izing" "."
OpenAI tokenizer: https://platform.openai.com/tokenizer


<a id="page-10"></a>

## Page 10: Input embeddings

![Original page 10](assets/1_Introduction_to_Transformers/page_010.png)

### Extracted content

10 | © WMG 2026
Input embeddings
Each token ID is looked up in an embedding matrix — a learned table of vectors, one
row per vocabulary token.
The result is a dense vector (e.g. 512 floats) representing the token in a continuous
space.
Token
ID
Embedding vector
"un"
4192
[0.21, -0.54, 0.88, ...]
"believe"
1679
[0.67, 0.12, -0.33, ...]
"able"
481
[-0.09, 0.74, 0.41, ...]
Similar tokens end up close together in vector space • Learned during training, not hand-crafted


<a id="page-11"></a>

## Page 11: Embeddings capture meaning

![Original page 11](assets/1_Introduction_to_Transformers/page_011.png)

### Extracted content

11 | © WMG 2026
Embeddings capture meaning
Dimension 1
Dim 2
Royalty
king
queen
prince
throne
Animals
cat
dog
fish
bird
Vehicles
car
bus
•
Each word is mapped to a point (vector) in a high-
dimensional space
•
Words with similar meanings cluster together —
"king" is near "queen", "cat" is near "dog"
•
Distances encode semantic relationships: the vector
from "man" to "woman" is parallel to "king" to
"queen"
•
These positions are learned during training — the
model discovers structure, not a human designer
•
Real spaces are 512–4096 dimensions; this figure
projects down to 2D for visualisation


<a id="page-12"></a>

## Page 12: Positional encoding: why we need it

![Original page 12](assets/1_Introduction_to_Transformers/page_012.png)

### Extracted content

12 | © WMG 2026
Positional encoding: why we need it
Problem: Transformers process all tokens in parallel — no built-in sense
of word order.
Without position info
"cat sat mat" = "mat sat cat"
Same embeddings, different meaning!
Solution: Add a position-dependent vector to each embedding before
feeding into the encoder.
Input = Embedding(token) + PE(position)
With position info
Token embedding + Position vector
Now each position is unique


<a id="page-13"></a>

## Page 13: Positional encoding

![Original page 13](assets/1_Introduction_to_Transformers/page_013.png)

### Extracted content

13 | © WMG 2026
Positional encoding
•
The original Transformer uses sinusoidal positional encodings. For even dimensions, we use sine
and for odd dimensions, cosine.
•
Each position gets its own unique pattern of sine/cosine values — think of it as a fingerprint for
that position in the sequence.
•
Low-dimension capture fine-grained differences between nearby positions (e.g. position 1 vs. 2).
•
High-dimension capture coarse structure across the whole sequence (e.g. early vs. late in a
sentence).
•
Adding this vector to the token embedding doesn't erase meaning — it enriches it. The token now
carries both what it is and where it is.
•
A key property: nearby positions produce similar fingerprints, so the model can naturally learn
that position 5 is close to position 6.


<a id="page-14"></a>

## Page 14: The Encoder: understanding the input

![Original page 14](assets/1_Introduction_to_Transformers/page_014.png)

### Extracted content

14 | © WMG 2026
The Encoder: understanding the input
• Input: positional-encoded token embeddings (the full input
sequence, all at once)
• What it does: each token attends to every other token to
build a rich, context-aware representation
• Output: a set of context-aware vectors — one per input
token — passed to the decoder
• Stacked N=6 times in the original paper — each layer
refines the representation further
Encoder Block (×N)
Multi-Head Self-Attention
Add & Layer Norm
Feed-Forward Network
Add & Layer Norm
⟲ residual
⟲ residual


<a id="page-15"></a>

## Page 15: The Decoder: generating the output

![Original page 15](assets/1_Introduction_to_Transformers/page_015.png)

### Extracted content

15 | © WMG 2026
The Decoder: generating the output
• Input: the output tokens generated so far (shifted right),
plus the encoder's output vectors
• What it does: attends to its own previous outputs, then
attends to the encoder output to link input context to
generation
• Output: one token at a time, auto-regressively — each
new token is fed back in as input for the next step
• Also stacked N=6 times, mirroring the encoder depth
• Final step: a linear layer + Softmax converts the output
into a probability distribution over the vocabulary to
predict the next token
Decoder Block (×N)
Masked Multi-Head
Self-Attention
Add & Layer Norm
Multi-Head Cross-Attention
(Q=decoder, K,V=encoder)
Add & Layer Norm
Feed-Forward Network
Add & Layer Norm


<a id="page-16"></a>

## Page 16: Summary

![Original page 16](assets/1_Introduction_to_Transformers/page_016.png)

### Extracted content

16 | © WMG 2026
Summary
"Attention is All You Need" (2017) — replaced recurrence with self-attention; foundation of all
modern LLMs
Positional Encoding — sin/cos vectors added to embeddings to preserve word order
Tokenisation & Embeddings — text split into tokens, each converted to a learned dense vector
Decoder Block — generates output one token at a time, auto-regressively, mirroring the encoder
structure but stacked N=6 times
Encoder Block — self-attention + feed-forward network, stacked 6 times, builds context-aware
representations


<a id="page-17"></a>

## Page 17: II. Attention mechanisms

![Original page 17](assets/1_Introduction_to_Transformers/page_017.png)

### Extracted content

17 | © WMG 2026
II. Attention mechanisms


<a id="page-18"></a>

## Page 18: Self-attention: the intuition

![Original page 18](assets/1_Introduction_to_Transformers/page_018.png)

### Extracted content

18 | © WMG 2026
Self-attention: the intuition
•
"The dog is sleeping on the table because it is not as cold as the floor."
•
What does 'it' refer to? → the table. How does the model know?
•
Self-attention lets every token look at every other token in the sequence
•
Each token asks: 'Who in this sentence is relevant to understanding me?'
•
The model learns to assign high attention weights to related tokens — regardless of
distance
•
This solves the long-range dependency problem that RNNs struggled with
•
Result: each token's representation becomes context-aware, informed by the entire
sequence


<a id="page-19"></a>

## Page 19: Query, Key, Value: the database analogy

![Original page 19](assets/1_Introduction_to_Transformers/page_019.png)

### Extracted content

19 | © WMG 2026
Query, Key, Value: the database analogy
Q
Query
What am I looking for?
The current token's question
to the sequence.
K
Key
What do I contain?
Each token's label or tag
— used for matching
against queries.
V
Value
What is my actual content?
The information that gets
retrieved once a match is
found.
Q, K, V are all derived from the same input embeddings via learned linear projections:
Q = XW_Q, K = XW_K, V = XW_V
Attention is not magic — it is a differentiable information retrieval system.


<a id="page-20"></a>

## Page 20: Scaled dot-product attention

![Original page 20](assets/1_Introduction_to_Transformers/page_020.png)

### Extracted content

20 | © WMG 2026
Scaled dot-product attention
Attention 𝑸, 𝑲, 𝑽= softmax 𝑸𝑲𝑻
𝒅𝒌
𝑽
Step 1
Q·K^T → compute similarity scores between every query and every key (dot product)
Step 2
/ √d_k → scale down by square root of key dimension to prevent softmax saturation
Step 3
softmax(·) → convert raw scores to a probability distribution (weights sum to 1)
Step 4
· V → weighted sum of value vectors — tokens with high relevance contribute more
Output: each token's representation is now context-aware, shaped by what it attended to.


<a id="page-21"></a>

## Page 21: Why divide by √d_k ?

![Original page 21](assets/1_Introduction_to_Transformers/page_021.png)

### Extracted content

21 | © WMG 2026
Why divide by √d_k ?
•
When d_k is large, the dot products Q·K^T grow
large in magnitude
•
Large values push softmax into regions with tiny
gradients (saturation)
•
Example: d_k = 512 → dot products ≈ O(√512) ≈
22.6
•
Dividing by √d_k ≈ 22.6 brings values back to a
reasonable range
•
This keeps the softmax output peaked enough to be
useful, but not so peaked that gradients vanish
•
Without scaling: model struggles to learn in early
training (all attention weights nearly uniform)
•
This is why it's called 'Scaled' Dot-Product
Attention
Line 3: QK^T dot product | Line 4: Scale | Line 6: Mask |
Line 7: Softmax | Line 8: Weighted sum
def attention(Q, K, V, mask=None):
d_k = Q.size(-1)
scores = torch.matmul(Q, K.transpose(-2, -1))
scores = scores / math.sqrt(d_k)
if mask is not None:
scores = scores.masked_fill(mask == 0, -1e9)
weights = F.softmax(scores, dim=-1)
output = torch.matmul(weights, V)
return output, weights


<a id="page-22"></a>

## Page 22: Attention is a matrix operation

![Original page 22](assets/1_Introduction_to_Transformers/page_022.png)

### Extracted content

22 | © WMG 2026
Attention is a matrix operation
For a sequence of n tokens with embedding dimension d:
X
(n × d)
input embeddings
W_Q, W_K, W_V
(d × d_k)
learned projections
Q = XW_Q
(n × d_k)
all queries stacked
K = XW_K
(n × d_k)
all keys stacked
V = XW_V
(n × d_k)
all values stacked
QK^T
(n × n)
attention score matrix
softmax(QK^T/√d_k)
(n × n)
attention weights
Output = Weights · V
(n × d_k)
context-aware embeddings
Everything is matrix multiplication — that's why transformers are so GPU-friendly.


<a id="page-23"></a>

## Page 23: Multi-head attention

![Original page 23](assets/1_Introduction_to_Transformers/page_023.png)

### Extracted content

23 | © WMG 2026
Multi-head attention
•
Instead of one attention function, run h parallel attention heads
•
Each head has its own learned projections: W_i^Q, W_i^K, W_i^V of dimension d_k = d_model / h

d_model = input embedding dimension, d_k= the projection dimension per attention head
•
Each head learns to attend to different types of relationships:
•
Head 1 might learn syntax • Head 2 might learn coreference • Head 3 might learn position
•
Outputs from all heads are concatenated → projected back via W_O to d_model
•
Original paper: h=8 heads, d_model=512, so d_k=64 per head
•
GPT-3: 96 heads
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O
where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)


<a id="page-24"></a>

## Page 24: Multi-head attention: visual

![Original page 24](assets/1_Introduction_to_Transformers/page_024.png)

### Extracted content

24 | © WMG 2026
Multi-head attention: visual
Input X
Head 1
Syntax
Head 2
Coreference
Head 3
Position
Head 4
Semantic
Concat → W_O projection → output (n × d_model)
Each head sees the full sequence but through different learned projections — like looking at a
sentence from multiple angles simultaneously.


<a id="page-25"></a>

## Page 25: What do different heads actually learn?

![Original page 25](assets/1_Introduction_to_Transformers/page_025.png)

### Extracted content

25 | © WMG 2026
What do different heads actually learn?
Researchers have probed trained transformers and found heads specialising in distinct linguistic roles:
Positional head
Attends to the previous or next token — learns local word order
"sat" → "cat" (previous word)
Syntactic head
Tracks grammatical structure — subject-verb, noun-adjective pairs
"dog" → "sleeping" (subject→verb)
Coreference head
Links pronouns to their antecedents across long distances
"it" → "table" (resolving pronoun)
Separator head
Attends to punctuation and special tokens — tracks sentence boundaries
"." → "[SEP]" (delimiter)
Rare token head
Focuses on unusual or informative words — named entities, numbers
"London" → high weight on itself
Source: Clark et al. (2019) 'What Does BERT Look At?' — analysis of 144 attention heads across 12 layers.


<a id="page-26"></a>

## Page 26: Why multiple heads, not one big attention?

![Original page 26](assets/1_Introduction_to_Transformers/page_026.png)

### Extracted content

26 | © WMG 2026
Why multiple heads, not one big attention?
Single head (d_k = 512)
•
One set of Q, K, V projections
•
Must encode ALL relationships into a single
attention pattern
•
Forced to compromise between syntax,
coreference,
and
semantics
in
one
distribution
•
Like having one eye that must focus on
everything at once
8 heads (d_k = 64 each)
•
8 independent attention patterns computed
in parallel
•
Head 1 can focus on syntax, Head 2 on
coreference, Head 3 on position, etc.
•
Each head has its own softmax — no
competition between relationship types
•
Concat + W_O mixes all perspectives into a
rich representation
•
Same parameter count! (8 × 64 × 512 × 3 =
512 × 512 × 3)


<a id="page-27"></a>

## Page 27: Masked self-attention (decoder)

![Original page 27](assets/1_Introduction_to_Transformers/page_027.png)

### Extracted content

27 | © WMG 2026
Masked self-attention (decoder)
Without mask (cheating)
•
When predicting "cat", the model can see
"sat" is next
•
Training loss drops to near zero — the
model just copies the answer
•
At inference, future tokens don't exist —
the model has never learned to predict
without them
•
Result: catastrophic failure at test time.
Generations are incoherent.
•
This is called "data leakage"
• The decoder generates tokens one at a time —
it must not see future tokens
• Without a mask: the model "cheats" during
training by copying future tokens → fails
completely at inference
• The mask enforces: "only look at what you've
already generated"


<a id="page-28"></a>

## Page 28: Masked self-attention (decoder)

![Original page 28](assets/1_Introduction_to_Transformers/page_028.png)

### Extracted content

28 | © WMG 2026
Masked self-attention (decoder)
Masked Attention = softmax( (Q·K^T + M) / √d_k ) · V
M is a mask matrix: upper-triangular positions set to −∞, which become 0 after softmax.
<s>
<s>
The
The
cat
cat
sat
sat
✓
−∞
−∞
−∞
✓
✓
−∞
−∞
✓
✓
✓
−∞
✓
✓
✓
✓
•
During training, the decoder must predict the next
token without seeing the future
•
When generating "cat", it can only attend to "<s>" and
"The"
•
Upper triangle of the attention matrix is masked with
−∞
•
After softmax: −∞ → 0 weight (completely blocked)
•
At inference: tokens are generated one at a time
anyway, but the mask is still applied during training for
efficiency


<a id="page-29"></a>

## Page 29: Masked self-attention (decoder)

![Original page 29](assets/1_Introduction_to_Transformers/page_029.png)

### Extracted content

29 | © WMG 2026
Masked self-attention (decoder)
• Output: a context-aware vector for each decoder token, capturing
everything generated so far
• This vector is not the final prediction — it still needs to look at the source
input
• Think of it as the decoder asking: "Given what I've said so far, what do I
need next?"
• This output becomes the Query (Q) in the next layer — cross-attention


<a id="page-30"></a>

## Page 30: Cross-attention (encoder-decoder)

![Original page 30](assets/1_Introduction_to_Transformers/page_030.png)

### Extracted content

30 | © WMG 2026
Cross-attention (encoder-decoder)
Key difference: Q comes from the decoder, but
K and V come from the encoder output.
Encoder output
(context vectors)
→ K, V
Decoder state
(current generation)
→ Q
• The encoder has already processed the full
input and produced a final output matrix —
one vector per input token
• The decoder does NOT receive raw Q, K, V from
the encoder — it
projects the encoder
output into K and V using learned weights
• Q comes from the
masked self-attention
output (what the decoder needs); K and V
come from the encoder output (what the
source offers)


<a id="page-31"></a>

## Page 31: Cross-attention (encoder-decoder)

![Original page 31](assets/1_Introduction_to_Transformers/page_031.png)

### Extracted content

31 | © WMG 2026
Cross-attention (encoder-decoder)
CrossAttention = softmax( Q_dec · K_enc^T / √d_k ) · V_enc
• Same scaled dot-product formula as self-attention — the only difference
is where Q, K, V come from
• Self-attention: Q, K, V from one sequence → Cross-attention: Q, K, V
from two sequences


<a id="page-32"></a>

## Page 32: Cross-attention: where it fits in the decoder

![Original page 32](assets/1_Introduction_to_Transformers/page_032.png)

### Extracted content

32 | © WMG 2026
Cross-attention: where it fits in the decoder
1. Masked self-attention
Decoder tokens attend to previous decoder tokens only
Q, K, V all from decoder (with causal mask)
Build context from what we've generated so far
2. Cross-attention
Decoder tokens attend to all encoder tokens
Q from decoder, K and V from encoder
Pull in information from the input sequence
3. Feed-forward network
Each token processed independently
No attention — just two linear layers + ReLU
Transform the combined representation non-linearly
These 3 sublayers repeat N times (N=6 in original). Each pass refines: first understand past output, then attend to
input, then transform.


<a id="page-33"></a>

## Page 33: Attention types: summary comparison

![Original page 33](assets/1_Introduction_to_Transformers/page_033.png)

### Extracted content

33 | © WMG 2026
Attention types: summary comparison
Self-Attention
Multi-Head
Masked Self
Cross-Attention
Q, K, V from
Same input
Same input
Same input
(decoder)
Q=decoder
K,V=encoder
Mask?
No
No
Yes (causal)
No
Heads
Single
h parallel
h parallel
h parallel
Where used
Encoder
Everywhere
Decoder
Decoder
Purpose
Build context
Multiple views
Prevent peeking
Link input
output


<a id="page-34"></a>

## Page 34: TITLE TO GO HERE

![Original page 34](assets/1_Introduction_to_Transformers/page_034.png)

### Extracted content

TITLE TO GO HERE
34 | © WMG 2026
Binding it all together - Encoder
• Input
Embedding
—
Raw
tokens
are
converted
into
dense
vectors
of
dimension d_model (e.g., 512). Each token
becomes a point in a continuous vector space
• Positional Encoding — A sinusoidal vector
is added to the embedding so the model
knows
token
order
(without
this,
the
transformer is order-blind)
• Output: A sequence of vectors, one per token,
carrying both meaning and position


<a id="page-35"></a>

## Page 35: TITLE TO GO HERE

![Original page 35](assets/1_Introduction_to_Transformers/page_035.png)

### Extracted content

TITLE TO GO HERE
35 | © WMG 2026
Binding it all together - Encoder
• Multi-Head Self-Attention — Each token attends
to all other tokens in the input. Input: Q, K, V all
come from the same sequence. Output: context-
enriched vectors of same shape
• Add & Norm — The original input is added back
(residual
connection)
and
layer-normalised.
Prevents vanishing gradients and stabilises training
• Feed-Forward Network — Two linear layers with
ReLU in between, applied independently to each
position. Adds representational capacity
• Add & Norm — Again: residual + normalisation
after the FFN
• Output of the encoder:
A rich contextual
representation of the
entire
input sequence
(passed to the cross attention of each of the N
stacked decoder layers)


<a id="page-36"></a>

## Page 36: TITLE TO GO HERE

![Original page 36](assets/1_Introduction_to_Transformers/page_036.png)

### Extracted content

TITLE TO GO HERE
36 | © WMG 2026
Binding it all together - Decoder
• Masked Multi-Head Self-Attention — The
decoder attends to its own previous outputs.
The mask prevents it from "seeing" future
tokens — ensures autoregressive generation
• Add & Norm — Residual + normalisation
• Cross-Attention
(Encoder–Decoder
Attention)
—
Q
comes
from
the
decoder, K and V come from the encoder
output. This is how the decoder "reads" the
source sequence at each generation step
• Add & Norm — Residual + normalisation
• Feed-Forward Network — Same as encoder:
position-wise transformation to combine all
gathered information
• Add & Norm — Final residual + normalisation


<a id="page-37"></a>

## Page 37: TITLE TO GO HERE

![Original page 37](assets/1_Introduction_to_Transformers/page_037.png)

### Extracted content

TITLE TO GO HERE
37 | © WMG 2026
Binding it all together - Decoder
• Linear Layer — Projects the decoder's output
vector (size d_model) up to the size of the full
vocabulary (e.g., 30,000 dimensions)
• Softmax
— Converts raw scores into a
probability distribution over all vocabulary
tokens
• Output Probabilities — The token with the
highest probability is selected as the next
generated word; the process then repeats
with this token fed back as the next decoder
input
• Key insight: The decoder generates one token
at a time, feeding each generated token back
in as input until an end-of-sequence token is
produced
•


<a id="page-38"></a>

## Page 38: Summary

![Original page 38](assets/1_Introduction_to_Transformers/page_038.png)

### Extracted content

38 | © WMG 2026
Summary
Self-attention computes Attention(Q,K,V) = softmax(QK^T/√d_k)V — every token attends to every
other
Masked attention blocks future tokens in the decoder (causal mask with −∞)
Multi-head attention runs h parallel heads, each learning different relationship types
Residual connections + layer norm stabilise deep stacks. FFN adds per-token non-linearity
Cross-attention connects encoder and decoder: Q from decoder, K and V from encoder


<a id="page-39"></a>

## Page 39: III. Hands-on Lab

![Original page 39](assets/1_Introduction_to_Transformers/page_039.png)

### Extracted content

39 | © WMG 2026
III. Hands-on Lab
(After break)


<a id="page-40"></a>

## Page 40: Decoding strategies: how to pick the next token

![Original page 40](assets/1_Introduction_to_Transformers/page_040.png)

### Extracted content

40 | © WMG 2026
Decoding strategies: how to pick the next token
After softmax, we have P(token | context) for all 50K+ tokens. How do we choose?
Greedy decoding
Always pick the highest-probability token: argmax P
✓ Deterministic, fast, no randomness
✗ Repetitive, boring, can get stuck in loops
Temperature sampling
Divide logits by T before softmax. T<1 = sharper, T>1 = flatter
✓ Controls creativity/randomness with a single knob
✗ Low-probability nonsense tokens can still be selected
Top-k sampling
Keep only the k highest-probability tokens, zero out the rest, re-normalise
✓ Eliminates low-probability tail
✗ Fixed k doesn't adapt — sometimes 5 tokens are good, sometimes 500
Top-p (nucleus) sampling Keep the smallest set of tokens whose cumulative probability ≥ p (e.g. p=0.9)
✓ Adapts to the distribution — narrow when confident, wide when uncertain
✗ Slightly more complex to implement
ChatGPT/Claude typically use top-p ≈ 0.9 with temperature ≈ 0.7. You can adjust these in the API.


<a id="page-41"></a>

## Page 41: Temperature: controlling randomness

![Original page 41](assets/1_Introduction_to_Transformers/page_041.png)

### Extracted content

41 | © WMG 2026
Temperature: controlling randomness
P(token) = softmax( logits / T )
T = 0.2 (sharp)
the
91%
a
5%
this
2%
my
1%
...
1%
Nearly deterministic.
Always picks 'the'.
Safe, repetitive.
T = 1.0 (neutral)
the
42%
a
21%
this
15%
my
12%
...
10%
Balanced.
Model's true distribution.
Default for most tasks.
T = 2.0 (flat)
the
24%
a
20%
this
19%
my
19%
...
18%
Nearly uniform.
Anything goes.
Creative but incoherent.


<a id="page-42"></a>

## Page 42: IV. Emerging Technologies &

![Original page 42](assets/1_Introduction_to_Transformers/page_042.png)

### Extracted content

42 | © WMG 2026
IV. Emerging Technologies &
Challenges


<a id="page-43"></a>

## Page 43: Scaling: Bigger = Smarter?

![Original page 43](assets/1_Introduction_to_Transformers/page_043.png)

### Extracted content

43 | © WMG 2026
Scaling: Bigger = Smarter?
The surprising discovery:
Make the model larger, feed it more data, use more compute — and it reliably gets better. No new
architecture needed.
Three things you can scale
•
Model size — more parameters (billions of numbers)
•
Data — more text, images, code fed during training
•
Compute — more GPUs and longer training runs
The key insight (Chinchilla, 2022)
•
Data and model size must grow together or you waste compute
The cost problem
•
Training GPT-4 ≈ $100M+ · one training run ≈ lifetime emissions of ~5 cars
Bottom line: scaling works — but it is expensive and we still don't fully understand why.


<a id="page-44"></a>

## Page 44: Making Transformers Smaller & Faster

![Original page 44](assets/1_Introduction_to_Transformers/page_044.png)

### Extracted content

44 | © WMG 2026
Making Transformers Smaller & Faster
The core problem: full-size transformers are too big to run on phones, cameras, or embedded chips.
Quantisation
Reduce how precisely weights are stored
Like rounding colours from RGB(123,45,200) to RGB(120,45,200) — harmless once,
damaging billions of times.
Pruning
Remove weights that contribute almost nothing
•
Large networks have many near-zero weights
•
Set them to exactly zero → smaller, faster model
•
Like removing rarely-used branches from a decision tree
•
Why accuracy still drops
•
Weights were tuned together as a system — removing any of them
disturbs the balance
•
Solutions being developed
•
Quantisation-aware training (QAT) — train with low precision from the
start
•
Structured pruning — remove whole attention heads, not random
weights
The gap is closing — but not yet closed, especially at aggressive compression for embedded/real-time systems.


<a id="page-45"></a>

## Page 45: Transformers Beyond Text

![Original page 45](assets/1_Introduction_to_Transformers/page_045.png)

### Extracted content

45 | © WMG 2026
Transformers Beyond Text
The same architecture that processes words can process almost anything, as long as we can
turn it into a sequence of tokens.


<a id="page-46"></a>

## Page 46: Recent Achievements Powered by Transformers

![Original page 46](assets/1_Introduction_to_Transformers/page_046.png)

### Extracted content

46 | © WMG 2026
Recent Achievements Powered by Transformers
AlphaFold 2 & 3
DeepMind, 2021–2024
Predicted the 3D structure of virtually every known protein. Decades-old biology
problem solved in months. Accelerating drug discovery worldwide.
Claude (Anthropic)
2023–2025
Conversational AI assistant used for writing, coding, research and analysis. Trained with
Constitutional AI to be safer and more honest.
Mistral & Mixtral
Mistral AI, 2023–2024
Powerful open-weight models from a French startup. Mixtral uses a 'mixture of
experts' approach — only activates part of the model per token, saving compute.
GitHub Copilot
GitHub / OpenAI, 2021–now
Autocompletes code in real time inside editors. Used by millions of developers daily.
Estimated to write ~46% of new code in supported languages.
Whisper
OpenAI, 2022
Near-human speech recognition in 99 languages. Open source. Powers transcription in
thousands of apps, including real-time captioning tools.
DALL·E 3 / Stable Diffusion
OpenAI / Stability AI, 2023
Generate photorealistic images from text descriptions. Transformers handle the text
understanding; diffusion models handle image generation.


<a id="page-47"></a>

## Page 47: warwick.ac.uk/wmg

![Original page 47](assets/1_Introduction_to_Transformers/page_047.png)

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
gina.souvalioti@warwick.ac.uk
