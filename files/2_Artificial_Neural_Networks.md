# 2 Artificial Neural Networks

> Upgraded Markdown conversion. The original page image is included for every page so diagrams, screenshots, tables, slide layout and visual details are preserved. Extracted text is also provided below each page for searchable study notes.

## Contents

- [Page 1: Introduction to](#page-1)
- [Page 2: Agenda](#page-2)
- [Page 3: Learning Outcomes](#page-3)
- [Page 4: I. The Perceptron](#page-4)
- [Page 5: Biological Inspiration](#page-5)
- [Page 6: Rosenblatt’s Perceptron (1958)](#page-6)
- [Page 7: Mathematical Formulation](#page-7)
- [Page 8: Pioneers of Neural Network Research](#page-8)
- [Page 9: The Step Activation Function](#page-9)
- [Page 10: The Perceptron as a Linear Classifier](#page-10)
- [Page 11: The Perceptron Learning Algorithm](#page-11)
- [Page 12: Worked Example: Learning the AND Gate](#page-12)
- [Page 13: Worked Example: Learning the OR Gate](#page-13)
- [Page 14: Worked Example: Learning the XOR Gate](#page-14)
- [Page 15: Think–Pair–Share](#page-15)
- [Page 16: Why Move Beyond the Perceptron?](#page-16)
- [Page 17: II. Gradient Descent](#page-17)
- [Page 18: Why Do We Need Gradient Descent?](#page-18)
- [Page 19: The Loss Function: Measuring Error](#page-19)
- [Page 20: The Loss Landscape: An Intuition](#page-20)
- [Page 21: The Gradient: Direction and Magnitude](#page-21)
- [Page 22: The Gradient Descent Algorithm](#page-22)
- [Page 23: The Learning Rate α](#page-23)
- [Page 24: Effects of Learning Rate Choice](#page-24)
- [Page 25: Differentiable Activation Functions](#page-25)
- [Page 26: Variants of Gradient Descent](#page-26)
- [Page 27: Challenges: Local Minima & Saddle Points](#page-27)
- [Page 28: Modern Optimisers: Beyond Vanilla GD](#page-28)
- [Page 29: Modern Optimisers: Beyond Vanilla GD](#page-29)
- [Page 30: Worked Example: 1D Gradient Descent](#page-30)
- [Page 31: Summary](#page-31)
- [Page 32: III. Artificial Neural](#page-32)
- [Page 33: From Optimisation to Architecture](#page-33)
- [Page 34: The Neuron: A Computational Unit](#page-34)
- [Page 35: The Neuron: Adding the Bias Term](#page-35)
- [Page 36: From Neuron to Network](#page-36)
- [Page 37: From Perceptron to Multi-Layer Networks](#page-37)
- [Page 38: The Multi-Layer Perceptron (MLP)](#page-38)
- [Page 39: Weight Notation & Biases per Layer](#page-39)
- [Page 40: Enumerating Weights (Layer 1 → 2)](#page-40)
- [Page 41: Enumerating Weights (Layer 2 → 3)](#page-41)
- [Page 42: Counting Parameters](#page-42)
- [Page 43: Forward Propagation: Layer 1 → 3](#page-43)
- [Page 44: Worked Example: Tracing a Forward Pass](#page-44)
- [Page 45: Think–Pair–Share](#page-45)
- [Page 46: Vectorisation](#page-46)
- [Page 47: Why Vectorisation?](#page-47)
- [Page 48: Neural Network Vectorisation](#page-48)
- [Page 49: Neural Network Vectorisation](#page-49)
- [Page 50: Forward Propagation: Data Through the Network](#page-50)
- [Page 51: Forward Propagation: Data Through the Network](#page-51)
- [Page 52: Forward Propagation: Data Through the Network](#page-52)
- [Page 53: Looping through the Layers](#page-53)
- [Page 54: Why We Need Non-linear Models](#page-54)
- [Page 55: Learn Useful Representations Automatically](#page-55)
- [Page 56: Designing an ANN: Practical Choices](#page-56)
- [Page 57: Overfitting and Regularisation](#page-57)
- [Page 58: Overfitting and Regularisation](#page-58)
- [Page 59: The Complete Training Pipeline](#page-59)
- [Page 60: Common Pitfalls and Diagnosis](#page-60)
- [Page 61: Building ANNs Responsibly](#page-61)
- [Page 62: Summary](#page-62)
- [Page 63: Do you](#page-63)
- [Page 64: warwick.ac.uk/wmg](#page-64)

---


<a id="page-1"></a>

## Page 1: Introduction to

![Original page 1](assets/2_Artificial_Neural_Networks/page_001.png)

### Extracted content

1 | © WMG 2026
Artificial Intelligence & Deep learning
Introduction to
Artificial Neural
Networks (ANNs)
Dr Leonardo Alves Dias
2026


<a id="page-2"></a>

## Page 2: Agenda

![Original page 2](assets/2_Artificial_Neural_Networks/page_002.png)

### Extracted content

2 | © WMG 2026
Agenda
I.
The Perceptron
II.
Gradient Descent
III. Artificial Neural Networks (ANNs)


<a id="page-3"></a>

## Page 3: Learning Outcomes

![Original page 3](assets/2_Artificial_Neural_Networks/page_003.png)

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

## Page 4: I. The Perceptron

![Original page 4](assets/2_Artificial_Neural_Networks/page_004.png)

### Extracted content

4 | © WMG 2026
I. The Perceptron


<a id="page-5"></a>

## Page 5: Biological Inspiration

![Original page 5](assets/2_Artificial_Neural_Networks/page_005.png)

### Extracted content

5 | © WMG 2026
Biological Inspiration
A threshold function
+1, 𝑖𝑓 𝑤0 + x𝑇𝑊> 0
−1, 𝑖𝑓 𝑤0 + x𝑇𝑊≤0
𝑧= 𝑤0 + x𝑇𝑊
Biological Neuron
•
Dendrites receive electrical signals from other neurons.
•
Cell Body (Soma) integrates and processes incoming signals.
•
Axon transmits the output signal to other neurons.
•
Synapse connection point; strength varies (learning).
•
Firing threshold – neuron activates only if the signal exceeds a threshold.
Artificial Neuron
•
Inputs (xᵢ) receive data values from features or other neurons.
•
Weights (wᵢ) scale each input (analogous to synaptic strength).
•
Summation (Σ) computes the weighted sum of all inputs.
•
Bias (b) shifts the decision boundary independently.
•
Activation function – determines whether the neuron “fires”, i.e., is active.
What is Perceptron - GeeksforGeeks


<a id="page-6"></a>

## Page 6: Rosenblatt’s Perceptron (1958)

![Original page 6](assets/2_Artificial_Neural_Networks/page_006.png)

### Extracted content

6 | © WMG 2026
Rosenblatt’s Perceptron (1958)
The perceptron was introduced as an algorithm for binary
classification.
It can be thought of as a simple neural network with a
threshold activation.
Key Properties:
•
Single Layer: Only one layer of computation inputs
directly to the output.
•
Binary Classifier: Output is classified into two
categories (e.g., 0 or 1).
•
Supervised: Learns from labelled examples using an
error signal.
•
Linear: Can only learn linearly separable decision
boundaries.
{0,1}
A threshold function
1, 𝑖𝑓 𝑤0 + x𝑇𝑊≥0
0, 𝑖𝑓 𝑤0 + x𝑇𝑊< 0
𝑧= 𝑤0 + x𝑇𝑊


<a id="page-7"></a>

## Page 7: Mathematical Formulation

![Original page 7](assets/2_Artificial_Neural_Networks/page_007.png)

### Extracted content

7 | © WMG 2026
Mathematical Formulation
Breaking down the perceptron computation step by step:
1.
Compute the Weighted Sum: Each input xi is
multiplied by its corresponding weight wi, then all
products are summed together with a bias term 𝑏.
z = w₁x₁ + w₂x₂ + … + wₙxₙ + b = Σ wᵢxᵢ + b
2.
Apply the Activation Function: The step function is
commonly used, and it outputs 1 when the weighted
sum exceeds the zero threshold, and 0 otherwise.
𝒶 = f(z) = { 1 if z ≥0, 0 if z < 0 }
3.
Produce the Binary Output: The output 𝒶 ∈ {0, 1}
represents the perceptron’s binary classification
decision.
A threshold function
1, 𝑖𝑓 𝑤0 + x𝑇𝑊≥0
0, 𝑖𝑓 𝑤0 + x𝑇𝑊< 0
𝑧= 𝑤0 + x𝑇𝑊
{0,1}
b


<a id="page-8"></a>

## Page 8: Pioneers of Neural Network Research

![Original page 8](assets/2_Artificial_Neural_Networks/page_008.png)

### Extracted content

8 | © WMG 2026
Pioneers of Neural Network Research
The ideas behind neural networks span decades and continents:
• McCulloch & Pitts (1943) — Proposed the first mathematical model of a neuron, laying the
theoretical foundation for all neural computation.
• Frank Rosenblatt (1958) — Developed the Perceptron, the first trainable neural network model.
• Kunihiko Fukushima (1979) — Invented the Neocognitron in Japan, a hierarchical, multi-layered
network that directly inspired modern Convolutional Neural Networks.
• Geoffrey Hinton, Yann LeCun & Yoshua Bengio — The "Godfathers of Deep Learning" who
persisted through two AI winters to develop backpropagation, CNNs, and deep belief networks.
• Fei-Fei Li — Created ImageNet, the large-scale dataset that catalysed the deep learning revolution
and demonstrated the critical role of data in AI.
Neural network research is a truly global endeavour, with transformative contributions from
researchers across every continent.


<a id="page-9"></a>

## Page 9: The Step Activation Function

![Original page 9](assets/2_Artificial_Neural_Networks/page_009.png)

### Extracted content

9 | © WMG 2026
The Step Activation Function
The original on/off switch of the perceptron.
Properties
•
Binary Output: Produces exactly 0 or 1, no intermediate
values.
•
Non-Differentiable: The discontinuity at z = 0 means we
cannot compute a gradient, which is a critical limitation.
That is, for z=3, the output is 1, and changing z to 3.1 still
yields 1. The derivative is zero, so the gradient tells the
model that "changing this weight had zero effect on the
output," and the model stops learning entirely.
•
Threshold Logic: Mimics biological all-or-nothing firing:
the neuron is either ON or OFF.
•
Historical Role: Adequate for the original perceptron but
replaced by sigmoid, ReLU, etc., in modern networks.
-0.2
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
-5
-4
-3
-2
-1
0
1
2
3
4
5
a
z


<a id="page-10"></a>

## Page 10: The Perceptron as a Linear Classifier

![Original page 10](assets/2_Artificial_Neural_Networks/page_010.png)

### Extracted content

10 | © WMG 2026
The Perceptron as a Linear Classifier
Decision Boundary
The perceptron’s equation
𝑤1𝑥1 + 𝑤2𝑥2 + 𝑏 = 0
defines a line (in 2D) or hyperplane (in higher
dimensions) that divides the input space into
two regions.
Above the line: z ≥ 0 → y = 1 (Class 1).
Below the line: z < 0 → y = 0 (Class 0).
Weights control the orientation; bias controls
the position of the boundary.
w₁x₁ + w₂x₂ + b = 0


<a id="page-11"></a>

## Page 11: The Perceptron Learning Algorithm

![Original page 11](assets/2_Artificial_Neural_Networks/page_011.png)

### Extracted content

11 | © WMG 2026
The Perceptron Learning Algorithm
How the perceptron learns to classify by adjusting its weights
1.
Initialise: Set all weights 𝑤ᵢ and bias 𝑏 to small random values
(or zeros).
2.
Forward Pass: For each training sample, compute the
predicted output: ŷ = 𝑓(Σ 𝑤ᵢ𝑥ᵢ + 𝑏)
3.
Compute Error (𝒆): Calculate the difference between the true
label and prediction: 𝑒 = 𝑦 – ŷ
4.
Update Weights: Adjust each weight: 𝑤ᵢ ← 𝑤ᵢ + 𝛼· 𝑒 · 𝑥ᵢ
and bias: 𝑏 ← 𝑏 + 𝛼· 𝑒
5.
Repeat: Iterate over all training samples (epochs) until
convergence or maximum iterations reached.
Inputs:
Training set 𝐷=
𝐱1 𝑦1
𝐱2 𝑦2
…
𝐱𝑁𝑦𝑁
Learning rate 𝛼
Maximum number of epochs: max_epochs
Output:
Weight vector 𝐰
Bias 𝑏
1. Initialise all weights 𝑤𝑖and bias 𝑏to 0 (or to
small random values)
2. For epoch = 1 to max_epochs:
errors ←0
For each training sample (x, y) in D:
net ←Σ (𝑤𝑖× 𝑥𝑖) + 𝑏
If net ≥ 0 then
ŷ ←1
Else
ŷ ←0
End If
If y ≠ ŷ then
For each weight w_i:
𝑤𝑖←𝑤𝑖+ 𝛼∗(𝑦−ŷ) ∗𝑥𝑖
End For
𝑏←𝑏+ 𝛼∗(𝑦−ŷ)
errors ←errors + 1
End If
End For
If errors = 0 then
Stop // convergence achieved
End If
3. Return 𝒘and 𝑏


<a id="page-12"></a>

## Page 12: Worked Example: Learning the AND Gate

![Original page 12](assets/2_Artificial_Neural_Networks/page_012.png)

### Extracted content

12 | © WMG 2026
Worked Example: Learning the AND Gate
The perceptron can learn simple logical functions. E.g.,
AND Truth Table
Learned Parameters:
w₁ = 1, w₂ = 1, b = –1.5
Therefore:
z = (1*0) + (1*0) – 1.5 = -1.5 < 0 → y = 0
z = (1*0) + (1*1) – 1.5 = -0.5 < 0 → y = 0
z = (1*1) + (1*0) – 1.5 = -0.5 < 0 → y = 0
z = (1*1) + (1*1) – 1.5 = 0.5 ≥ 0 → y = 1
x₁
x₂
y
0
0
0
0
1
0
1
0
0
1
1
1


<a id="page-13"></a>

## Page 13: Worked Example: Learning the OR Gate

![Original page 13](assets/2_Artificial_Neural_Networks/page_013.png)

### Extracted content

13 | © WMG 2026
Worked Example: Learning the OR Gate
The perceptron can learn simple logical functions.
OR Truth Table
Learned Parameters:
w₁ = 1, w₂ = 1, b = –0.5
Therefore:
z = (1*0) + (1*0) – 0.5 = -0.5 < 0 → y = 0
z = (1*0) + (1*1) – 0.5 = 0.5 ≥ 0 → y = 1
z = (1*1) + (1*0) – 0.5 = 0.5 ≥ 0 → y = 1
z = (1*1) + (1*1) – 0.5 = 1.5 ≥ 0 → y = 1
x₁
x₂
y
0
0
0
0
1
1
1
0
1
1
1
1


<a id="page-14"></a>

## Page 14: Worked Example: Learning the XOR Gate

![Original page 14](assets/2_Artificial_Neural_Networks/page_014.png)

### Extracted content

14 | © WMG 2026
Worked Example: Learning the XOR Gate
Note that the positive points are on opposite
corners, meaning no single straight line can separate
them.
The Solution: Multiple Layers!
XOR requires two decision boundaries working
together.
This means we need:
•
Hidden Layer: An intermediate layer of neurons
that
creates
new,
non-linear
feature
representations.
•
Multiple Neurons: Each hidden neuron learns a
different linear boundary; together they solve
non-linear problems.


<a id="page-15"></a>

## Page 15: Think–Pair–Share

![Original page 15](assets/2_Artificial_Neural_Networks/page_015.png)

### Extracted content

15 | © WMG 2026
Think–Pair–Share
Take 2 minutes with your peer:
We saw that XOR cannot be solved by a single perceptron because no single straight line
separates the positive and negative points.
1. Sketch how two linear decision boundaries could work together to correctly classify all four XOR
inputs.
2. What role does each hidden neuron play in creating these boundaries?
3. Can you think of another real-world problem that requires more than one linear boundary?
Be ready to share your sketch with the class!


<a id="page-16"></a>

## Page 16: Why Move Beyond the Perceptron?

![Original page 16](assets/2_Artificial_Neural_Networks/page_016.png)

### Extracted content

16 | © WMG 2026
Why Move Beyond the Perceptron?
The perceptron gives us a single-layer, binary classifier — but it has critical limitations:
•
It can only learn linearly separable problems (XOR fails).
•
The step activation function is non-differentiable, so we cannot compute gradients.
•
There is no principled way to extend the learning rule to multiple layers.
To overcome these, we need a smooth optimisation method that can guide learning through
differentiable functions.
Next: Gradient Descent — the engine that will power multi-layer networks.


<a id="page-17"></a>

## Page 17: II. Gradient Descent

![Original page 17](assets/2_Artificial_Neural_Networks/page_017.png)

### Extracted content

17 | © WMG 2026
II. Gradient Descent


<a id="page-18"></a>

## Page 18: Why Do We Need Gradient Descent?

![Original page 18](assets/2_Artificial_Neural_Networks/page_018.png)

### Extracted content

18 | © WMG 2026
Why Do We Need Gradient Descent?
The problem we left off with: The perceptron learning rule works by flipping weights when
predictions are wrong, but it relies on the step function, which is non-differentiable. This means:
(1) We cannot measure how wrong a prediction is – only that it is wrong;
(2) We cannot compute the direction to adjust weights smoothly;
(3) There is no principled way to extend learning to multiple layers. Therefore, we need a
method that uses smooth, differentiable functions to guide learning.
What we need from an optimisation method:
A way to measure error
Continuously
Not just “right or wrong” but a
smooth numerical score that
tells us exactly how far off each
prediction is. This is the loss
function.
A direction to move
Weights
At every point in the parameter
space, we need to know: should
each
weight
increase
or
decrease, and by how much?
This is the gradient, i.e., the
slope
or
"steepness"
of
a
function.
A rule that works
across layers
The update rule must propagate
error
information
backwards
through an arbitrary number of
layers.
This
is
known
as
backpropagation.


<a id="page-19"></a>

## Page 19: The Loss Function: Measuring Error

![Original page 19](assets/2_Artificial_Neural_Networks/page_019.png)

### Extracted content

19 | © WMG 2026
The Loss Function: Measuring Error
A single number that quantifies how wrong our model’s predictions are
Core Idea: The loss function 𝐿(𝑤, 𝑏) takes the model’s parameters (weights and biases) as input and
returns a scalar value indicating total prediction error across all training samples. Training a neural
network means finding the parameter values that minimise this function. The loss function must be
differentiable so we can compute which direction to adjust each parameter.
Two fundamental loss functions:
Mean Squared Error (MSE)
L = 1
n ෍
1
n
yi − ෝyi 2
Used for: Regression tasks (predicting continuous
values)
Squaring penalises large errors more heavily than
small ones. The function is smooth and convex for
linear models, making optimisation straightforward.
Binary Cross-Entropy (BCE)
L = −1
n ෍
1
n
yilog(ෝyi) + (1 −yi)log(1 −ෝyi)
Used for: Binary classification tasks.
Heavily penalises confident wrong predictions
(predicting 0.99 when the true label is 0). Requires
outputs in the range (0, 1), which is why we pair it
with the sigmoid activation function.


<a id="page-20"></a>

## Page 20: The Loss Landscape: An Intuition

![Original page 20](assets/2_Artificial_Neural_Networks/page_020.png)

### Extracted content

20 | © WMG 2026
The Loss Landscape: An Intuition
Imagine the loss function as a mountain range: we want to
find the lowest valley.
The Mountain Analogy:
•
You are blindfolded on a mountainside. You can’t see
the landscape, but you can feel the slope of the ground
beneath your feet.
•
The slope tells you which direction is downhill. In
mathematical terms, the gradient of the loss function
tells you the direction of steepest increase, so you
move in the opposite direction.
•
Each step takes you closer to the valley. The size of
each step is controlled by the learning rate α. Too large
and you overshoot; too small and you take forever.
•
The valley is the minimum loss, i.e., the set of weight
values where your model’s predictions are as accurate
as possible on the training data.


<a id="page-21"></a>

## Page 21: The Gradient: Direction and Magnitude

![Original page 21](assets/2_Artificial_Neural_Networks/page_021.png)

### Extracted content

21 | © WMG 2026
The Gradient: Direction and Magnitude
The mathematical compass that tells us how to change each weight
The gradient of a function is a vector of all its partial derivatives. For a loss function 𝐿 with weights
𝑤₁, 𝑤₂, … , 𝑤ₙ and bias 𝑏:
∇𝐿 = [ 𝜕𝐿/𝜕𝑤₁, 𝜕𝐿/𝜕𝑤₂, … , 𝜕𝐿/𝜕𝑤ₙ, 𝜕𝐿/𝜕𝑏 ]
What each partial derivative tells us:
𝝏𝑳/𝝏𝒘ᵢ > 𝟎
Increasing 𝑤ᵢ increases the loss. Therefore, we should decrease 𝑤ᵢ to reduce error.
𝝏𝑳/𝝏𝒘ᵢ < 𝟎
Increasing 𝑤ᵢ decreases the loss. Therefore, we should increase 𝑤ᵢ to reduce error.
𝝏𝑳/𝝏𝒘ᵢ = 𝟎
The loss is at a critical point with respect to 𝑤ᵢ. This weight is already locally optimal
(a minimum, maximum, or saddle point).
|𝝏𝑳/𝝏𝒘ᵢ| large The loss is very sensitive to changes in 𝑤ᵢ: a small change in this weight causes a big
change in prediction error. This weight needs a bigger update.


<a id="page-22"></a>

## Page 22: The Gradient Descent Algorithm

![Original page 22](assets/2_Artificial_Neural_Networks/page_022.png)

### Extracted content

22 | © WMG 2026
The Gradient Descent Algorithm
An iterative optimisation procedure that minimises the loss function
1.
Initialise Parameters: Set all weights 𝑤 and bias 𝑏 to small random values (e.g., drawn from a
normal distribution). This gives the algorithm a random starting point on the loss landscape.
2.
Forward Pass: Pass the training data through the network to compute predictions ŷ. Using the
current parameters, calculate the output for every training example.
3.
Compute the Loss: Evaluate the loss function 𝐿(𝑤, 𝑏) to get a single scalar measuring total
prediction error across the training set (e.g., MSE or cross-entropy).
4.
Compute the Gradient: Calculate the partial derivative of 𝐿 with respect to every parameter:
∇𝐿 = [𝜕𝐿/𝜕𝑤₁, 𝜕𝐿/𝜕𝑤₂, … , 𝜕𝐿/𝜕𝑏]. This tells us the direction and rate of steepest ascent.
5.
Update Parameters: Move each parameter in the opposite direction of its gradient, scaled by
the learning rate α: 𝑤 ← 𝑤 – 𝛼 · 𝜕𝐿/𝜕𝑤 and 𝑏 ← 𝑏 – 𝛼 · 𝜕𝐿/𝜕𝑏.
6.
Repeat: Go back to Step 2. Each full pass through the training data is called an epoch. Continue
until the loss converges (stops decreasing significantly) or a maximum number of epochs is
reached.
The update rule 𝑤 ← 𝑤 – 𝛼∇𝐿 is an important equation in deep learning. Everything else is how we
compute ∇𝐿 efficiently.


<a id="page-23"></a>

## Page 23: The Learning Rate α

![Original page 23](assets/2_Artificial_Neural_Networks/page_023.png)

### Extracted content

23 | © WMG 2026
The Learning Rate α
The single most important hyperparameter in gradient descent.
α Too Small
Convergence is extremely slow.
The model may need hundreds
of thousands of iterations to
reach
a
good
solution.
In
practice,
training
may
be
stopped before convergence,
resulting in an under-trained
model.
α Just Right
Smooth,
steady
convergence
toward the minimum. The loss
decreases consistently, slowing
down as it approaches the
optimum. This is the behaviour
we aim for.
α Too Large
The
updates
overshoot
the
minimum repeatedly, causing
the loss to oscillate or even
diverge (increase). The model
never
converges
and
may
produce
NaN
values
as
gradients explode.
0
2
4
6
8
10
12
1
2
3
4
5
6
7
8
9
10
11
12
0
2
4
6
8
10
12
1
2
3
4
5
6
7
8
9
10
11
12
0
5
10
15
20
25
1
2
3
4
5
6
7
8
9
10
11
12


<a id="page-24"></a>

## Page 24: Effects of Learning Rate Choice

![Original page 24](assets/2_Artificial_Neural_Networks/page_024.png)

### Extracted content

24 | © WMG 2026
Effects of Learning Rate Choice
α Too Small
Convergence is extremely slow. The
model may need hundreds of thousands
of iterations to reach a good solution.
α Just Right
Smooth, steady convergence toward the
minimum.
The
loss
decreases
consistently,
slowing
down
as
it
approaches the optimum. This is the
behaviour we aim for.
α Too Large
The updates overshoot the minimum
repeatedly, causing the loss to oscillate
or even diverge (increase).


<a id="page-25"></a>

## Page 25: Differentiable Activation Functions

![Original page 25](assets/2_Artificial_Neural_Networks/page_025.png)

### Extracted content

25 | © WMG 2026
Differentiable Activation Functions
Gradient descent requires that every component of the network be differentiable. The step
function has zero gradient everywhere (except at the discontinuity where it is undefined), so we
replace it with smooth functions whose derivatives are well-defined. These smooth activations
allow the chain rule to propagate gradients through every layer.
Sigmoid
𝜎(𝑧) = 1 / (1 + 𝑒⁻ᶿ)
Output: (0, 1)
✓
Smooth,
probabilistic
interpretation, good for binary
output layers.
✗ Vanishing gradients for large
|z|; outputs not zero-centred;
slow
convergence
in
deep
networks.
Tanh
tanh(𝑧) = (𝑒ᶿ – 𝑒⁻ᶿ)/(𝑒ᶿ + 𝑒⁻ᶿ)
Output: (–1, 1)
✓
Zero-centred
output
(helps
gradient flow); stronger gradients
than sigmoid.
✗ Still suffers vanishing gradients
for extreme inputs; computationally
more expensive than ReLU.
ReLU
𝑓(𝑧) = max(0, 𝑧)
Output: [0, ∞)
✓ Computationally efficient; no vanishing
gradient for positive inputs; enables
sparse activation
✗ Dying ReLU problem: neurons with
negative input always output 0 and stop
learning. Variants: Leaky ReLU, ELU, GELU.
0
0.2
0.4
0.6
0.8
1
-3
-2.5
-2
-1.5
-1
-0.5
0
0.5
1
1.5
2
2.5
3
-1.5
-1
-0.5
0
0.5
1
1.5
-3
-2.5
-2
-1.5
-1
-0.5
0
0.5
1
1.5
2
2.5
3
0
0.5
1
1.5
2
2.5
3
3.5
-3
-2.5
-2
-1.5
-1
-0.5
0
0.5
1
1.5
2
2.5
3


<a id="page-26"></a>

## Page 26: Variants of Gradient Descent

![Original page 26](assets/2_Artificial_Neural_Networks/page_026.png)

### Extracted content

26 | © WMG 2026
Variants of Gradient Descent
How much data we use per update fundamentally changes the learning dynamics
Batch Gradient Descent
Uses the entire training set per update.
How: Computes the gradient of the loss
averaged over all 𝑛 training samples,
then makes a single weight update.
✓ Stable, deterministic convergence;
true
gradient
direction;
guaranteed
convergence to a local minimum (for
convex problems).
✗ Very slow for large datasets (must
process all data before one update);
requires the entire dataset in memory;
can get stuck in local minima.
Best
for:
Small
datasets,
convex
problems, when memory is not a
constraint.
Stochastic GD (SGD)
Uses one sample per update
How: Computes the gradient from a
single
randomly
selected
training
example, then immediately updates the
weights.
✓
Very
fast
updates;
natural
regularisation from noise; can escape
local
minima
due
to
stochastic
fluctuations.
✗ High variance in gradient estimates;
noisy convergence; may never fully
converge
(oscillates
around
the
minimum).
Best for: Online learning, very large
datasets, when you want to escape poor
local minima.
Mini-Batch GD
Uses a subset (batch) per update
How: Computes the gradient from a
random subset of 𝑚 samples (typically
32–256), then updates weights.
✓ Best of both worlds: stable enough for
convergence but stochastic enough to
escape local minima. Efficient GPU
utilisation.
✗ Introduces a new hyperparameter
(batch size); requires tuning. Too small →
noisy;
too
large
→
slow,
poor
generalisation.
Best for: Default choice in practice.
Nearly all modern deep learning uses
mini-batch SGD or its variants (Adam,
RMSProp).


<a id="page-27"></a>

## Page 27: Challenges: Local Minima & Saddle Points

![Original page 27](assets/2_Artificial_Neural_Networks/page_027.png)

### Extracted content

27 | © WMG 2026
Challenges: Local Minima & Saddle Points
Gradient descent does not guarantee finding the global optimum
Local Minimum: A point where the loss is lower than all nearby
points, but not necessarily the lowest overall. Gradient descent
converges here because the gradient is zero, but better solutions
may exist elsewhere. In high dimensions, true local minima are rare
– most “traps” are actually saddle points.
Global Minimum: The absolute lowest point of the loss function.
This is the ideal solution, but it may be computationally infeasible
to find. In practice, any sufficiently good local minimum often
generalises well.
Saddle Point: A point where the gradient is zero, but it is a
minimum in some dimensions and a maximum in others. Very
common in high-dimensional spaces. SGD’s noise helps escape
these, unlike batch gradient descent.
Plateau: A flat region where the gradient is near zero. Learning
slows dramatically. Momentum and adaptive learning rate methods
(Adam) help traverse plateaus more efficiently.
0
1
2
3
4
5
6
7
8
9
local
minimum
global
minimum
saddle
point


<a id="page-28"></a>

## Page 28: Modern Optimisers: Beyond Vanilla GD

![Original page 28](assets/2_Artificial_Neural_Networks/page_028.png)

### Extracted content

28 | © WMG 2026
Modern Optimisers: Beyond Vanilla GD
Enhancements that make gradient descent faster, more stable, and more robust.
SGD + Momentum
𝑣 ← 𝛽𝑣 + ∇𝐿; 𝑤 ← 𝑤 – 𝛼𝑣
Accumulates a velocity term from past
gradients (typically β = 0.9). Like a ball rolling
downhill – it builds speed in consistent
directions and dampens oscillations. Helps
traverse
ravines
and
plateaus
more
efficiently.
RMSProp
𝑠 ← 𝛽𝑠 + (1– 𝛽)(∇𝐿)²; 𝑤 ← 𝑤 – 𝛼∇𝐿/√𝑠
Adapts the learning rate for each parameter
individually based on the magnitude of
recent gradients. Parameters with large
gradients get smaller updates (prevents
explosion); parameters with small gradients
get larger updates (prevents stalling).


<a id="page-29"></a>

## Page 29: Modern Optimisers: Beyond Vanilla GD

![Original page 29](assets/2_Artificial_Neural_Networks/page_029.png)

### Extracted content

29 | © WMG 2026
Modern Optimisers: Beyond Vanilla GD
Enhancements that make gradient descent faster, more stable, and more robust.
AdamW
Adam + decoupled weight decay
Fixes a subtle issue in Adam where L2
regularisation and adaptive learning
rates interact poorly. Decouples weight
decay from the gradient update. Widely
used in transformer training (BERT, GPT,
ViT) and recommended as the modern
default.
Adam
Combines Momentum + RMSProp
Maintains both a first-moment estimate
(mean of gradients = momentum) and a
second-moment
estimate
(mean
of
squared gradients = adaptive rate).
Includes bias correction for the initial
estimates. The default choice for most
deep learning tasks. Typical defaults: α =
0.001, β₁ = 0.9, β₂ = 0.999.


<a id="page-30"></a>

## Page 30: Worked Example: 1D Gradient Descent

![Original page 30](assets/2_Artificial_Neural_Networks/page_030.png)

### Extracted content

30 | © WMG 2026
Worked Example: 1D Gradient Descent
Tracing through the algorithm step by step with 𝐿(𝑤) = (𝑤 – 3)²
Setup: 𝐿(𝑤) = (𝑤 – 3)²

𝜕𝐿/𝜕𝑤 = 2(𝑤 – 3)
𝛼 = 0.1

𝑤₀ = 0
Notice: the gradient is large when we’re far from the minimum (–6.00) and shrinks as we
approach it (–0.41). Each step moves us closer, but the steps get smaller. The loss decreases
monotonically: 9.00 → 5.76 → 3.69 → … → 0.00
Step
w
∂L/∂w = 2(w–3)
Update: w – α·∂L/∂w
New w
L(w)
0
0.00
2(0–3) = –6.00
0 – 0.1(–6) = 0 + 0.6
0.60
9.00
1
0.60
2(0.6–3) = –4.80
0.6 – 0.1(–4.8)
1.08
5.76
2
1.08
2(1.08–3) = –3.84
1.08 – 0.1(–3.84)
1.46
3.69
5
2.03
–1.95
2.03 + 0.195
2.22
0.95
10
2.80
–0.41
2.80 + 0.041
2.84
0.04
∞
3.00
0.00
3 – 0 = 3
3.00 ✓
0.00 ✓


<a id="page-31"></a>

## Page 31: Summary

![Original page 31](assets/2_Artificial_Neural_Networks/page_031.png)

### Extracted content

31 | © WMG 2026
Summary
The Perceptron: Foundation of Neural Computation
Weighted sum + step activation | Linear classifier | Fails on XOR
The Perceptron Learning Algorithm
Forward pass → compute error → update weights | Converges for linearly separable data
Gradient Descent: Enabling Multi-Layer Learning
Loss function → gradient → smooth weight updates | Learning rate controls step size


<a id="page-32"></a>

## Page 32: III. Artificial Neural

![Original page 32](assets/2_Artificial_Neural_Networks/page_032.png)

### Extracted content

32 | © WMG 2026
III. Artificial Neural
Network (ANN)


<a id="page-33"></a>

## Page 33: From Optimisation to Architecture

![Original page 33](assets/2_Artificial_Neural_Networks/page_033.png)

### Extracted content

33 | © WMG 2026
From Optimisation to Architecture
We now have the key ingredients:
•
Gradient Descent gives us a smooth, principled way to update weights.
•
Differentiable activation functions (sigmoid, ReLU) replace the step function.
•
The chain rule lets us propagate gradients through multiple layers.
The question becomes: how do we assemble neurons into a network architecture that can learn
complex, non-linear patterns?


<a id="page-34"></a>

## Page 34: The Neuron: A Computational Unit

![Original page 34](assets/2_Artificial_Neural_Networks/page_034.png)

### Extracted content

34 | © WMG 2026
The Neuron: A Computational Unit
The building block of any neural network is a
single neuron:
- Inputs:
𝑥= [𝑥1, 𝑥2, 𝑥3]
- Parameters/Weights:
𝑤= [𝑤1, 𝑤2, 𝑤3]
-
𝑓 is the activation function.
Therefore:
𝑎= 𝑓(𝑤1𝑥1 + 𝑤2𝑥2 + 𝑤3𝑥3)
ℎ𝑤𝑥= 𝑎= 𝑓(𝑤1𝑥1 + 𝑤2𝑥2 + 𝑤3𝑥3)
x1
x2
x3
hW
a


<a id="page-35"></a>

## Page 35: The Neuron: Adding the Bias Term

![Original page 35](assets/2_Artificial_Neural_Networks/page_035.png)

### Extracted content

35 | © WMG 2026
The Neuron: Adding the Bias Term
x1
x2
x3
hW
a
x0
We add a bias term to shift the activation function:
- Inputs:
𝑥= 𝑥0, 𝑥1, 𝑥2, 𝑥3
where 𝑥0 = 1:
𝑥= 1, 𝑥1, 𝑥2, 𝑥3
- Parameters/Weights:
𝑤= [𝑤0, 𝑤1, 𝑤2, 𝑤3]
- 𝑓 is the activation function.
Therefore:
ℎ𝑤𝑥= 𝑓(𝑤0 + 𝑤1𝑥1 + 𝑤2𝑥2 + 𝑤3𝑥3)


<a id="page-36"></a>

## Page 36: From Neuron to Network

![Original page 36](assets/2_Artificial_Neural_Networks/page_036.png)

### Extracted content

36 | © WMG 2026
From Neuron to Network
A neural network is formed by connecting
multiple neurons in layers.
Each neuron in one layer sends its output
to every neuron in the next layer (fully
connected).
x1
x2
x3
hW
a1
(2)
a2
(2)
a1
(3)
Input Layer
Output Layer
Hidden Layer
Layer 1
Layer 3
Layer 2


<a id="page-37"></a>

## Page 37: From Perceptron to Multi-Layer Networks

![Original page 37](assets/2_Artificial_Neural_Networks/page_037.png)

### Extracted content

37 | © WMG 2026
From Perceptron to Multi-Layer Networks
Single neuron → Layers of neurons
Stacking neurons creates intermediate representations.
Each layer transforms the data, enabling the network to
learn non-linear boundaries. The first hidden layer
might learn simple edges; deeper layers compose
them into complex patterns.
Step activation → Smooth activations
Replacing the step function with sigmoid/ReLU makes
every operation differentiable. This allows the chain
rule to propagate gradients from the output layer back
through every hidden layer to the input weights.
Perceptron rule → Backpropagation
Instead of the simple error-correction rule (𝑤 ← 𝑤
+ 𝛼𝑒𝑦), we use the chain rule of calculus to compute
the gradient of the loss with respect to every weight in
every layer. This is backpropagation – the core training
algorithm.
Recall: A single perceptron computes a
weighted sum, applies a step function, and
produces a binary output. It can only learn
linearly separable patterns (AND, OR), but
fails on non-linear problems like XOR.
Recall: Gradient descent minimises a differentiable loss by iteratively
adjusting weights along the negative gradient. Smooth activations
(sigmoid, tanh, ReLU) make every operation differentiable.
The key insight: Combining layers of neurons with smooth activations
lets us train deep networks end-to-end via backpropagation.


<a id="page-38"></a>

## Page 38: The Multi-Layer Perceptron (MLP)

![Original page 38](assets/2_Artificial_Neural_Networks/page_038.png)

### Extracted content

38 | © WMG 2026
The Multi-Layer Perceptron (MLP)
The foundational architecture of artificial neural
networks
Input
Layer
Hidden
Layer 1
Hidden
Layer 2
Output
Layer
x₁
x₂
x₃
ŷ₁
ŷ₂
Input Layer
Receives the raw feature values. One neuron per feature.
No computation occurs here; it simply passes data
forward. For a 28×28-pixel image, this layer has 784
neurons.
Hidden Layer(s)
Where the network learns internal representations. Each
neuron computes 𝑧 = Σ𝑤ᵢ𝑥ᵢ + 𝑏, then applies an
activation function. Depth (number of hidden layers) and
width (neurons per layer) are key design choices. More
layers = more abstract features.
Output Layer
Produces the final prediction. Architecture depends on
the task: one neuron with a sigmoid for binary
classification; 𝑛 neurons with a softmax for multi-class;
one neuron with a linear activation for regression.


<a id="page-39"></a>

## Page 39: Weight Notation & Biases per Layer

![Original page 39](assets/2_Artificial_Neural_Networks/page_039.png)

### Extracted content

39 | © WMG 2026
Weight Notation & Biases per Layer
Conventional weight
representation:
wdo
(L)
L = layer
d = destination neuron
o = origin neuron
Note: Biases are added per layer
but not always drawn in diagrams.
x1
x2
x3
hW
a1
(2)
a2
(2)
a1
(3)
x0
a0
(2)
Layer 1
Layer 3
Layer 2


<a id="page-40"></a>

## Page 40: Enumerating Weights (Layer 1 → 2)

![Original page 40](assets/2_Artificial_Neural_Networks/page_040.png)

### Extracted content

40 | © WMG 2026
Enumerating Weights (Layer 1 → 2)
How many weights/parameters?
For a1
(2): 𝑤11
1 , 𝑤12
1 , 𝑤13
(1)
For a2
(2): 𝑤21
1 , 𝑤22
1 , 𝑤23
(1)
With biases
For a1
(2): 𝑤10
(1), 𝑤11
1 , 𝑤12
1 , 𝑤13
(1)
For a2
(2): 𝑤20
(1), 𝑤21
1 , 𝑤22
1 , 𝑤23
(1)
x1
x2
x3
hW
a1
(2)
a2
(2)
a1
(3)
w11
(1)
w12
(1)
w13
(1)
w21
(1)
w22
(1)
w23
(1)


<a id="page-41"></a>

## Page 41: Enumerating Weights (Layer 2 → 3)

![Original page 41](assets/2_Artificial_Neural_Networks/page_041.png)

### Extracted content

41 | © WMG 2026
Enumerating Weights (Layer 2 → 3)
What about the 3rd Layer?
For a1
(3): 𝑤10
(2), 𝑤11
2 , 𝑤12
2
x1
x2
x3
hW
a1
(2)
a2
(2)
a1
(3)
w11
(1)
w12
(1)
w13
(1)
w21
(1)
w22
(1)
w23
(1)
x0
w10
(1)
w20
(1)
a0
(2)


<a id="page-42"></a>

## Page 42: Counting Parameters

![Original page 42](assets/2_Artificial_Neural_Networks/page_042.png)

### Extracted content

42 | © WMG 2026
Counting Parameters
How to calculate the total number of
parameters between two layers in general?
𝑙𝑎𝑦𝑒𝑟𝑖+ 1 × 𝑙𝑎𝑦𝑒𝑟𝑖+1
Therefore, between layers 1 and 2:
3 + 1 × 2 = 8
A total 8 parameters.
What about layers 2 and 3?
2 + 1 × 1 = 3 parameters
x1
x2
x3
hW
a1
(2)
a2
(2)
a1
(3)
w11
(1)
w12
(1)
w13
(1)
w21
(1)
w22
(1)
w23
(1)
x0
w10
(1)
w20
(1)
a0
(2)
w10
(2)
w11
(2)
w12
(2)
Why this matters: Parameter count directly
affects memory requirements, risk of overfitting,
and computational cost. More parameters need
proportionally more training data.


<a id="page-43"></a>

## Page 43: Forward Propagation: Layer 1 → 3

![Original page 43](assets/2_Artificial_Neural_Networks/page_043.png)

### Extracted content

43 | © WMG 2026
Forward Propagation: Layer 1 → 3
Therefore:
a1
(2) = 𝑓(𝑤10
1 𝑥0 + 𝑤11
(1)𝑥1 + 𝑤12
(1)𝑥2 + 𝑤13
(1)𝑥3)
a2
(2) = 𝑓(𝑤20
1 𝑥0 + 𝑤21
(1)𝑥1 + 𝑤22
(1)𝑥2 + 𝑤23
(1)𝑥3)
Note: 𝑥0 = 1
Now, we can go to the next layer (layer 3):
a1
(3) = 𝑓𝑤10
2 𝑎0
2 + 𝑤11
2 𝑎1
2 + 𝑤12
2 𝑎2
2
Note: 𝑎0 = 1
x1
x2
x3
hW
a1
(2)
a2
(2)
a1
(3)


<a id="page-44"></a>

## Page 44: Worked Example: Tracing a Forward Pass

![Original page 44](assets/2_Artificial_Neural_Networks/page_044.png)

### Extracted content

44 | © WMG 2026
Worked Example: Tracing a Forward Pass
Consider a 2-2-1 network with sigmoid activation.
Inputs: x = [0.5, 0.8]
Key insight: The input [0.5, 0.8] was
transformed through two non-linear
layers to produce a continuous output
of 0.584.
hW
Layer 1 → 2 weights: Biases: b = [0.1, -0.2]
w₁₁ = 0.4, w₁₂ = 0.3
w₂₁ = -0.5, w₂₂ = 0.6
Step 1: Compute hidden layer pre-activations
z₁ = (0.4)(0.5) + (0.3)(0.8) + 0.1 = 0.2 + 0.24 + 0.1 = 0.54
z₂ = (-0.5)(0.5) + (0.6)(0.8) + (-0.2) = -0.25 + 0.48 - 0.2 = 0.03
Step 2: Apply sigmoid activation
a₁ = σ(0.54) = 1/(1+e⁻⁰·⁵⁴) ≈ 0.632
a₂ = σ(0.03) = 1/(1+e⁻⁰·⁰³) ≈ 0.507
Step 3: Compute output (weights: v₁=0.7, v₂=-0.4, bias=0.1)
z₃ = (0.7)(0.632) + (-0.4)(0.507) + 0.1 = 0.442 - 0.203 + 0.1 = 0.339
hw = σ(0.339) ≈ 0.584
0.5
0.8
0.4
0.6
1
1
0.632
0.507
0.584


<a id="page-45"></a>

## Page 45: Think–Pair–Share

![Original page 45](assets/2_Artificial_Neural_Networks/page_045.png)

### Extracted content

45 | © WMG 2026
Think–Pair–Share
In pairs, work through this exercise (3 minutes):
Given a 2-2-1 network with ReLU activation and these weights:
Layer 1→2: w = [[0.5, -0.3], [0.2, 0.8]], biases = [0.1, -0.1]
Layer 2→3: v = [0.6, 0.4], bias = 0.05
Trace the forward pass for input x = [1.0, 0.0]:
1. What are the pre-activations (z) at the hidden layer?
2. After applying ReLU, what are the activations (a)?
3. What is the final output?
Compare: how does the output change if you use input x = [0.0, 1.0] instead?


<a id="page-46"></a>

## Page 46: Vectorisation

![Original page 46](assets/2_Artificial_Neural_Networks/page_046.png)

### Extracted content

46 | © WMG 2026
Vectorisation
z = wx + b
Non-vectorised:
z=0
For i in range (nx):

z+=w[i] * x[i]
z+=b
Vectorised:
z=np.dot(w,x) + b

wx
w =
.
.
.
.
.
x =
.
.
.
.
.
Conventionally, instead of computing one neuron
at a time with a for-loop, we compute all neurons in
a layer simultaneously using matrix multiplication.
We perform this with vectorisation.


<a id="page-47"></a>

## Page 47: Why Vectorisation?

![Original page 47](assets/2_Artificial_Neural_Networks/page_047.png)

### Extracted content

47 | © WMG 2026
Why Vectorisation?
GPUs are designed for exactly this — thousands of parallel
multiply-and-add operations.
https://www.datagravity.dev/p/the-ai-first-transformation-of-the


<a id="page-48"></a>

## Page 48: Neural Network Vectorisation

![Original page 48](assets/2_Artificial_Neural_Networks/page_048.png)

### Extracted content

48 | © WMG 2026
Neural Network Vectorisation
𝑊(1) = 𝑤10
1
𝑤11
(1)
𝑤12
(1)
𝑤20
1
𝑤21
(1)
𝑤22
(1)
𝑤13
(1)
𝑤23
(1)
a1
(2) = 𝑓(𝑤10
1 𝑥0 + 𝑤11
(1)𝑥1 + 𝑤12
(1)𝑥2 + 𝑤13
(1)𝑥3)
a2
(2) = 𝑓(𝑤20
1 𝑥0 + 𝑤21
(1)𝑥1 + 𝑤22
(1)𝑥2 + 𝑤23
(1)𝑥3)
a1
(3) = 𝑓(𝑤10
2 𝑎0
2 + 𝑤11
2 𝑎1
2 + 𝑤12
2 𝑎2
(2))
x1
x2
x3
hW
a1
(2)
a2
(2)
a1
(3)
𝑥=
𝑥0
𝑥1
𝑥2
𝑥3


<a id="page-49"></a>

## Page 49: Neural Network Vectorisation

![Original page 49](assets/2_Artificial_Neural_Networks/page_049.png)

### Extracted content

49 | © WMG 2026
Neural Network Vectorisation
𝑎(2) = 𝑓𝑊(1)𝑥

𝑎(2) = 𝑓
𝑤10
1 𝑥0
𝑤11
(1)𝑥1
𝑤12
(1)𝑥2
𝑤20
1 𝑥0
𝑤21
(1)𝑥1
𝑤22
(1)𝑥2
𝑤13
(1)𝑥3
𝑤23
(1)𝑥3
𝑎(2) = a1
(2)
a2
(2)
a1
(2) = 𝑓(𝑤10
1 𝑥0 + 𝑤11
(1)𝑥1 + 𝑤12
(1)𝑥2 + 𝑤13
(1)𝑥3)
a2
(2) = 𝑓(𝑤20
1 𝑥0 + 𝑤21
(1)𝑥1 + 𝑤22
(1)𝑥2 + 𝑤23
(1)𝑥3)
a1
(3) = 𝑓(𝑤10
2 𝑎0
2 + 𝑤11
2 𝑎1
2 + 𝑤12
2 𝑎2
(2))
Note that 𝑥0 = 𝑎0
(2) = 1 (𝑏𝑖𝑎𝑠)
x1
x2
x3
hW
a1
(2)
a2
(2)
a1
(3)


<a id="page-50"></a>

## Page 50: Forward Propagation: Data Through the Network

![Original page 50](assets/2_Artificial_Neural_Networks/page_050.png)

### Extracted content

50 | © WMG 2026
Forward Propagation: Data Through the Network
𝑎(2) = 𝑓𝑊(1)𝑥 (note x can be represented by a(1) now for easy loop-based iterations)
x1
x2
x3
hW
a1
(2)
a2
(2)
a1
(3)
x0
a0
(2)
Layer 1
Layer 3
Layer 2
a1
(1)
a2
(1)
a3
(1)
hW
a1
(2)
a2
(2)
a1
(3)
a0
(1)
a0
(2)
Layer 1
Layer 3
Layer 2


<a id="page-51"></a>

## Page 51: Forward Propagation: Data Through the Network

![Original page 51](assets/2_Artificial_Neural_Networks/page_051.png)

### Extracted content

51 | © WMG 2026
Forward Propagation: Data Through the Network
Therefore:
𝑎(1) = [𝑥0, 𝑥1, 𝑥2, 𝑥3]
𝑎(2) = 𝑓𝑊(1)𝑎(1)
𝑎(3) = 𝑓𝑊(2)𝑎(2)
Note, 𝑎0
(1) = 𝑎0
(2) = 1 (𝑏𝑖𝑎𝑠)
a1
(1)
a2
(1)
a3
(1)
hW
a1
(2)
a2
(2)
a1
(3)


<a id="page-52"></a>

## Page 52: Forward Propagation: Data Through the Network

![Original page 52](assets/2_Artificial_Neural_Networks/page_052.png)

### Extracted content

52 | © WMG 2026
Forward Propagation: Data Through the Network
How an input is transformed layer by layer into a
prediction:
General Formula for Layer 𝑙:
𝑧𝑙= 𝑊𝑙∙𝑎𝑙−1 + 𝑏𝑙
𝑎𝑙= 𝑓(𝑧𝑙)
where,
𝑊𝑙 is the weight matrix,
𝑎𝑙−1 is the previous layer’s activations,
𝑏𝑙 is the bias vector,
and 𝑓 is the activation function.
𝑎(1) = [𝑥0, 𝑥1, 𝑥2, 𝑥3]
𝑎(2) = 𝑓𝑊(1)𝑎(1)
𝑎(3) = 𝑓𝑊(2)𝑎(2)
𝑎0
(1) = 𝑎0
(2) = 1 (𝑏𝑖𝑎𝑠)


<a id="page-53"></a>

## Page 53: Looping through the Layers

![Original page 53](assets/2_Artificial_Neural_Networks/page_053.png)

### Extracted content

53 | © WMG 2026
Looping through the Layers
for i = 1 to n - 1
a0
(i)= 1 (added bias term)
a(i+1) = g(W(i) a(i))
hΘ
a0
(2)
a1
(n)
a1
(2)
a2
(2)
a0
(1)
a1
(1)
a2
(1)
a0
(3)
a1
(3)
a2
(3)
a0
(n-1)
a1
(n-1)
a2
(n-1)
…
Forward Propagation


<a id="page-54"></a>

## Page 54: Why We Need Non-linear Models

![Original page 54](assets/2_Artificial_Neural_Networks/page_054.png)

### Extracted content

54 | © WMG 2026
Why We Need Non-linear Models
The Curse of Polynomial Features
To learn non-linear decision boundaries with a single perceptron,
we must manually engineer polynomial features:
•
O(n2) features for quadratic boundaries
•
O(n3) features for cubic boundaries
For 2 variables, this is manageable. But for n = 100 features, even
quadratic terms give ~5,000 features. This is computationally
explosive and impractical.
How do ANNs help? Instead of hand-crafting features, hidden
layers learn useful representations automatically.
x1
x2


<a id="page-55"></a>

## Page 55: Learn Useful Representations Automatically

![Original page 55](assets/2_Artificial_Neural_Networks/page_055.png)

### Extracted content

55 | © WMG 2026
Learn Useful Representations Automatically
Building Logic with Neurons
Each neuron in the hidden layer learns a different linear decision boundary. By combining these
boundaries, the network can represent non-linear regions.
hΘ
a0
(2)
a1
(n)
a1
(2)
a2
(2)
a0
(1)
a1
(1)
a2
(1)
a0
(3)
a1
(3)
a2
(3)
a0
(n-1)
a1
(n-1)
a2
(n-1)
…
a1
(n) = g(w10
(n-1) a0
(n-1) + w11
(n-1) a1
(n-1) + w12
(n-1)a2
(n-1))
new features


<a id="page-56"></a>

## Page 56: Designing an ANN: Practical Choices

![Original page 56](assets/2_Artificial_Neural_Networks/page_056.png)

### Extracted content

56 | © WMG 2026
Designing an ANN: Practical Choices
The key decisions you must make when building a neural network from scratch
Decision
Guideline
Typical Choices
Number of
hidden layers
Start with 1–2 for tabular data. 3–5 for complex patterns. More
layers = more abstract features but harder to train. Always start
simple.
1 layer: logistic regression++; 2–3
layers: most tabular tasks; 5+: images,
text, audio
Neurons per
layer
Common patterns: funnel (wide→narrow), constant width, or
expanding. Typically 32–1024. Wider layers = more capacity but
more parameters and risk of overfitting.
[128, 64, 32] funnel; [256, 256, 256]
constant; Powers of 2 are efficient on
GPUs
Activation
functions
ReLU is the default for hidden layers (fast, avoids vanishing
gradients). Sigmoid for binary output. Softmax for multi-class
output. Linear for regression output.
Hidden: ReLU (or Leaky ReLU, GELU for
transformers); Output: task-dependent
Loss function
Determined by the task. BCE for binary classification. Categorical
cross-entropy for multi-class. MSE for regression. Must match the
output activation.
Binary: BCE + sigmoid; Multi-class: CCE
+ softmax; Regression: MSE + linear
Optimiser
and
learning rate
Adam (α=0.001) is the safe default. SGD+momentum for fine-
tuning. Use learning rate schedulers (cosine, step decay) for better
convergence.
Adam:
α=0.001;
SGD:
α=0.01,
momentum=0.9;
AdamW
for
transformers
Weight
initialisation
Random initialisation breaks symmetry so neurons learn different
features. Xavier/Glorot for sigmoid/tanh. He/Kaiming for ReLU.
Never initialise all weights to zero.
He: w ~ N(0, √(2/n_in)); Xavier: w ~ N(0,
√(1/n_in))


<a id="page-57"></a>

## Page 57: Overfitting and Regularisation

![Original page 57](assets/2_Artificial_Neural_Networks/page_057.png)

### Extracted content

57 | © WMG 2026
Overfitting and Regularisation
Preventing the network from memorising training data instead of learning
generalisable patterns.
• The Overfitting Problem: Neural networks have millions of learnable
parameters, far more than the number of training samples in most
cases. This gives them the capacity to memorise training data perfectly
(training loss → 0) while performing poorly on unseen data. The gap
between training and validation performance is the generalisation gap.
Regularisation techniques reduce this gap.
• Dropout: During training, randomly set a fraction p (typically 0.2–0.5)
of neuron outputs to zero at each forward pass. This forces the network
to learn redundant representations; no single neuron can become a
critical bottleneck. At inference, all neurons are active, but outputs are
scaled by (1–p). This is the "standard" dropout. In modern frameworks
like PyTorch and TensorFlow, inverted dropout
is the standard
implementation, in which outputs are scaled by 1/(1−p) during
training, so no scaling is needed at inference.
0
1
2
3
4
5
6
1
5
10
15
20
25
30
35
40
45
50
Training Loss
Validation Loss
Overfitting
starts here ↓


<a id="page-58"></a>

## Page 58: Overfitting and Regularisation

![Original page 58](assets/2_Artificial_Neural_Networks/page_058.png)

### Extracted content

58 | © WMG 2026
Overfitting and Regularisation
Preventing the network from memorising training data instead of learning
generalisable patterns.
•
L2 Regularisation (Weight Decay): Add a penalty term 𝜆Σ𝑤ᵢ² to the
loss function. This discourages large weights, pushing the model
toward simpler, smoother decision boundaries. Equivalent to applying
a Gaussian prior on the weights. Controlled by the hyperparameter λ.
•
Early Stopping: Monitor validation loss during training. Stop when
validation loss begins to increase (even if training loss continues to
decrease). Restore the weights from the epoch with the lowest
validation loss. Simple, effective, and universally applicable.
•
Batch Normalisation: Normalise the inputs to each layer to have zero
mean and unit variance, then scale and shift with learnable parameters.
Stabilises training, allows higher learning rates, and acts as a mild
regulariser. Applied before or after the activation function.
0
1
2
3
4
5
6
1
5
10
15
20
25
30
35
40
45
50
Training Loss
Validation Loss
Overfitting
starts here ↓


<a id="page-59"></a>

## Page 59: The Complete Training Pipeline

![Original page 59](assets/2_Artificial_Neural_Networks/page_059.png)

### Extracted content

59 | © WMG 2026
The Complete Training Pipeline
Putting it all together: from raw data to a trained model.
1. Prepare Data: Split into train/validation/test sets (e.g., 70/15/15). Normalise features (zero mean, unit
variance or min-max scaling). One-hot encode categorical targets. Create mini-batches using a
DataLoader.
2. Define Architecture: Choose the number of layers, neurons per layer, and activation functions. Select
the loss function and optimiser. This defines the model's hypothesis space.
3. Initialise Weights: Apply He initialisation (for ReLU) or Xavier (for sigmoid/tanh). Good initialisation
ensures activations and gradients stay within a reasonable range.
4. Training Loop: For each epoch: shuffle data → for each mini-batch: forward pass → compute loss →
backward pass (backpropagation) → update weights. [We will formalise backpropagation in the next
session.]
5. Monitor and Regularise: Track training vs. validation loss curves. Apply early stopping when validation
loss plateaus. Use dropout, weight decay, or data augmentation as needed.
6. Evaluate on Test Set: Only after all tuning is complete, evaluate on the held-out test set exactly once.
Report metrics: accuracy, precision, recall, F1, AUC as appropriate.


<a id="page-60"></a>

## Page 60: Common Pitfalls and Diagnosis

![Original page 60](assets/2_Artificial_Neural_Networks/page_060.png)

### Extracted content

60 | © WMG 2026
Common Pitfalls and Diagnosis
What to look for when training goes wrong:
Pitfall
Likely Cause
Fix
Loss is NaN or
explodes
Learning rate too high; exploding gradients;
numerical overflow in loss computation (e.g.,
log(0)).
Reduce learning rate; add gradient clipping; check
data preprocessing; add epsilon to log computations.
Loss doesn’t
decrease
Learning
rate
too
small;
poor
weight
initialisation; bug in the data pipeline; dead
ReLU neurons (all activations are zero).
Increase learning rate; try different initialisation; verify
data labels; use Leaky ReLU; print gradient norms.
Training loss low,
validation loss
high
Overfitting. The model has memorised the
training data but cannot generalise to unseen
data.
Add
dropout,
L2
regularisation,
or
data
augmentation. Reduce model size. Apply early
stopping. Gather more training data if possible.
Both losses high
(underfitting)
Model too simple for the task; insufficient
training
time;
learning
rate
too
high
(oscillation); data too noisy.
Increase model capacity (more layers/neurons). Train
longer. Use learning rate scheduling. Clean data.
Vanishing
gradients
Deep networks with sigmoid/tanh activations.
Gradients
shrink
exponentially
as
they
propagate backward through many layers.
Use ReLU or its variants. Apply batch normalisation.
Use residual connections (skip connections). Try
LSTM/GRU for sequences.


<a id="page-61"></a>

## Page 61: Building ANNs Responsibly

![Original page 61](assets/2_Artificial_Neural_Networks/page_061.png)

### Extracted content

61 | © WMG 2026
Building ANNs Responsibly
As practitioners, we know that the models we build have real-world consequences. Consider these
questions throughout your work:
•
What does your training data include and exclude? Biased data produces biased models.
Representation gaps in training sets lead to unequal performance across demographic groups.
•
Who might be harmed by errors in your model? A misclassification in a spam filter is
inconvenient; in a medical diagnosis system, it can be life-threatening. The stakes determine the
rigour required.
•
ANNs are only as fair as their data. Research by Joy Buolamwini and Timnit Gebru demonstrated
that commercial facial recognition systems exhibited significant accuracy disparities across skin
tones and genders.
•
Transparency matters. Can you explain why your model made a particular prediction?
Interpretability is not just a technical challenge, but an ethical obligation in high-stakes
applications.
Study finds gender and skin-type bias in commercial artificial-intelligence systems | MIT News


<a id="page-62"></a>

## Page 62: Summary

![Original page 62](assets/2_Artificial_Neural_Networks/page_062.png)

### Extracted content

62 | © WMG 2026
Summary
ANN Architecture: Neurons, Layers, and Connections
Single neuron → multi-layer network | Weight notation | Parameter counting
Forward Propagation and Vectorisation
Layer-by-layer computation | Matrix operations for efficiency | Non-linear hypothesis
Practical Implementation Pipeline
Architecture choices | Overfitting & regularisation | Responsible AI | Next: Backpropagation


<a id="page-63"></a>

## Page 63: Do you

![Original page 63](assets/2_Artificial_Neural_Networks/page_063.png)

### Extracted content

63 | © WMG 2026
Do you
have any
questions?


<a id="page-64"></a>

## Page 64: warwick.ac.uk/wmg

![Original page 64](assets/2_Artificial_Neural_Networks/page_064.png)

### Extracted content

64 | © WMG 2026
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
