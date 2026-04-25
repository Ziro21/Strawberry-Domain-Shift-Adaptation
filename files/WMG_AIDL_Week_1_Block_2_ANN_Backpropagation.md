# WMG AIDL Week 1 Block 2 ANN Backpropagation

> Upgraded Markdown conversion. The original page image is included for every page so diagrams, screenshots, tables, slide layout and visual details are preserved. Extracted text is also provided below each page for searchable study notes.

## Contents

- [Page 1: Introduction to](#page-1)
- [Page 2: Agenda](#page-2)
- [Page 3: Learning Outcomes](#page-3)
- [Page 4: ANN: Back Propagation](#page-4)
- [Page 5: PERCEPTRON & ANN](#page-5)
- [Page 6: What a perceptron does](#page-6)
- [Page 7: Perceptron to regression](#page-7)
- [Page 8: Perceptron to logistic regression](#page-8)
- [Page 9: Two Perceptrons](#page-9)
- [Page 10: Notation*](#page-10)
- [Page 11: Combining two sigmoid](#page-11)
- [Page 12: Combining two sigmoid](#page-12)
- [Page 13: Combining neurons allows us to model interesting](#page-13)
- [Page 14: Different weights change the shape and position](#page-14)
- [Page 15: Neural networks can model any reasonable](#page-15)
- [Page 16: Adding layers allows us to model increasingly complex](#page-16)
- [Page 17: (2)](#page-17)
- [Page 18: (n)](#page-18)
- [Page 19: (2)](#page-19)
- [Page 20: (n)](#page-20)
- [Page 21: BACK PROPAGATION](#page-21)
- [Page 22: x =](#page-22)
- [Page 23: Linear Regression Cost](#page-23)
- [Page 24: Linear Regression Gradient Descent](#page-24)
- [Page 25: Gradient descent visualisation](#page-25)
- [Page 26: Examples](#page-26)
- [Page 27: Batch Gradient Descent](#page-27)
- [Page 28: Stochastic Gradient Descent](#page-28)
- [Page 29: Stochastic Gradient Descent](#page-29)
- [Page 30: Mini Batch Gradient Descent](#page-30)
- [Page 31: ANN Gradient Descent](#page-31)
- [Page 32: (2)](#page-32)
- [Page 33: Intuition](#page-33)
- [Page 34: Intuition](#page-34)
- [Page 35: Intuition](#page-35)
- [Page 36: Intuition](#page-36)
- [Page 37: Intuition](#page-37)
- [Page 38: Intuition](#page-38)
- [Page 39: Intuition](#page-39)
- [Page 40: Intuition](#page-40)
- [Page 41: a(4)](#page-41)
- [Page 42: Initialising weights](#page-42)
- [Page 43: (2)](#page-43)
- [Page 44: Initial thoughts on architecture](#page-44)
- [Page 45: Back Propagation Visualisation](#page-45)
- [Page 46: A few examples](#page-46)
- [Page 47: A few examples](#page-47)
- [Page 48: A few examples](#page-48)
- [Page 49: House Price Prediction](#page-49)
- [Page 50: House Price Prediction](#page-50)
- [Page 51: House Price Prediction](#page-51)
- [Page 52: Digit Recognition](#page-52)
- [Page 53: Digit Recognition](#page-53)
- [Page 54: Digit Recognition](#page-54)
- [Page 55: Image Classification](#page-55)
- [Page 56: Image classification](#page-56)
- [Page 57: Page 57](#page-57)
- [Page 58: Interesting demos](#page-58)
- [Page 59: Summary](#page-59)

---


<a id="page-1"></a>

## Page 1: Introduction to

![Original page 1](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_001.png)

### Extracted content

1 | © WMG 2026
Artificial Intelligence & Deep learning
Introduction to
Artificial Neural
Networks (ANN)
Dr Manoj Babu
2026


<a id="page-2"></a>

## Page 2: Agenda

![Original page 2](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_002.png)

### Extracted content

2 | © WMG 2026
2 | © WMG 2026
Agenda
I.
Recap of ANN, forward propagation
II.
Back-propagation
III. ANN examples
IV. Intro to CNN


<a id="page-3"></a>

## Page 3: Learning Outcomes

![Original page 3](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_003.png)

### Extracted content

3 | © WMG 2026
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
Large
Language Models (LLMs)
•
Introduction
to
Transformers.
•
Retrieval-Augmented
Generation Systems


<a id="page-4"></a>

## Page 4: ANN: Back Propagation

![Original page 4](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_004.png)

### Extracted content

4 | © WMG 2026
ANN: Back Propagation


<a id="page-5"></a>

## Page 5: PERCEPTRON & ANN

![Original page 5](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_005.png)

### Extracted content

5 | © WMG 2026
PERCEPTRON & ANN
Recap


<a id="page-6"></a>

## Page 6: What a perceptron does

![Original page 6](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_006.png)

### Extracted content

6 | © WMG 2026
What a perceptron does
𝑓
Inputs
𝑥1
𝑥2
𝑦
Processing
Output
𝑊
Decision
boundary
𝑧= 𝑤0 + x𝑇𝑊
A threshold function
+1, 𝑖𝑓 𝑤0 + x𝑇𝑊> 0
−1, 𝑖𝑓 𝑤0 + x𝑇𝑊≤0
𝑦= {+1, −1}
𝑥1, 𝑥2
𝑤2
𝑤1
𝑤0
Affine transformation
Activation


<a id="page-7"></a>

## Page 7: Perceptron to regression

![Original page 7](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_007.png)

### Extracted content

7 | © WMG 2026
Perceptron to regression
𝑓
Inputs
𝑥1
𝑥𝑛
𝑦
Processing
Output
Best fit line
𝑧= 𝑤0 + x𝑇𝑊
𝑦= [−∞, +∞]
𝑥1, 𝑥2
𝑤2
𝑤1
𝑤0
2D Sigmoid source: http://www.machinelearning.ru
+
+
+
+
+
+
+
+
Linear Activation
𝑦= 𝑧


<a id="page-8"></a>

## Page 8: Perceptron to logistic regression

![Original page 8](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_008.png)

### Extracted content

8 | © WMG 2026
Perceptron to logistic regression
𝑓
Inputs
𝑥1
𝑥𝑛
𝑦
Processing
Output
Decision
boundary
𝑧= 𝑤0 + x𝑇𝑊
A sigmoid function
1
1 + 𝑒−𝑧
𝑦= [0,1]
𝑥1, 𝑥2
𝑤2
𝑤1
𝑤0
2D Sigmoid source: http://www.machinelearning.ru
Non-linear Activation


<a id="page-9"></a>

## Page 9: Two Perceptrons

![Original page 9](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_009.png)

### Extracted content

9 | © WMG 2026
Two Perceptrons
𝑓
Inputs
𝑥1
𝑥2
𝑦= 𝑧1 + 𝑧2
Processing
Output
𝑓
𝑧1
𝑧2
𝑊1
Decision boundary
𝑊2
1
Bias not always shown


<a id="page-10"></a>

## Page 10: Notation*

![Original page 10](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_010.png)

### Extracted content

10 | © WMG 2026
Notation*
𝑓
Inputs
𝑥1
𝑥𝑛
𝑦
Processing
Output
𝑧= 𝑤0 + x𝑇𝑊
A sigmoid function
1
1 + 𝑒−𝑧
𝑦= [0,1]
𝑥1, 𝑥2
𝑤2
𝑤1
𝑤0
𝜎
𝑥1
𝑥𝑛
𝑦
𝑤2
𝑤1
𝑤0
𝜎
𝑥1
𝑦
𝑤1
𝑤0
1D
*For next few slides


<a id="page-11"></a>

## Page 11: Combining two sigmoid

![Original page 11](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_011.png)

### Extracted content

11 | © WMG 2026
Combining two sigmoid
𝑌
𝑌′
𝑥
𝜎
𝜎
ℎ1
ℎ2
Affine
(h1 + h2)
𝑞= ℎ1 + ℎ2
Not a
probability!
𝑞= 𝑊31ℎ1 + 𝑊32ℎ2 + 𝑊30
Source [1] CS109A, Protopapas, Rader, Tanner


<a id="page-12"></a>

## Page 12: Combining two sigmoid

![Original page 12](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_012.png)

### Extracted content

12 | © WMG 2026
Combining two sigmoid
𝑌
𝑌′
𝑋
𝜎(XW1)
𝜎(XW2)
ℎ1
ℎ2
𝜎(hW3)
Need to learn W1, W2 and W3.
𝑞= 𝑊31ℎ1 + 𝑊32ℎ2 + 𝑊30
𝑝=
1
1 + 𝑒−𝑞
Passing through sigmoid
yields probability
Source [1] CS109A, Protopapas, Rader, Tanner


<a id="page-13"></a>

## Page 13: Combining neurons allows us to model interesting

![Original page 13](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_013.png)

### Extracted content

13 | © WMG 2026
Combining neurons allows us to model interesting
functions
X
Y
Source [1] CS109A, Protopapas, Rader, Tanner


<a id="page-14"></a>

## Page 14: Different weights change the shape and position

![Original page 14](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_014.png)

### Extracted content

14 | © WMG 2026
Different weights change the shape and position
X
Y
Source [1] CS109A, Protopapas, Rader, Tanner


<a id="page-15"></a>

## Page 15: Neural networks can model any reasonable

![Original page 15](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_015.png)

### Extracted content

15 | © WMG 2026
Neural networks can model any reasonable
function
X
Y
Source [1] CS109A, Protopapas, Rader, Tanner


<a id="page-16"></a>

## Page 16: Adding layers allows us to model increasingly complex

![Original page 16](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_016.png)

### Extracted content

16 | © WMG 2026
Adding layers allows us to model increasingly complex
functions
16
X
Y
Source [1] CS109A, Protopapas, Rader, Tanner


<a id="page-17"></a>

## Page 17: (2)

![Original page 17](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_017.png)

### Extracted content

17 | © WMG 2026
hΘ
a0
(2)
a1
(n)
for i = 1 to n - 1
a0
(i)= 1 (added bias term)
a(i+1) = g(W(i) a(i))
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


<a id="page-18"></a>

## Page 18: (n)

![Original page 18](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_018.png)

### Extracted content

18 | © WMG 2026
hW
a1
(n)
a0
(n-1)
a1
(n-1)
a2
(n-1)
a1
(n) = g(w10
(n-1) a0
(n-1) + w11
(n-1) a1
(n-1) + w12
(n-1)a2
(n-1))
Intuition
Similar to logistic regression


<a id="page-19"></a>

## Page 19: (2)

![Original page 19](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_019.png)

### Extracted content

19 | © WMG 2026
hW
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
Intuition
new features


<a id="page-20"></a>

## Page 20: (n)

![Original page 20](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_020.png)

### Extracted content

20 | © WMG 2026
hW
a1
(n)
a0
(n-1)
a1
(n-1)
a2
(n-1)
a1
(n) = g(w10
(n-1) a0
(n-1) + w11
(n-1) a1
(n-1) + w12
(n-1)a2
(n-1))
Intuition
Significantly more complex features
vs. logistic regression


<a id="page-21"></a>

## Page 21: BACK PROPAGATION

![Original page 21](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_021.png)

### Extracted content

21 | © WMG 2026
BACK PROPAGATION


<a id="page-22"></a>

## Page 22: x =

![Original page 22](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_022.png)

### Extracted content

22 | © WMG 2026
x =
𝑥0
𝑥1
⋮𝑥𝑛

W =
w0
w1
⋮w𝑛

m training pairs
{(𝑥1 , 𝑦(1)), (𝑥2 , 𝑦(2)) , (𝑥3 , 𝑦(3)), …, (𝑥𝑚, 𝑦𝑚))}
n features; 𝑥0 = 1
n weights/parameters
Notation


<a id="page-23"></a>

## Page 23: Linear Regression Cost

![Original page 23](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_023.png)

### Extracted content

23 | © WMG 2026
Linear Regression Cost
J(W) =
1
2𝑚σ𝑖=1
𝑚(ℎ𝑤(𝑥𝑖) −𝑦(𝑖))2
ℎ𝑊𝑥= w0𝑥0 + w1𝑥1+w2𝑥2 + … + w𝑛𝑥𝑛
Vectorise:
ℎ𝑊𝑥= W𝑇𝑥


<a id="page-24"></a>

## Page 24: Linear Regression Gradient Descent

![Original page 24](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_024.png)

### Extracted content

24 | © WMG 2026
Linear Regression Gradient Descent
min
w J(W)
wj = wj - 𝛼
ⅆ
ⅆ𝑊 J(W)
wj = wj - 𝛼
1
𝑚σ𝑖=1
𝑚(ℎ𝑊(𝑥𝑖) −𝑦𝑖)𝑥𝑗
(𝑖)
= min
w
1
2𝑚෍
𝑖=1
𝑚
(ℎ𝑤(𝑥𝑖) −𝑦(𝑖))2


<a id="page-25"></a>

## Page 25: Gradient descent visualisation

![Original page 25](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_025.png)

### Extracted content

25 | © WMG 2026
Gradient descent visualisation
https://www.youtube.com/watch?v=kJgx2RcJKZY


<a id="page-26"></a>

## Page 26: Examples

![Original page 26](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_026.png)

### Extracted content

29 | © WMG 2026
Examples
https://playground.tensorflow.org/
Large learning rate
Small learning rate


<a id="page-27"></a>

## Page 27: Batch Gradient Descent

![Original page 27](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_027.png)

### Extracted content

31 | © WMG 2026
Batch Gradient Descent
• Traditional gradient descent
Also know as batch gradient descent
• Requires loop over all training examples

{(𝑥1 , 𝑦(1)), (𝑥2 , 𝑦(2)) , (𝑥3 , 𝑦(3)), …, (𝑥𝑚, 𝑦(𝑚))}
• Once all values ℎ𝑊(𝑥𝑖) computed

Wj = Wj - 𝛼σ𝑖=1
𝑚(ℎ𝑊(𝑥𝑖) −𝑦𝑖)𝑥𝑗
(𝑖)
What if m is very large? (not uncommon to be order 106)


<a id="page-28"></a>

## Page 28: Stochastic Gradient Descent

![Original page 28](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_028.png)

### Extracted content

32 | © WMG 2026
Stochastic Gradient Descent
• Update gradient at each data point
Randomly shuffle data
𝑓𝑜𝑟 𝑖 = 1: 𝑚
Wj = Wj − 𝛼(ℎ𝑊(𝑥𝑖) −𝑦𝑖)𝑥𝑗
(𝑖)


<a id="page-29"></a>

## Page 29: Stochastic Gradient Descent

![Original page 29](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_029.png)

### Extracted content

33 | © WMG 2026
Stochastic Gradient Descent
https://www.youtube.com/watch?v=bvy_B-bdT3A


<a id="page-30"></a>

## Page 30: Mini Batch Gradient Descent

![Original page 30](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_030.png)

### Extracted content

34 | © WMG 2026
Mini Batch Gradient Descent
• Hybrid stochastic and batch
• After each batch compute and apply derivative
Batch size b, training size m
for i = 1:b:m // epoch

Wj = Wj - 𝛼
1
𝑏σ𝑘=𝑖
𝑘+𝑏(ℎ𝑊(𝑥𝑘) −𝑦𝑘)𝑥𝑗
(𝑘)
Why use instead of stochastic gradient descent?


<a id="page-31"></a>

## Page 31: ANN Gradient Descent

![Original page 31](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_031.png)

### Extracted content

40 | © WMG 2026
ANN Gradient Descent
min
W J(W)
Require to compute:
J(W)
ⅆ
ⅆwij
(l) J(W)


<a id="page-32"></a>

## Page 32: (2)

![Original page 32](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_032.png)

### Extracted content

41 | © WMG 2026
hW
a0
(2)
a1
(n)
for i = 1 to n - 1
a0
(i)= 1 (added bias term)
a(i+1) = g(W(i) a(i))
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
Recap: Forward Propagation
We need to find gradient of loss
function J, wrt. each weight w


<a id="page-33"></a>

## Page 33: Intuition

![Original page 33](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_033.png)

### Extracted content

44 | © WMG 2026
Intuition
J(W) =
1
2 (ℎ𝑤(𝑥) −𝑦)2
J(W) =
1
2𝑚σ𝑖=1
𝑚(ℎ𝑤(𝑥𝑖) −𝑦(𝑖))2
hW
a
x
ⅆ
ⅆ𝑤 J(W) =
ⅆ𝐽
ⅆℎ
ⅆℎ
ⅆ𝑤 = (ℎ𝑤(𝑥) −𝑦)x
w
ℎ𝑊𝑥= 𝑤𝑥
Assume simplest ANN
-
No activation
-
Can ignore a
w = 𝑤 - 𝛼(ℎ𝑤(𝑥) −𝑦)x


<a id="page-34"></a>

## Page 34: Intuition

![Original page 34](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_034.png)

### Extracted content

45 | © WMG 2026
Intuition
J(W) =
1
2 (ℎ𝑤(𝑥) −𝑦)2
J(W) =
1
2𝑚σ𝑖=1
𝑚(ℎ𝑤(𝑥𝑖) −𝑦(𝑖))2
hW =0.3
y = 0.5
𝛼 = 0.1
a
x
= 1.5
ⅆ
ⅆ𝑤 J(W) = (0.3 −0.5)1.5
w = 0.2
ℎ𝑊𝑥= 𝑤𝑥 = 1.5 * 0.2 = 0.3
Assume simplest ANN
-
No activation
-
Can ignore a
w = 0.2 - 𝛼(−0.3) = 0.23


<a id="page-35"></a>

## Page 35: Intuition

![Original page 35](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_035.png)

### Extracted content

46 | © WMG 2026
Intuition
J(W) =
1
2 (ℎ𝑤(𝑥) −𝑦)2
J(W) =
1
2𝑚σ𝑖=1
𝑚(ℎ𝑤(𝑥𝑖) −𝑦(𝑖))2
hW =0.3
y = 0.5
𝛼 = 0.1
a
x
= 1.5
ⅆ
ⅆ𝑤 J(W) = (0.3 −0.5)1.5
w = 0.2
ℎ𝑊𝑥= 𝑤𝑥 = 1.5 * 0.2 = 0.3
w = 0.2 - 𝛼(−0.3) = 0.23
J(W)
ℎ𝑊𝑥


<a id="page-36"></a>

## Page 36: Intuition

![Original page 36](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_036.png)

### Extracted content

47 | © WMG 2026
Intuition
hW =0.3
y = 0.5
𝛼 = 0.1
a
x
= 1.5
w = 0.2
Assume simplest ANN
-
No activation
-
Can ignore a


<a id="page-37"></a>

## Page 37: Intuition

![Original page 37](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_037.png)

### Extracted content

48 | © WMG 2026
Intuition
hW =0.3
y = 0.5
𝛼 = 0.01
a
x
= 1.5
w = 0.2
Assume simplest ANN
-
No activation
-
Can ignore a
𝛼 too small


<a id="page-38"></a>

## Page 38: Intuition

![Original page 38](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_038.png)

### Extracted content

49 | © WMG 2026
Intuition
hW =0.3
y = 0.5
𝛼 = 1.5
a
x
= 1.5
w = 0.2
Assume simplest ANN
-
No activation
-
Can ignore a
𝛼 too large


<a id="page-39"></a>

## Page 39: Intuition

![Original page 39](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_039.png)

### Extracted content

50 | © WMG 2026
Intuition
hW
a
x
w
Assume simplest ANN
-
No activation
-
Can ignore a


<a id="page-40"></a>

## Page 40: Intuition

![Original page 40](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_040.png)

### Extracted content

51 | © WMG 2026
Intuition
hW
a
x
w
Assume simplest ANN
-
No activation
-
Can ignore a


<a id="page-41"></a>

## Page 41: a(4)

![Original page 41](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_041.png)

### Extracted content

52 | © WMG 2026
a(4)
a(3)
w(3)
a(2)
w(2)
a(1)
w(1)
ⅆ
ⅆ𝑤(3) J(W) =
ⅆ𝐽
ⅆa(4)
ⅆa(4)
ⅆw(3) = (a(4) −𝑦)a(3)
w(3)= w(3)- 𝛼(a(4) −𝑦)a(3)
ⅆ𝐽
ⅆa(4)
ⅆa(4)
ⅆa(3)
ⅆa(3)
ⅆw(2) = w(3)(a(4) −𝑦)a(2)
w(2)= w(2)- 𝛼w(3)(a(4) −𝑦)a(2)
…
w(1)= w(1)- 𝛼w(2)w(3)(a(4) −𝑦)a(3)
Intuition


<a id="page-42"></a>

## Page 42: Initialising weights

![Original page 42](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_042.png)

### Extracted content

53 | © WMG 2026
Initialising weights
•
What would happen if all weights equal?
•
On initialisation
•
For example 0
•
Randomise weight initialisation


<a id="page-43"></a>

## Page 43: (2)

![Original page 43](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_043.png)

### Extracted content

54 | © WMG 2026
hW
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
Initialising weights


<a id="page-44"></a>

## Page 44: Initial thoughts on architecture

![Original page 44](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_044.png)

### Extracted content

55 | © WMG 2026
Initial thoughts on architecture
•
Input – size of features
•
Output – size of classification/regression
•
Hidden layers
•
Try one layer first
•
More is better
•
Although more computational cost
•
Similar sized hidden layers


<a id="page-45"></a>

## Page 45: Back Propagation Visualisation

![Original page 45](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_045.png)

### Extracted content

56 | © WMG 2026
Back Propagation Visualisation
https://www.youtube.com/watch?v=Ilg3gGewQ5U


<a id="page-46"></a>

## Page 46: A few examples

![Original page 46](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_046.png)

### Extracted content

57 | © WMG 2026
A few examples
•
Quick note – fully connected
Image from http://neuralnetworksanddeeplearning.com/chap6.html


<a id="page-47"></a>

## Page 47: A few examples

![Original page 47](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_047.png)

### Extracted content

58 | © WMG 2026
A few examples
•
Quick note – fully connected
x1
x2
x3
x4
x5
x6
x7
x8
y1
y2
y3
y4
Image from http://neuralnetworksanddeeplearning.com/chap6.html


<a id="page-48"></a>

## Page 48: A few examples

![Original page 48](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_048.png)

### Extracted content

59 | © WMG 2026
A few examples
•
Quick note – fully connected
x1
x2
x3
x4
x5
x6
x7
x8
y1
y2
y3
y4
9 nodes
9 nodes
9 nodes


<a id="page-49"></a>

## Page 49: House Price Prediction

![Original page 49](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_049.png)

### Extracted content

60 | © WMG 2026
House Price Prediction
x1 size
x2 no. of
bedrooms
x3 postcode
x4 schools
x5 population density
h(x) price
16 nodes


<a id="page-50"></a>

## Page 50: House Price Prediction

![Original page 50](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_050.png)

### Extracted content

61 | © WMG 2026
House Price Prediction
x1 size
x2 no. of
bedrooms
x3 postcode
x4 schools
x5 population density
h(x) price
32 nodes


<a id="page-51"></a>

## Page 51: House Price Prediction

![Original page 51](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_051.png)

### Extracted content

62 | © WMG 2026
House Price Prediction
x1 size
x2 no. of
bedrooms
x3 postcode
x4 schools
x5 population density
h(x) price
16 nodes
8 nodes
16 nodes


<a id="page-52"></a>

## Page 52: Digit Recognition

![Original page 52](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_052.png)

### Extracted content

65 | © WMG 2026
Digit Recognition
MINST dataset – handwritten characters
https://ml4a.github.io/ml4a/neural_networks/


<a id="page-53"></a>

## Page 53: Digit Recognition

![Original page 53](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_053.png)

### Extracted content

66 | © WMG 2026
Digit Recognition
https://ml4a.github.io/ml4a/neural_networks/


<a id="page-54"></a>

## Page 54: Digit Recognition

![Original page 54](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_054.png)

### Extracted content

67 | © WMG 2026
Digit Recognition
64 nodes
64 nodes
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


<a id="page-55"></a>

## Page 55: Image Classification

![Original page 55](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_055.png)

### Extracted content

68 | © WMG 2026
Image Classification
•
CIFAR10
•
32x32 images
•
10 classes
•
6000 images
•
Per class


<a id="page-56"></a>

## Page 56: Image classification

![Original page 56](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_056.png)

### Extracted content

69 | © WMG 2026
Image classification
512 nodes
512 nodes
512 nodes
plane
car
bird
cat
deer
dog
frog
horse
ship
truck


<a id="page-57"></a>

## Page 57: Page 57

![Original page 57](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_057.png)

### Extracted content

70 | © WMG 2026


<a id="page-58"></a>

## Page 58: Interesting demos

![Original page 58](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_058.png)

### Extracted content

71 | © WMG 2026
Interesting demos
• https://jalammar.github.io/visual-interactive-guide-
basics-neural-networks/
• http://playground.tensorflow.org
• https://www.cs.ryerson.ca/~aharley/vis/conv/ - Number
visualisation


<a id="page-59"></a>

## Page 59: Summary

![Original page 59](assets/WMG_AIDL_Week_1_Block_2_ANN_Backpropagation/page_059.png)

### Extracted content

72 | © WMG 2026
Summary
•
Overview of progress
•
Features and image classification
•
Imagenet
•
Neural networks
•
Neural model
Questions?
