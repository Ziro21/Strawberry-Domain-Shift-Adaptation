# WMG AIDL Block 2.1

> Upgraded Markdown conversion. The original page image is included for every page so diagrams, screenshots, tables, slide layout and visual details are preserved. Extracted text is also provided below each page for searchable study notes.

## Contents

- [Page 1: Introduction to](#page-1)
- [Page 2: Learning Outcomes](#page-2)
- [Page 3: Agenda](#page-3)
- [Page 4: Recap](#page-4)
- [Page 5: CNN](#page-5)
- [Page 6: Motivation](#page-6)
- [Page 7: Image](#page-7)
- [Page 8: Image Representation](#page-8)
- [Page 9: Using ANNs (Fully Connected)](#page-9)
- [Page 10: Fully Connected](#page-10)
- [Page 11: Feature Selection](#page-11)
- [Page 12: Convolutions](#page-12)
- [Page 13: Automated Feature Extraction](#page-13)
- [Page 14: Automated Feature Extraction](#page-14)
- [Page 15: Basic Convolution](#page-15)
- [Page 16: Basic Convolution](#page-16)
- [Page 17: Weights calculation](#page-17)
- [Page 18: https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca](#page-18)
- [Page 19: Stride](#page-19)
- [Page 20: https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca](#page-20)
- [Page 21: Stride (2) Convolution 3x3](#page-21)
- [Page 22: Padding](#page-22)
- [Page 23: https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca](#page-23)
- [Page 24: Padding](#page-24)
- [Page 25: https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca](#page-25)
- [Page 26: Dumoulin & Visin (2018) A guide to convolution arithmetic for deep learning](#page-26)
- [Page 27: Questions](#page-27)
- [Page 28: Calculating output of next layer](#page-28)
- [Page 29: Dilated Convolutions](#page-29)
- [Page 30: Pooling](#page-30)
- [Page 31: Convolutional Layers](#page-31)
- [Page 32: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53](#page-32)
- [Page 33: Convolutional Layers](#page-33)
- [Page 34: Convolutional Layers](#page-34)
- [Page 35: Convolutional Layers](#page-35)
- [Page 36: Weights calculation](#page-36)
- [Page 37: Weights calculation](#page-37)
- [Page 38: Max Pooling](#page-38)
- [Page 39: Flattening](#page-39)
- [Page 40: Flattening](#page-40)
- [Page 41: Putting it all together](#page-41)
- [Page 42: Putting it all together](#page-42)
- [Page 43: Predicting Beauty of Scene](#page-43)
- [Page 44: Predicting Beauty of Scene](#page-44)
- [Page 45: Source: Seresinhe CI, Preis T, Moat HS. 2017 Using deep learning to quantify the beauty of outdoor places.R.](#page-45)
- [Page 46: Image Classification](#page-46)
- [Page 47: Image classification ANN](#page-47)
- [Page 48: Image classification CNN](#page-48)
- [Page 49: Marnerides, D., Bashford‐Rogers, T., Hatchett, J., & Debattista, K. (2018, May). ExpandNet: A deep convolutional neural network for](#page-49)
- [Page 50: Deep Colorization](#page-50)
- [Page 51: Advancing Understanding](#page-51)
- [Page 52: Instancing Segmentation](#page-52)
- [Page 53: High Resolution Synthesis](#page-53)
- [Page 54: CNNs – Key takeaways](#page-54)
- [Page 55: Demos](#page-55)
- [Page 56: QUESTIONS?](#page-56)

---


<a id="page-1"></a>

## Page 1: Introduction to

![Original page 1](assets/WMG_AIDL_Block_2.1/page_001.png)

### Extracted content

1 | © WMG 2026
Artificial Intelligence & Deep learning
Introduction to
Artificial Neural
Networks (ANN)
Dr Manoj Babu
2026


<a id="page-2"></a>

## Page 2: Learning Outcomes

![Original page 2](assets/WMG_AIDL_Block_2.1/page_002.png)

### Extracted content

2 | © WMG 2026
2 | © WMG 2026
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


<a id="page-3"></a>

## Page 3: Agenda

![Original page 3](assets/WMG_AIDL_Block_2.1/page_003.png)

### Extracted content

3 | © WMG 2026
3 | © WMG 2026
Agenda
I.
Image fundamentals
II.
Intro to CNN
III. CNN architecture fundamentals
IV. CNN examples


<a id="page-4"></a>

## Page 4: Recap

![Original page 4](assets/WMG_AIDL_Block_2.1/page_004.png)

### Extracted content

4 | © WMG 2026
Recap
• ANN
• Notations and architecture
• Forward propagation
• Examples
• Back propagation intuition


<a id="page-5"></a>

## Page 5: CNN

![Original page 5](assets/WMG_AIDL_Block_2.1/page_005.png)

### Extracted content

5 | © WMG 2026
CNN
Slides credit: Dr. Demetris Marnerides, Prof. Kurt Debattista


<a id="page-6"></a>

## Page 6: Motivation

![Original page 6](assets/WMG_AIDL_Block_2.1/page_006.png)

### Extracted content

6 | © WMG 2026
Motivation
•
CNNs
•
Predominant in CV
•
Why use CNNs?
By Aphex34 - Own work, CC BY-SA 4.0,
https://commons.wikimedia.org/w/index.php?curid=45679374


<a id="page-7"></a>

## Page 7: Image

![Original page 7](assets/WMG_AIDL_Block_2.1/page_007.png)

### Extracted content

7 | © WMG 2026
Image
• 2-dimensional array
• Buffer is moved to video memory
• Displayed
http://pippin.gimp.org/image_processing/chap_dir.html


<a id="page-8"></a>

## Page 8: Image Representation

![Original page 8](assets/WMG_AIDL_Block_2.1/page_008.png)

### Extracted content

8 | © WMG 2026
Image Representation
•
Array of pixel values
•
RGB values
•
For colour display
•
Typically each colour
•
Represented by byte
•
256 possible values (pixel depth)
"Rgb-raster-image" by Gringer - Own work. Licensed under CC0 via Wikimedia Commons -
http://commons.wikimedia.org/wiki/File:Rgb-raster-image.svg#mediaviewer/File:Rgb-raster-image.svg


<a id="page-9"></a>

## Page 9: Using ANNs (Fully Connected)

![Original page 9](assets/WMG_AIDL_Block_2.1/page_009.png)

### Extracted content

9 | © WMG 2026
Using ANNs (Fully Connected)
What about full HD image?


<a id="page-10"></a>

## Page 10: Fully Connected

![Original page 10](assets/WMG_AIDL_Block_2.1/page_010.png)

### Extracted content

10 | © WMG 2026
Fully Connected
• Full HD
• 1920 * 1080 * 3 = 6,220,800
• Connect to layer of 1,024
• Weights?
• 6,370,100,224
• At 4K with 2,048 hidden layer
• 50,960,795,648
• Impractical in space and time
https://www.pinterest.co.uk/pin/583990276651302060/?lp=true


<a id="page-11"></a>

## Page 11: Feature Selection

![Original page 11](assets/WMG_AIDL_Block_2.1/page_011.png)

### Extracted content

12 | © WMG 2026
Feature Selection
Satýlmýs, P., Bashford-Rogers, T., Chalmers, A., & Debattista, K. (2016). A machine-
learning driven sky model. IEEE computer graphics and applications, 37(1), 80-91
Dalal, N., & Triggs, B. (2005, June). Histograms of oriented gradients for human
detection. In 2005 IEEE computer society conference on computer vision and pattern
recognition (CVPR'05) (Vol. 1, pp. 886-893). IEEE.


<a id="page-12"></a>

## Page 12: Convolutions

![Original page 12](assets/WMG_AIDL_Block_2.1/page_012.png)

### Extracted content

13 | © WMG 2026
Convolutions
•
Likewise, for 2D vectors (matrices, images)
𝑓𝑖𝑙𝑡𝑒𝑟=
1
0
1
0
1
0
1
0
1


<a id="page-13"></a>

## Page 13: Automated Feature Extraction

![Original page 13](assets/WMG_AIDL_Block_2.1/page_013.png)

### Extracted content

15 | © WMG 2026
Automated Feature Extraction
https://towardsdatascience.com/cnn-application-on-structured-data-automated-feature-extraction-8f2cd28d9a7e
The network learns the elements of the filter + weights of the NN


<a id="page-14"></a>

## Page 14: Automated Feature Extraction

![Original page 14](assets/WMG_AIDL_Block_2.1/page_014.png)

### Extracted content

16 | © WMG 2026
Automated Feature Extraction
https://towardsdatascience.com/cnn-application-on-structured-data-automated-feature-extraction-8f2cd28d9a7e
The network learns the elements of the filter + weights of the NN


<a id="page-15"></a>

## Page 15: Basic Convolution

![Original page 15](assets/WMG_AIDL_Block_2.1/page_015.png)

### Extracted content

20 | © WMG 2026
Basic Convolution
https://indoml.com/2018/03/07/student-notes-convolutional-neural-networks-cnn-introduction/


<a id="page-16"></a>

## Page 16: Basic Convolution

![Original page 16](assets/WMG_AIDL_Block_2.1/page_016.png)

### Extracted content

21 | © WMG 2026
Basic Convolution


<a id="page-17"></a>

## Page 17: Weights calculation

![Original page 17](assets/WMG_AIDL_Block_2.1/page_017.png)

### Extracted content

22 | © WMG 2026
Weights calculation
•
Calculate number of parameters
•
3 x 3 kernel
•
1 input channel
•
1 Output channel:
•
(3 x 3) (kernel) + 1 bias
•
Total: 10


<a id="page-18"></a>

## Page 18: https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca

![Original page 18](assets/WMG_AIDL_Block_2.1/page_018.png)

### Extracted content

27 | © WMG 2026
https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca
Convolution 3x3


<a id="page-19"></a>

## Page 19: Stride

![Original page 19](assets/WMG_AIDL_Block_2.1/page_019.png)

### Extracted content

28 | © WMG 2026
Stride
https://indoml.com/2018/03/07/student-notes-convolutional-neural-networks-cnn-introduction/


<a id="page-20"></a>

## Page 20: https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca

![Original page 20](assets/WMG_AIDL_Block_2.1/page_020.png)

### Extracted content

29 | © WMG 2026
https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca
Convolution 3x3


<a id="page-21"></a>

## Page 21: Stride (2) Convolution 3x3

![Original page 21](assets/WMG_AIDL_Block_2.1/page_021.png)

### Extracted content

30 | © WMG 2026
Stride (2) Convolution 3x3
https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca


<a id="page-22"></a>

## Page 22: Padding

![Original page 22](assets/WMG_AIDL_Block_2.1/page_022.png)

### Extracted content

31 | © WMG 2026
Padding
https://indoml.com/2018/03/07/student-notes-convolutional-neural-networks-cnn-introduction/


<a id="page-23"></a>

## Page 23: https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca

![Original page 23](assets/WMG_AIDL_Block_2.1/page_023.png)

### Extracted content

32 | © WMG 2026
https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca
Convolution 3x3


<a id="page-24"></a>

## Page 24: Padding

![Original page 24](assets/WMG_AIDL_Block_2.1/page_024.png)

### Extracted content

33 | © WMG 2026
Padding
https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca


<a id="page-25"></a>

## Page 25: https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca

![Original page 25](assets/WMG_AIDL_Block_2.1/page_025.png)

### Extracted content

34 | © WMG 2026
https://towardsdatascience.com/convolutional-neural-networks-for-beginners-practical-guide-with-python-and-keras-dc688ea90dca


<a id="page-26"></a>

## Page 26: Dumoulin & Visin (2018) A guide to convolution arithmetic for deep learning

![Original page 26](assets/WMG_AIDL_Block_2.1/page_026.png)

### Extracted content

35 | © WMG 2026
Dumoulin & Visin (2018) A guide to convolution arithmetic for deep learning
What are the
properties of
this
convolution
operation?


<a id="page-27"></a>

## Page 27: Questions

![Original page 27](assets/WMG_AIDL_Block_2.1/page_027.png)

### Extracted content

36 | © WMG 2026
Questions
•
5x5 input, 3x3 kernel, stride 1
•
5x5 input, 3x3 kernel, stride 1, padding 1
•
5x5 input, 5x5 kernel, stride 1
•
10x10 input, 5x5 kernel, stride 1, padding 2
•
10x10 input, 3x3 kernel, stride 2, padding 1


<a id="page-28"></a>

## Page 28: Calculating output of next layer

![Original page 28](assets/WMG_AIDL_Block_2.1/page_028.png)

### Extracted content

37 | © WMG 2026
Calculating output of next layer
•
𝑙 = layer
•
𝑝 = padding
•
𝑓 = kernel size
•
𝑠 = stride
𝑛(𝑙) = 𝑛(𝑙−1) + 2𝑝(𝑙−1) −𝑓(𝑙)
𝑠(𝑙)
+ 1


<a id="page-29"></a>

## Page 29: Dilated Convolutions

![Original page 29](assets/WMG_AIDL_Block_2.1/page_029.png)

### Extracted content

38 | © WMG 2026
Dilated Convolutions


<a id="page-30"></a>

## Page 30: Pooling

![Original page 30](assets/WMG_AIDL_Block_2.1/page_030.png)

### Extracted content

39 | © WMG 2026
Pooling
https://indoml.com/2018/03/07/student-notes-convolutional-neural-networks-cnn-introduction/


<a id="page-31"></a>

## Page 31: Convolutional Layers

![Original page 31](assets/WMG_AIDL_Block_2.1/page_031.png)

### Extracted content

40 | © WMG 2026
Convolutional Layers
https://indoml.com/2018/03/07/student-notes-convolutional-neural-networks-cnn-introduction/


<a id="page-32"></a>

## Page 32: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

![Original page 32](assets/WMG_AIDL_Block_2.1/page_032.png)

### Extracted content

41 | © WMG 2026
https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
Multi-channel convolution


<a id="page-33"></a>

## Page 33: Convolutional Layers

![Original page 33](assets/WMG_AIDL_Block_2.1/page_033.png)

### Extracted content

42 | © WMG 2026
Convolutional Layers
https://indoml.com/2018/03/07/student-notes-convolutional-neural-networks-cnn-introduction/


<a id="page-34"></a>

## Page 34: Convolutional Layers

![Original page 34](assets/WMG_AIDL_Block_2.1/page_034.png)

### Extracted content

43 | © WMG 2026
Convolutional Layers
https://indoml.com/2018/03/07/student-notes-convolutional-neural-networks-cnn-introduction/


<a id="page-35"></a>

## Page 35: Convolutional Layers

![Original page 35](assets/WMG_AIDL_Block_2.1/page_035.png)

### Extracted content

44 | © WMG 2026
Convolutional Layers
•
Expressing it more simply
https://indoml.com/2018/03/07/student-notes-convolutional-neural-networks-cnn-introduction/


<a id="page-36"></a>

## Page 36: Weights calculation

![Original page 36](assets/WMG_AIDL_Block_2.1/page_036.png)

### Extracted content

45 | © WMG 2026
Weights calculation
•
Calculate number of parameters
•
3 x 3 kernel
•
Grey scale input
•
Output: 4 Channels
•
(3 x 3) (kernel) x 4 (output) = 36
•
Bias = 4
•
Total: 40


<a id="page-37"></a>

## Page 37: Weights calculation

![Original page 37](assets/WMG_AIDL_Block_2.1/page_037.png)

### Extracted content

46 | © WMG 2026
Weights calculation
•
Calculate number of parameters
•
3 x 3 kernel
•
RGB input
•
Output 8
•
(3 x 3) (kernel) x 3(depth RGB) x 8 (output) = 216
•
Bias = 8
•
Total: 224


<a id="page-38"></a>

## Page 38: Max Pooling

![Original page 38](assets/WMG_AIDL_Block_2.1/page_038.png)

### Extracted content

47 | © WMG 2026
Max Pooling
https://indoml.com/2018/03/07/student-notes-convolutional-neural-networks-cnn-introduction/


<a id="page-39"></a>

## Page 39: Flattening

![Original page 39](assets/WMG_AIDL_Block_2.1/page_039.png)

### Extracted content

49 | © WMG 2026
Flattening
https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-step-3-flattening


<a id="page-40"></a>

## Page 40: Flattening

![Original page 40](assets/WMG_AIDL_Block_2.1/page_040.png)

### Extracted content

50 | © WMG 2026
Flattening
•
Input to Fully Connected Network
https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-step-4-full-connection


<a id="page-41"></a>

## Page 41: Putting it all together

![Original page 41](assets/WMG_AIDL_Block_2.1/page_041.png)

### Extracted content

51 | © WMG 2026
Putting it all together
•
Classifying a digit


<a id="page-42"></a>

## Page 42: Putting it all together

![Original page 42](assets/WMG_AIDL_Block_2.1/page_042.png)

### Extracted content

52 | © WMG 2026
Putting it all together
https://indoml.com/2018/03/07/student-notes-convolutional-neural-networks-cnn-introduction/
Interactive Demo: https://poloclub.github.io/cnn-explainer/


<a id="page-43"></a>

## Page 43: Predicting Beauty of Scene

![Original page 43](assets/WMG_AIDL_Block_2.1/page_043.png)

### Extracted content

53 | © WMG 2026
Predicting Beauty of Scene
Source: Seresinhe CI, Preis T, Moat HS. 2017 Using deep learning to quantify the beauty of outdoor places.R.
Soc. open sci. 4: 170170. http://dx.doi.org/10.1098/rsos.170170


<a id="page-44"></a>

## Page 44: Predicting Beauty of Scene

![Original page 44](assets/WMG_AIDL_Block_2.1/page_044.png)

### Extracted content

54 | © WMG 2026
Predicting Beauty of Scene
Source: Seresinhe CI, Preis T, Moat HS. 2017 Using deep learning to quantify the beauty of outdoor places.R.
Soc. open sci. 4: 170170. http://dx.doi.org/10.1098/rsos.170170


<a id="page-45"></a>

## Page 45: Source: Seresinhe CI, Preis T, Moat HS. 2017 Using deep learning to quantify the beauty of outdoor places.R.

![Original page 45](assets/WMG_AIDL_Block_2.1/page_045.png)

### Extracted content

55 | © WMG 2026
Source: Seresinhe CI, Preis T, Moat HS. 2017 Using deep learning to quantify the beauty of outdoor places.R.
Soc. open sci. 4: 170170. http://dx.doi.org/10.1098/rsos.170170


<a id="page-46"></a>

## Page 46: Image Classification

![Original page 46](assets/WMG_AIDL_Block_2.1/page_046.png)

### Extracted content

56 | © WMG 2026
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


<a id="page-47"></a>

## Page 47: Image classification ANN

![Original page 47](assets/WMG_AIDL_Block_2.1/page_047.png)

### Extracted content

57 | © WMG 2026
Image classification ANN
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


<a id="page-48"></a>

## Page 48: Image classification CNN

![Original page 48](assets/WMG_AIDL_Block_2.1/page_048.png)

### Extracted content

58 | © WMG 2026
Image classification CNN
k: 3x3
s: 2
p: 1
f: 8
16x16, depth 8
Max Pool
f: 2
s: 2
k: 3x3
s: 2
p: 1
f: 32
4x4, depth 32
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
32x32, depth 3


<a id="page-49"></a>

## Page 49: Marnerides, D., Bashford‐Rogers, T., Hatchett, J., & Debattista, K. (2018, May). ExpandNet: A deep convolutional neural network for

![Original page 49](assets/WMG_AIDL_Block_2.1/page_049.png)

### Extracted content

59 | © WMG 2026
Marnerides, D., Bashford‐Rogers, T., Hatchett, J., & Debattista, K. (2018, May). ExpandNet: A deep convolutional neural network for
high dynamic range expansion from low dynamic range content. In Computer Graphics Forum (Vol. 37, No. 2, pp. 37-49).
LDR to HDR


<a id="page-50"></a>

## Page 50: Deep Colorization

![Original page 50](assets/WMG_AIDL_Block_2.1/page_050.png)

### Extracted content

60 | © WMG 2026
Deep Colorization
Cheng et al., 2016
Globally and Locally Consistent
Image Completion
Iizuka et al., 2017
Inpainting


<a id="page-51"></a>

## Page 51: Advancing Understanding

![Original page 51](assets/WMG_AIDL_Block_2.1/page_051.png)

### Extracted content

61 | © WMG 2026
Advancing Understanding
•
Machine Learning
Karpathy, A., & Fei-Fei, L. (2015). Deep visual-semantic
alignments for generating image descriptions. In Proceedings of
the IEEE Conference on Computer Vision and Pattern
Recognition (pp. 3128-3137).


<a id="page-52"></a>

## Page 52: Instancing Segmentation

![Original page 52](assets/WMG_AIDL_Block_2.1/page_052.png)

### Extracted content

62 | © WMG 2026
Instancing Segmentation
•
For Automotive
https://www.youtube.com/watch?v=AbXzU9ZzqF4


<a id="page-53"></a>

## Page 53: High Resolution Synthesis

![Original page 53](assets/WMG_AIDL_Block_2.1/page_053.png)

### Extracted content

63 | © WMG 2026
High Resolution Synthesis
https://www.youtube.com/watch?v=S1OwOd-war8


<a id="page-54"></a>

## Page 54: CNNs – Key takeaways

![Original page 54](assets/WMG_AIDL_Block_2.1/page_054.png)

### Extracted content

64 | © WMG 2026
CNNs – Key takeaways
•
Parameter sharing
•
Saving time and space
•
Automated feature extraction
•
Practical method for many applications


<a id="page-55"></a>

## Page 55: Demos

![Original page 55](assets/WMG_AIDL_Block_2.1/page_055.png)

### Extracted content

65 | © WMG 2026
Demos
• https://poloclub.github.io/cnn-explainer/


<a id="page-56"></a>

## Page 56: QUESTIONS?

![Original page 56](assets/WMG_AIDL_Block_2.1/page_056.png)

### Extracted content

66 | © WMG 2026
QUESTIONS?
