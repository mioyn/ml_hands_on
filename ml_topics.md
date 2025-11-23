
# ✅ **1. Fundamentals of Machine Learning**


## **1.1 What is Machine Learning?**

Machine Learning (ML) is a field of artificial intelligence where computer systems **learn patterns from data** instead of being explicitly programmed.

**Key idea:**
The model improves performance on a task **T** based on experience **E** measured by performance metric **P**.

**Example:**

* Gmail spam filter learns from labeled email examples.
* A model predicts house prices from historical data.

---

## **1.2 Supervised vs Unsupervised Learning**

### **Supervised Learning**

* Data contains **input X** and **correct output Y**.
* The model learns a mapping **X → Y**.

**Examples**

* Classifying emails as *spam/not spam*
* Predicting house price (a number)
* Diagnosing diseases from medical images

### **Unsupervised Learning**

* Only **input X**, no labels Y.
* Model discovers hidden structure.

**Examples**

* Customer segmentation (clusters)
* Dimensionality reduction (PCA)
* Anomaly detection

**Key difference:**
Supervised uses labeled data; unsupervised finds structure on its own.

---

## **1.3 Classification vs Regression**

### **Classification**

Outputs **categories** (discrete classes).
Examples:

* Predict dog vs cat
* Predict sentiment (positive/negative)
* Predict digit (0–9)

### **Regression**

Outputs **numbers** (continuous).
Examples:

* Predict temperature
* Predict stock price
* Predict age from a photograph

---

## **1.4 Identify Classification and Regression Tasks**

| Task                                              | Type           |
| ------------------------------------------------- | -------------- |
| Predict height                                    | Regression     |
| Predict whether customer will cancel subscription | Classification |
| Predict next day’s rainfall amount (mm)           | Regression     |
| Predict if a tumor is malignant                   | Classification |

---

# ✅ **2. Linear & Logistic Regression**

---

## **2.1 Linear Regression**

Predicts a **continuous** variable.

### **Model equation**

$
\hat{y} = w_0 + w_1 x
$

For multiple features:
$
\hat{y} = w_0 + \sum_{i=1}^{n} w_i x_i
$

### **Example**

Predicting house price from size.

---

## **2.2 Logistic Regression**

Used for **binary classification**.

### **Model equation**

$
p = \sigma(w_0 + w_1 x)
$
where
$
\sigma(z) = \frac{1}{1 + e^{-z}}
$

Output is a **probability between 0 and 1**.

---

## **2.3 Other ML Methods (3 examples)**

* **Decision Trees**
* **Random Forests**
* **Support Vector Machines (SVM)**

---

## **2.4 Describe one supervised model**

### **Random Forests**

An ensemble of many decision trees. Each tree votes, and the forest aggregates predictions.
Advantages: robust, handles nonlinearity, reduces overfitting.

---

# ✅ **3. Feature Engineering**

---

## **3.1 What is Feature Engineering?**

The process of transforming raw data into meaningful features that help ML models learn better.

### **Why important?**

Better features ⇒ simpler models ⇒ higher accuracy.

---

## **3.2 Three Feature Engineering Methods**

* **Normalization/Standardization**
* **One-Hot Encoding**
* **Polynomial features**

---

# ✅ **4. Multi-Layer Perceptrons (MLP)**

---

## **4.1 Draw & Annotate an MLP**

```
Input Layer     Hidden Layer         Output Layer
   x1  x2    →   ● ---- w ----→        ○
                 ● ---- w ----→        ○
                 ● ---- w ----→
```

* Nodes represent neurons
* Lines represent weights
* Each neuron: linear combination + activation

---

## **4.2 Manual Computation Example**

**Given:**
Input: (x_1 = 1, x_2 = 2)
Weights:

Hidden node h1:
$
h1 = x_1(2) + x_2(3) = 1*2 + 2*3 = 8
$

Hidden node h2:
$
h2 = x_1(1) + x_2(1) = 1 + 2 = 3
$

Output neuron:
$
y = h1(1) + h2(2) = 8*1 + 3*2 = 14
$

---

## **4.3 Linear vs Nonlinear Models**

* **Linear**: only weighted sums (no nonlinear activation).
* **Nonlinear**: uses activation functions → can model complex patterns.

---

## **4.4 Activation Functions (ASCII)**

### Step

```
y=0 for x<0
y=1 for x>=0
```

### Sigmoid

$
\sigma(x)=\frac{1}{1+e^{-x}}
$

### tanH

Range: (-1, 1)

### ReLU

$
f(x)=\max(0,x)
$

---

## **4.5 Softmax**

Converts logits into **probabilities** that sum to 1.

$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$

---

## **4.6 Parameters in Dense Layer**

$
\text{Parameters} = (input_dim+1) \times output_dim
$

Example: 10 inputs → 5 neurons
Parameters = (10 + 1) × 5 = 55

---

## **4.7 Advantages of TensorFlow/PyTorch**

* GPU acceleration
* Automatic differentiation
* Pretrained models
* High-level APIs
* Scalability

---

## **4.8 ANN Application (not CNN/LLM)**

**Fraud detection:**
ANNs detect unusual transaction patterns indicating credit card fraud.

---

## **4.9 ANN Hyperparameters**

* Learning rate
* Hidden layers
* Batch size
* Activation function
* Optimizer

---

## **4.10 John Hopfield & Geoffrey Hinton**

* **Hopfield:** invented Hopfield Networks (associative memory).
* **Hinton:** father of deep learning; invented backpropagation improvements, RBMs.

---

# ✅ **5. Training Curves & Generalization**

---

## **5.1 Ideal Learning Curve**

Error decreases smoothly during training.

```
Loss
|\
| \
|  \
|   \______
|___________ Epochs
```

---

## **5.2 Accuracy Curve**

Accuracy increases until plateau.

---

## **5.3 Interpret Curves**

* Smooth decrease: good learning
* Divergence: high learning rate
* Training ↓ but validation ↑ = overfitting

---

## **5.4 Why Scaling Matters**

Unscaled data → gradients explode or vanish → slow convergence.

---

## **5.5 Overfitting & Underfitting**

* **Overfitting:** memorizes training data
* **Underfitting:** model too simple

---

## **5.6 Why ANNs Overfit**

Too many parameters relative to data.

---

## **5.7 MSE**

$
MSE = \frac{1}{n}\sum(y-\hat{y})^{2}
$

Used for regression.

---

## **5.8 Binary Cross-Entropy**

$
L = -[y\log(p) + (1-y)\log(1-p)]
$

Used for binary classification.

---

## **5.9 Regularization**

Techniques to reduce overfitting.

**Three methods:**

* Dropout
* L2 weight decay
* Early stopping

---

# ✅ **6. Datasets & Evaluation**

---

## **6.1 Training / Validation / Test**

* **Training**: learn weights
* **Validation**: tune hyperparameters
* **Test**: final unbiased evaluation

---

## **6.2 Data Leakage**

Using test data information during training → artificially good performance.

---

## **6.3 Metrics**

**Classification:** accuracy, F1, precision, recall
**Regression:** MSE, MAE, R²

---

## **6.4 Confusion Matrix Calculations**

### Example matrix

|        | Pred 1 | Pred 0 |
| ------ | ------ | ------ |
| True 1 | 50     | 10     |
| True 0 | 5      | 100    |

* TP = 50
* FN = 10
* FP = 5
* TN = 100

### Accuracy

$
\frac{TP+TN}{TP+TN+FP+FN} = \frac{150}{165}=0.91
$

---

## **6.5 Accuracy vs BCE**

**Accuracy:** easy to understand but ignores probability confidence.
**BCE:** considers confidence; better for optimization.

---

## **6.6 MAE**

$
MAE = \frac{1}{n}\sum|y-\hat{y}|
$

---

## **6.7 ROC Curve Comparison**

Better classifier = curve closer to top-left.

---

## **6.8 Model Drift**

When data distribution changes over time → model degrades.

---

## **6.9 LLM Evaluation Method**

**Human preference evaluation** (A/B testing).

---

# ✅ **7. Optimization & Backpropagation**

---

## **7.1 Gradient Descent Equation**

$
w = w - \eta \frac{\partial L}{\partial w}
$

---

## **7.2 Why 2nd Derivative Won’t Work**

Loss surface is high-dimensional, non-convex, no closed-form solution.

---

## **7.3 What Happens in One Epoch**

* Forward pass
* Loss computed
* Backpropagation
* Weight update

---

## **7.4 Why Random Initialization Instead of Zero**

Zero initialization makes all neurons identical → no learning.

---

## **7.5 Learning Rate Too High/Low**

* Too high → divergence
* Too low → slow learning

---

## **7.6 SGD**

Update weights using **one sample at a time**.

## **Mini-Batch GD**

Use a small batch (e.g., 32 samples).

---

## **7.7 Backpropagation**

Algorithm computing gradients layer by layer using chain rule.

---

## **7.8 Vanishing Gradient**

Gradients shrink to near zero in deep networks → slow learning.

---

# ✅ **8. CNNs**

---

## **8.1 Draw CNN (VGG-style)**

```
Input → Conv → Conv → Pool → Conv → Conv → Pool → Flatten → Dense → Output
```

* Convolutional kernel = small matrix (e.g., 3×3)
* Filter = set of kernels
* Feature map = output of convolution
* Pooling = downsampling

---

## **8.2 3×3 Convolution Example**

Input patch:

```
1 2 1
0 1 0
2 1 2
```

Kernel:

```
1 0 1
0 1 0
1 0 1
```

Multiply & sum:
= 1*1 + 2*0 + 1*1 + 0*0 + 1*1 + 0*0 + 2*1 + 1*0 + 2*1
= 1 + 0 + 1 + 0 + 1 + 0 + 2 + 0 + 2 = **7**

---

## **8.3 Pooling Example**

MaxPooling(2×2):

```
1 5
2 3  → max = 5
```

AveragePooling → average = (1+5+2+3)/4 = 2.75

---

## **8.4 CNN Advantages**

* Local pattern detection
* Parameter efficient
* Translation invariance

---

## **8.5 Pretrained CNN Pros/Cons**

**Pros:** Saves time, requires less data
**Cons:** May not fit domain perfectly

---

## **8.6 Data Augmentation**

Artificially expanding dataset (rotate, flip, noise).

---

## **8.7 Transfer Learning**

Using pretrained networks as feature extractors.

---

## **8.8 Image Segmentation**

Classify **each pixel** into a category (e.g., tumor vs non-tumor).

---

## **8.9 CNN Application**

**Autonomous driving:**
CNNs detect lanes, signs, pedestrians.

---

# ✅ **9. Embeddings & Similarity**

---

## **9.1 Cosine Similarity**

Measures similarity between two vectors.

Inputs: two vectors
Output: value from -1 to 1

---

## **9.2 What Are Embeddings?**

Dense vector representations of text/images capturing meaning.

---

## **9.3 Why Useful?**

Enables semantic search, clustering, recommendation systems.

---

## **9.4 Applications**

* Duplicate question detection
* Document search
* Recommendation systems

---

## **9.5 Tokenization**

Splitting text into tokens (words, subwords).

---

## **9.6 Context Length**

Maximum number of tokens the model can attend to.

---

## **9.7 Semantic Similarity**

How close meanings of two texts are.

Examples:

* High: “car” vs “automobile”
* Moderate: “cat” vs “animal”
* Low: “banana” vs “democracy”

---

# ✅ **10. LLMs**

---

## **10.1 Two LLM Hyperparameters**

* **Temperature:** randomness
* **Top-k / Top-p:** sampling diversity

---

## **10.2 Embedding vs Generative Models**

* **Embedding model:** outputs vectors
* **Generative model:** produces text

---

## **10.3 Commercial vs Open Source LLMs**

**Commercial:** GPT-4, Claude
**Open source:** LLaMA, Mistral

---

## **10.4 What 8B/12B/32B Means**

Number of parameters (billions). Larger = more capable but more expensive.

---

## **10.5 Co-Pilot vs Agents**

* **Co-Pilot:** assists user (autocomplete)
* **Agent:** autonomous multi-step reasoner

---

## **10.6 Prompting**

Crafting inputs so models produce desired outputs.

---

## **10.7 Prompt Engineering Techniques**

### **Few-shot**

Provide examples.

### **Chain-of-Thought**

Ask model to think step-by-step.

### **Role prompting**

“Act as a lawyer.”

---

## **10.8 Why Prompt Engineering Matters**

Better prompts → more accuracy, safety, reliability.

---

## **10.9 Typical Use Cases**

* **CursorAI:** coding aid
* **GPT-4:** general reasoning
* **vaderSentiment:** simple sentiment analysis
* **LLaMA:** research, deployable locally

---

## **10.10 Types of LLM Models**

* **Instruction model:** follows commands
* **Reasoning model:** enhanced step-by-step logic
* **MoE:** multiple expert subnetworks
* **Base model:** pretrained but not instruction tuned

---
