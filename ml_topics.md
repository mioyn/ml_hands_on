# AI Recap Questions – Machine Learning Fundamentals

---

## 1.1 Define what Machine Learning is

Machine Learning (ML) is a subfield of Artificial Intelligence (AI) that focuses on building systems that can **learn patterns from data and make decisions or predictions without being explicitly programmed** for every possible scenario.

Instead of writing fixed rules, ML algorithms:

* Take **input data** (features)
* Learn **patterns, relationships, or structures** from the data
* Use what they learned to **generalize** to new, unseen data

### Formal Definition

> Machine Learning is the study of computer algorithms that improve their performance at a task through experience (data).

### Example

* Email spam filtering: The model learns from labeled emails (spam / not spam) and improves over time.
* Recommendation systems: Netflix or YouTube learns user preferences from past behavior.

---

## 1.2 Define and distinguish supervised and unsupervised learning

### Supervised Learning

Supervised learning is an ML approach where the model is trained on **labeled data**, meaning each training example has:

* Input features (X)
* A known output/target label (y)

The goal is to learn a mapping from inputs to outputs.

**Common tasks:**

* Classification
* Regression

**Examples:**

* Predicting house prices (price is known during training)
* Classifying emails as spam or not spam

### Unsupervised Learning

Unsupervised learning is an ML approach where the model is trained on **unlabeled data**.

The goal is to discover hidden patterns, structures, or relationships in the data.

**Common tasks:**

* Clustering
* Dimensionality reduction

**Examples:**

* Grouping customers based on purchasing behavior
* Topic modeling on text documents

### Key Differences

| Aspect       | Supervised Learning | Unsupervised Learning |
| ------------ | ------------------- | --------------------- |
| Labeled data | Yes                 | No                    |
| Output known | Yes                 | No                    |
| Goal         | Predict outputs     | Discover structure    |

---

## 1.3 Define and distinguish classification and regression

### Classification

Classification is a supervised learning task where the output variable is **categorical**.

The goal is to assign an input to one of several predefined classes.

**Examples:**

* Spam vs Not Spam
* Disease: Yes / No
* Image labels: Cat, Dog, Car

### Regression

Regression is a supervised learning task where the output variable is **continuous (numeric)**.

The goal is to predict a real-valued number.

**Examples:**

* House price prediction
* Temperature forecasting
* Salary estimation

### Comparison

| Aspect         | Classification   | Regression        |
| -------------- | ---------------- | ----------------- |
| Output type    | Discrete classes | Continuous values |
| Example output | "Spam"           | 250000            |

---

## 1.4 Identify classification and regression tasks

### Classification Tasks

* Email spam detection
* Credit card fraud detection
* Image recognition
* Sentiment analysis (positive / negative)

### Regression Tasks

* Predicting stock prices
* Forecasting sales
* Estimating energy consumption
* Predicting exam scores

**Rule of Thumb:**

* If the output is a **category → Classification**
* If the output is a **number → Regression**

---

## 1.5 Explain Linear Regression, including the model equation

Linear Regression is a supervised learning algorithm used for **regression tasks**.

It models the relationship between input features and a continuous output by fitting a straight line (or hyperplane).

### Model Equation

For a single feature:

$
y = \beta_0 + \beta_1 x 
$

For multiple features:

$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n 
$

Where:

* $ y $ = predicted output
* $ x_i $ = input features
* $ \beta_0 $ = intercept
* $ \beta_i $ = coefficients (weights)

### How it Works

* The model finds coefficients that minimize the **Mean Squared Error (MSE)**.
* Common optimization method: **Gradient Descent**.

### Example


Predicting house price based on size:
$
price = 50,000 + 1,200 \times size 
$

---

## 1.6 Explain Logistic Regression, including the model equation (sigmoid)

Logistic Regression is a supervised learning algorithm used for **binary classification**.

Despite its name, it is a **classification algorithm**, not regression.

### Core Idea

* Uses a linear combination of inputs
* Applies a **sigmoid function** to map outputs to probabilities between 0 and 1

### Model Equation

Linear part:
$
z = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n 
$

Sigmoid function:
$ \sigma(z) = \frac{1}{1 + e^{-z}} $

Final output:
$ P(y=1|x) = \sigma(z) $

### Decision Rule

* If probability ≥ 0.5 → Class 1
* Else → Class 0

### Example

* Predicting whether a customer will churn (Yes / No)
* Disease diagnosis (Positive / Negative)

---

## 1.7 Name 3 ML methods other than neural networks, Linear Regression, and Logistic Regression

1. **Decision Trees**
2. **Support Vector Machines (SVM)**
3. **k-Nearest Neighbors (k-NN)**

Other examples:

* Random Forest
* Naive Bayes
* K-Means Clustering

---

## 1.8 Describe one other supervised model of your choice in a few sentences

### Decision Tree

A Decision Tree is a supervised learning algorithm that models decisions using a **tree-like structure**.

* Each internal node represents a feature test
* Each branch represents an outcome
* Each leaf node represents a final prediction

**Advantages:**

* Easy to understand and interpret
* Handles both numerical and categorical data

**Example:**
Predict loan approval based on income, credit score, and employment status.

---

## 1.9 Explain what feature engineering is and why it is important in modeling

Feature engineering is the process of **creating, transforming, and selecting input features** to improve model performance.

### Why It Is Important

* ML models learn only from provided features
* Better features → better predictions
* Can significantly improve accuracy even with simple models

### Examples

* Converting dates into day, month, and year
* Creating interaction features (e.g., price × quantity)
* Scaling numerical values

> Often, **feature engineering matters more than the choice of model**.

---

## 1.10 Name 3 feature engineering methods

1. **Feature Scaling**

   * Standardization (z-score)
   * Min-Max scaling

2. **Encoding Categorical Variables**

   * One-Hot Encoding
   * Label Encoding

3. **Feature Transformation**

   * Log transformation
   * Polynomial features

Other methods:

* Feature selection
* Dimensionality reduction (PCA)
* Handling missing values

---

# AI Recap Questions – Artificial Neural Networks (ANNs)

---

## 2.1 Draw and annotate the structure of a Multi-Layer Perceptron (MLP)

A **Multi-Layer Perceptron (MLP)** is a type of **feedforward artificial neural network** consisting of multiple layers of neurons.

### Basic Structure

```
Input Layer        Hidden Layer(s)         Output Layer

 x1  ──┐            h1  ──┐
 x2  ──┼──► ( w ) ─► h2  ──┼──► ( w ) ─►  y
 x3  ──┘            h3  ──┘
```

### Layer-wise Explanation

1. **Input Layer**

   * Receives raw input features (x₁, x₂, x₃, …)
   * No computation happens here

2. **Hidden Layer(s)**

   * Perform weighted sums of inputs
   * Apply activation functions
   * Responsible for learning complex patterns

3. **Output Layer**

   * Produces final prediction
   * Activation depends on task:

     * Regression → Linear
     * Binary classification → Sigmoid
     * Multi-class classification → SoftMax

### Key Components

* **Weights (w)**: Learnable parameters
* **Bias (b)**: Shifts activation
* **Activation function**: Introduces non-linearity

---

## 2.2 Manually calculate the output of a simple two-layer MLP

Consider a **simple MLP** with:

* 2 input neurons
* 1 hidden neuron
* 1 output neuron
* All weights and biases are integers

### Given

Inputs:

* x₁ = 2, x₂ = 1

Hidden layer:

* w₁ = 3, w₂ = 2
* bias b₁ = 1
* Activation: ReLU

Output layer:

* w₃ = 4
* bias b₂ = 0
* Activation: Linear

### Step 1: Hidden layer calculation

$ z_1 = (2 \times 3) + (1 \times 2) + 1 = 6 + 2 + 1 = 9 $

Apply ReLU:
$ h = \max(0, 9) = 9 $

### Step 2: Output layer calculation

$ y = (9 \times 4) + 0 = 36 $

### Final Output

$ \boxed{y = 36} $

---

## 2.3 Distinguish linear and non-linear models

### Linear Models

Linear models assume a **linear relationship** between inputs and outputs.

Example equation:
$ y = w_1x_1 + w_2x_2 + b $

**Characteristics:**

* Simple and interpretable
* Cannot model complex patterns
* Examples: Linear Regression, Logistic Regression

### Non-Linear Models

Non-linear models can learn **complex relationships** due to non-linear activation functions.

**Characteristics:**

* Can approximate complex functions
* More powerful but harder to interpret
* Examples: Neural Networks, Decision Trees

### Key Difference

| Aspect               | Linear        | Non-Linear          |
| -------------------- | ------------- | ------------------- |
| Pattern complexity   | Low           | High                |
| Activation functions | None / Linear | ReLU, Sigmoid, tanH |

---

## 2.4 Draw and identify activation functions

### Step Function

* Output is binary
* Used in early perceptrons

$ f(x) = \begin{cases} 1 & x \ge 0 \ 0 & x < 0 \end{cases} $

---

### Sigmoid Function

* Outputs values between 0 and 1
* Interpretable as probability

$ \sigma(x) = \frac{1}{1 + e^{-x}} $

---

### tanH (Hyperbolic Tangent)

* Outputs values between -1 and 1
* Zero-centered

$ tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $

---

### ReLU (Rectified Linear Unit)

* Most commonly used in deep networks

$ f(x) = \max(0, x) $

---

## 2.5 Explain what the SoftMax activation function does

SoftMax is used in the **output layer of multi-class classification models**.

### What It Does

* Converts raw scores (logits) into probabilities
* Ensures:

  * All values are between 0 and 1
  * Sum of probabilities = 1

### Formula

$ SoftMax(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} $

### Example

Logits: [2, 1, 0]

SoftMax output:

* Class 1: 0.67
* Class 2: 0.24
* Class 3: 0.09

---

## 2.6 Calculate the number of parameters in a Dense Layer

### Formula

For a fully connected layer:

$ Parameters = (Inputs \times Neurons) + Biases $

### Example

* Input features = 10
* Neurons = 5

Weights = 10 × 5 = 50
Biases = 5

**Total Parameters = 55**

---

## 2.7 Enumerate advantages of using TensorFlow or PyTorch

1. Automatic differentiation (backpropagation)
2. GPU and TPU acceleration
3. Large ecosystem and community support
4. Pre-built layers and models
5. Easy experimentation and debugging
6. Deployment support (mobile, cloud)

---

## 2.8 Describe one practical application of ANNs (other than CNNs or LLMs)

### Time-Series Forecasting

Artificial Neural Networks are widely used in **time-series forecasting**, such as:

* Electricity demand prediction
* Stock market trend analysis
* Weather forecasting

ANNs can learn temporal patterns from historical data and generalize to future values.

---

## 2.9 Enumerate a few hyperparameters of ANNs

Common ANN hyperparameters include:

1. Learning rate
2. Number of hidden layers
3. Number of neurons per layer
4. Activation functions
5. Batch size
6. Number of epochs
7. Optimizer (SGD, Adam)

---

## 2.10 Name what John Hopfield and Geoffrey Hinton achieved

### John Hopfield

* Introduced **Hopfield Networks**
* A form of recurrent neural network
* Used for associative memory and pattern completion

### Geoffrey Hinton

* Pioneer of **Deep Learning**
* Popularized backpropagation
* Major contributions to:

  * Boltzmann Machines
  * Deep Belief Networks
  * Modern neural networks

Often called the **“Godfather of Deep Learning”**.

---

# AI Recap Questions – Training Artificial Neural Networks (ANNs)

---

## 3.1 Draw an ideal learning curve for ANN training

A **learning curve** shows how the **training loss and validation loss** change over training epochs.

### Ideal Learning Curve (Loss vs Epochs)

```
Loss
│\
│ \        Validation Loss
│  \_____
│        \____
│
│\
│ \________________ Training Loss
│
└────────────────────────── Epochs
```

### Interpretation

* Training loss decreases steadily
* Validation loss decreases and stabilizes close to training loss
* Small gap between curves → good generalization

---

## 3.2 Draw an accuracy curve during ANN training

Accuracy curves show **model performance improvement over time**.

### Ideal Accuracy Curve (Accuracy vs Epochs)

```
Accuracy
│        ________ Validation Accuracy
│       /
│      /
│_____/________________ Training Accuracy
│
└────────────────────────── Epochs
```

### Interpretation

* Training accuracy increases
* Validation accuracy increases and plateaus
* Curves stay close → no overfitting

---

## 3.3 Interpret typical learning and accuracy curves

### Common Scenarios

#### 1. Good Fit

* Training and validation loss decrease
* Accuracy curves converge

#### 2. Overfitting

```
Loss
│\
│ \____ Validation Loss increases
│
│ \____________ Training Loss continues decreasing
└──────── Epochs
```

* Training accuracy → very high
* Validation accuracy → stagnates or drops

#### 3. Underfitting

* Both losses remain high
* Both accuracies are low

---

## 3.4 Explain why scaling data is important when training ANNs

Feature scaling ensures that **all input features contribute equally** to learning.

### Why Scaling Matters

* ANNs use **gradient-based optimization**
* Large feature values dominate gradients
* Leads to slow or unstable convergence

### Example

Unscaled features:

* Age: 0–100
* Salary: 20,000–200,000

Salary dominates learning unless scaled.

### Common Scaling Methods

* Standardization (mean = 0, std = 1)
* Min-Max scaling (0 to 1)

---

## 3.5 Define overfitting and underfitting

### Overfitting

Overfitting occurs when a model:

* Learns training data **too well**, including noise
* Performs poorly on unseen data

**Symptoms:**

* Very low training loss
* High validation loss

### Underfitting

Underfitting occurs when a model:

* Is too simple to capture patterns
* Performs poorly on both training and validation data

---

## 3.6 Recognize overfitting from model training results

Overfitting can be identified when:

* Training loss keeps decreasing
* Validation loss starts increasing
* Training accuracy → near 100%
* Validation accuracy → stagnates or drops

### Visual Clue

* Large gap between training and validation curves

---

## 3.7 Explain why many ANNs overfit if trained long enough

Reasons ANNs tend to overfit:

1. High model capacity (many parameters)
2. Ability to memorize training data
3. Insufficient training data
4. Too many training epochs

Given enough time, ANNs can **fit noise instead of signal**.

---

## 3.8 Define Mean Squared Error (MSE) and its use

Mean Squared Error is a **regression loss function**.

### Equation

$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $

Where:

* ( y_i ) = true value
* ( \hat{y}_i ) = predicted value

### Properties

* Penalizes large errors heavily
* Smooth and differentiable

### Example

True values: [3, 5]
Predictions: [2, 7]

MSE = ((1)² + (-2)²) / 2 = 2.5

---

## 3.9 Define Binary Cross-Entropy and its use

Binary Cross-Entropy is a **loss function for binary classification**.

### Equation

$ L = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)] $

### Usage

* Used with Sigmoid activation
* Measures distance between true labels and predicted probabilities

### Example

True label = 1
Predicted probability = 0.9 → low loss
Predicted probability = 0.1 → high loss

---

## 3.10 Define regularization

Regularization is a set of techniques used to **prevent overfitting** by constraining model complexity.

### Core Idea

* Penalize large weights
* Encourage simpler models
* Improve generalization

---

## 3.11 Name 3 regularization methods for ANNs

1. **L2 Regularization (Weight Decay)**

   * Penalizes large weights

2. **Dropout**

   * Randomly disables neurons during training

3. **Early Stopping**

   * Stops training when validation loss increases

Other methods:

* L1 regularization
* Data augmentation
* Batch normalization

---

# AI Recap Questions – Model Evaluation

---

## 4.1 Distinguish training, validation, and test datasets

In machine learning, data is commonly split into **three distinct subsets** to build and evaluate models properly.

### Training Dataset

* Used to **train the model**
* The model learns parameters (weights, biases) from this data
* Usually the **largest portion** of the dataset (e.g., 60–70%)

**Example:**

* A neural network updates its weights using training data via backpropagation

---

### Validation Dataset

* Used to **tune hyperparameters** and make modeling decisions
* Helps monitor overfitting during training
* Not used to update model weights

**Example:**

* Choosing number of hidden layers
* Deciding when to stop training (early stopping)

---

### Test Dataset

* Used **only once** after all training and tuning is complete
* Provides an **unbiased estimate** of final model performance
* Must remain completely unseen during training

---

### Summary Table

| Dataset    | Purpose              | Used for Learning? |
| ---------- | -------------------- | ------------------ |
| Training   | Learn parameters     | Yes                |
| Validation | Tune hyperparameters | No                 |
| Test       | Final evaluation     | No                 |

---

## 4.2 Describe what data leakage from the test dataset is

**Data leakage** occurs when information from the **test dataset unintentionally influences the model during training or validation**.

### Why It Is Dangerous

* Leads to overly optimistic performance
* Model fails in real-world deployment

### Common Causes

1. Performing feature scaling **before** splitting data
2. Using test data for hyperparameter tuning
3. Target leakage (features derived from future information)

### Example

If the mean and standard deviation of the *entire dataset* are used for scaling, the test data has leaked into training.

---

## 4.3 Identify metrics for classification and regression problems

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Log loss (Cross-Entropy)

### Regression Metrics

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² (Coefficient of Determination)

---

## 4.4 Calculate the accuracy from a confusion matrix

### Confusion Matrix

|                 | Predicted Positive | Predicted Negative |
| --------------- | ------------------ | ------------------ |
| Actual Positive | TP = 40            | FN = 10            |
| Actual Negative | FP = 5             | TN = 45            |

### Accuracy Formula

$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} $

### Calculation

$ Accuracy = \frac{40 + 45}{40 + 45 + 5 + 10} = \frac{85}{100} = 0.85 $

**Accuracy = 85%**

---

## 4.5 Calculate TP, FP, TN, FN from a confusion matrix

From the same matrix:

* **True Positive (TP):** 40
* **False Positive (FP):** 5
* **True Negative (TN):** 45
* **False Negative (FN):** 10

### Definitions

* TP: Correctly predicted positive
* FP: Incorrectly predicted positive
* TN: Correctly predicted negative
* FN: Incorrectly predicted negative

---

## 4.6 Describe pros and cons of accuracy versus binary cross-entropy

### Accuracy

**Pros:**

* Easy to understand
* Intuitive

**Cons:**

* Misleading for imbalanced datasets
* Ignores confidence of predictions

---

### Binary Cross-Entropy (Log Loss)

**Pros:**

* Considers prediction probabilities
* Penalizes confident wrong predictions
* Better for probabilistic models

**Cons:**

* Less intuitive
* Harder to interpret directly

---

## 4.7 Define the Mean Absolute Error (MAE)

Mean Absolute Error measures the **average absolute difference** between predicted and true values.

### Equation

$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $

### Properties

* Treats all errors equally
* More robust to outliers than MSE

### Example

True values: [3, 5]
Predictions: [2, 7]

MAE = (1 + 2) / 2 = 1.5

---

## 4.8 Compare two classification models based on their ROC curve

The **ROC (Receiver Operating Characteristic) curve** plots:

* True Positive Rate (Recall)
* False Positive Rate

### Interpretation

* Curve closer to top-left → better model
* Diagonal line → random guessing

### Comparison

* Model A has higher ROC-AUC than Model B
* Model A performs better across thresholds

### AUC Values

| AUC     | Interpretation |
| ------- | -------------- |
| 0.5     | Random         |
| 0.7–0.8 | Acceptable     |
| 0.8–0.9 | Good           |
| > 0.9   | Excellent      |

---

## 4.9 Define model drift or data drift

**Data drift (model drift)** occurs when the **statistical properties of input data change over time**.

### Causes

* Changes in user behavior
* Market trends
* Sensor degradation

### Impact

* Model performance degrades
* Requires retraining

### Example

A spam filter trained on 2020 emails may fail on 2025 email patterns.

---

## 4.10 Describe one evaluation method used for LLMs

### Human Evaluation

Human evaluators assess LLM outputs based on:

* Relevance
* Coherence
* Factual accuracy
* Helpfulness

### Example

Annotators rank multiple model responses to the same prompt. These rankings are often used in **RLHF (Reinforcement Learning from Human Feedback)**.

Other methods:

* BLEU / ROUGE
* Win-rate comparisons
* Benchmark datasets

---

# AI Recap Questions – Gradient Descent and Backpropagation

---

## 5.1 Write the equation for Gradient Descent

Gradient Descent is an optimization algorithm used to **minimize a loss (cost) function** by iteratively updating model parameters in the direction of steepest descent.

### Core Equation

$ \theta_{new} = \theta_{old} - \eta \cdot \nabla J(\theta) $

Where:

* ( \theta ) = model parameters (weights, biases)
* ( \eta ) = learning rate
* ( J(\theta) ) = loss (cost) function
* ( \nabla J(\theta) ) = gradient (first derivative) of the loss

### Intuition

* The gradient tells us **which direction increases the loss most**
* We move in the **opposite direction** to reduce the loss

---

## 5.2 Annotate an image of gradient descent optimization

Gradient Descent can be visualized as moving downhill on a loss surface.

```
Loss (J)
│          •  (start)
│        •
│      •
│    •
│  •        ← parameter updates
│•
└────────────────────────── θ (parameters)
```

### Annotations

* X-axis: Model parameters (θ)
* Y-axis: Loss value J(θ)
* Dots: Iterative updates
* Goal: Reach the global or local minimum

---

## 5.3 Explain why optimal parameters cannot be found analytically

In simple models (e.g., linear regression), optimal parameters can be found analytically using calculus.

However, for **neural networks**, this is usually impossible because:

* Loss functions are **highly non-linear**
* Composed of many nested functions
* Result in a **non-convex optimization problem**
* Second derivatives (Hessian matrix) are extremely large and expensive to compute

### Consequence

There is **no closed-form solution**, so we rely on **iterative numerical methods** like Gradient Descent.

---

## 5.4 Describe what happens during one training epoch

One training epoch means the model has seen **all training samples once**.

### Steps in One Epoch

* Split training data into batches
* Perform forward pass to compute predictions
* Calculate loss
* Compute gradients via backpropagation
* Update weights using gradient descent
* Repeat for all batches

---

## 5.5 Explain why weights are initialized randomly instead of zero

If all weights are initialized to zero:

* All neurons in a layer receive identical gradients
* Neurons learn the same features
* Network fails to break symmetry

### Random Initialization

* Breaks symmetry
* Allows neurons to learn different features
* Enables effective learning

> Biases can be initialized to zero, but **weights should not**.

---

## 5.6 Describe effects of learning rate being too high or too low

### Learning Rate Too High

* Updates overshoot the minimum
* Loss oscillates or diverges
* Training becomes unstable

### Learning Rate Too Low

* Very slow convergence
* Training takes too long
* Can get stuck in local minima

---

## 5.7 Define Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent updates parameters using **one training example at a time**.

### Characteristics

* Faster updates
* Noisy but can escape local minima
* Computationally efficient for large datasets

---

## 5.8 Define Mini-batch Gradient Descent

Mini-batch Gradient Descent is a compromise between batch GD and SGD.

### Key Idea

* Uses small batches (e.g., 32, 64, 128 samples)
* Balances stability and efficiency
* Most commonly used in practice

---

## 5.9 Describe what backpropagation is

Backpropagation is an algorithm used to **efficiently compute gradients** of the loss function with respect to all model parameters.

### How It Works

* Applies the chain rule of calculus
* Propagates error backward from output to input layers
* Computes partial derivatives for each weight
* Enables gradient-based optimization

---

## 5.10 Describe the vanishing gradient problem in one sentence

The vanishing gradient problem occurs when gradients become extremely small in deep networks, causing early layers to learn very slowly or stop learning entirely.

---


# AI Recap Questions – Convolutional Neural Networks (CNNs)

---

## 6.1 Draw and annotate the structure of a typical CNN (e.g. VGG-16)

A **Convolutional Neural Network (CNN)** is a specialized neural network designed to process **grid-like data**, especially images.

### Typical CNN Structure (VGG-style)

```
Input Image (224×224×3)
        │
[ Convolution + ReLU ]  ← feature extraction
        │
[ Convolution + ReLU ]
        │
[ Max Pooling ]         ← downsampling
        │
[ Convolution + ReLU ]
        │
[ Max Pooling ]
        │
[ Flatten ]             ← convert 3D → 1D
        │
[ Fully Connected ]
        │
[ SoftMax Output ]      ← class probabilities
```

### Explanation of Layers

* **Convolution Layers**: Extract local features (edges, textures)
* **Pooling Layers**: Reduce spatial dimensions
* **Flatten Layer**: Converts feature maps to a vector
* **Dense Layers**: Perform classification

---

## 6.2 Use the terms: convolutional kernel, filter, feature map, pooling, flattening

* **Convolutional Kernel (Filter):**
  A small matrix (e.g. 3×3) that slides over the image to detect patterns

* **Filter:**
  Another name for a kernel; each filter learns a specific feature

* **Feature Map:**
  The output produced after applying a filter to the input

* **Pooling:**
  Downsamples feature maps (MaxPooling or AveragePooling)

* **Flattening:**
  Converts multi-dimensional feature maps into a 1D vector

---

## 6.3 Calculate the result of a 3×3 convolutional kernel

### Input Patch


      1  2  3 
      4  5  6 
      7  8  9

### Kernel


      1  0  -1 
      1  0  -1 
      1  0  -1


### Convolution Calculation

$
(1×1) + (2×0) + (3×-1) +
(4×1) + (5×0) + (6×-1) +
(7×1) + (8×0) + (9×-1)
$

$
= (1 - 3) + (4 - 6) + (7 - 9) = -6
$

**Output value = -6**

---

## 6.4 Calculate the result of a MaxPooling or AveragePooling layer

### Feature Map


      1  3  2  4 
      5  6  1  2 
      2  4  8  1 
      1  2  3  0


### 2×2 MaxPooling


      6  4 
      4  8


### 2×2 AveragePooling


      3.75  2.25 
      2.25  3.0


---

## 6.5 Name advantages of CNNs over MLPs

1. Parameter sharing reduces number of weights
2. Local connectivity exploits spatial structure
3. Translation invariance via pooling
4. Better performance on images
5. Less prone to overfitting than MLPs on images

---

## 6.6 Name advantages and disadvantages of pretrained CNNs

### Advantages

* Faster training
* Require less labeled data
* Proven strong feature extractors

### Disadvantages

* Large model size
* Domain mismatch issues
* Limited architecture flexibility

---

## 6.7 Describe what data augmentation is

Data augmentation artificially increases dataset size by applying transformations such as:

* Rotation
* Flipping
* Cropping
* Brightness adjustment

This improves generalization and reduces overfitting.

---

## 6.8 Describe what transfer learning is

Transfer learning reuses a **pretrained CNN** on a new task.

### How It Works

* Freeze early layers
* Retrain final layers on new dataset

### Example

Using ImageNet-trained VGG-16 to classify medical images.

---

## 6.9 Define what image segmentation is

Image segmentation is the task of assigning a **class label to every pixel** in an image.

### Types

* Semantic segmentation
* Instance segmentation

---

## 6.10 Describe one practical application of CNNs

### Medical Image Analysis

CNNs are used to detect tumors in MRI or CT scans.

They automatically learn features from images and assist doctors with faster and more accurate diagnosis.

---
# AI Recap Questions – Text and Embeddings

---

## 7.1 Describe what cosine similarity is used for, its input and output

**Cosine similarity** is a metric used to measure how **similar two vectors are in direction**, regardless of their magnitude.

### What It Is Used For

Cosine similarity is widely used to:

* Compare **text embeddings**
* Measure **semantic similarity** between documents or sentences
* Perform **information retrieval** (search)
* Detect **duplicate or near-duplicate text**

### Input

* Two vectors of equal length (e.g., embeddings)

  * $ \vec{A} = [a_1, a_2, ..., a_n] $
  * $ \vec{B} = [b_1, b_2, ..., b_n] $

### Output

* A single numeric value in the range **[-1, 1]**

  * 1 → identical direction (very similar)
  * 0 → orthogonal (no similarity)
  * -1 → opposite direction (very dissimilar)

### Equation

$ \text{cosine_similarity}(A, B) = \frac{A \cdot B}{||A|| , ||B||} $

---

## 7.2 Define what embeddings are

**Embeddings** are **dense numerical vector representations** of data (such as words, sentences, or images) that capture **semantic meaning**.

### Key Characteristics

* Fixed-length vectors (e.g., 384, 768 dimensions)
* Similar items have similar vectors
* Learned from large datasets

### Example

* Word embedding for "king" is numerically close to "queen"
* Sentence embeddings place semantically similar sentences near each other

---

## 7.3 Explain why embeddings are useful

Embeddings are useful because they:

1. Capture **semantic meaning**, not just surface form
2. Convert unstructured data (text, images) into numbers
3. Enable efficient similarity search
4. Reduce dimensionality compared to one-hot encoding

### Comparison Example

| Representation   | Similarity Awareness |
| ---------------- | -------------------- |
| One-hot encoding | No                   |
| Embeddings       | Yes                  |

---

## 7.4 Enumerate 3 applications of text embeddings

1. **Semantic Search**

   * Retrieve documents based on meaning, not keywords

2. **Text Clustering**

   * Group similar documents or sentences

3. **Recommendation Systems**

   * Recommend articles or products based on content similarity

Other applications:

* Question answering
* Chatbots
* Plagiarism detection

---

## 7.5 Define Tokenisation

**Tokenisation** is the process of breaking text into smaller units called **tokens**.

### Types of Tokens

* Words: "Machine" | "Learning"
* Subwords: "learn" + "##ing"
* Characters

### Example

Sentence:

> "Deep learning is powerful"

Tokens:

> ["Deep", "learning", "is", "powerful"]

---

## 7.6 Describe briefly what context length is

**Context length** refers to the **maximum number of tokens** a language model can consider at one time.

### Why It Matters

* Limits how much text the model can remember
* Longer context enables:

  * Better coherence
  * Understanding long documents

### Example

If a model has a context length of 4,096 tokens, it cannot process text beyond that limit in a single pass.

---

## 7.7 Explain semantic similarity in a few sentences

**Semantic similarity** measures how closely two pieces of text are related in **meaning**, not exact wording.

It relies on embeddings and similarity metrics such as cosine similarity.

Two sentences can be semantically similar even if they share few or no common words.

---

## 7.8 Give examples for high, low, and moderate semantic similarity

### High Semantic Similarity

* "The cat is sleeping on the sofa"
* "A feline is resting on the couch"

### Moderate Semantic Similarity

* "The cat is sleeping on the sofa"
* "The dog is lying on the bed"

### Low Semantic Similarity

* "The cat is sleeping on the sofa"
* "Quantum physics explains atomic behavior"

---

# AI Recap Questions – LLMs and Prompting

---

## 8.1 Name 2 hyperparameters of LLMs and what they do

1. **Temperature**

   * Controls randomness in output generation
   * Low temperature (e.g., 0.2) → deterministic responses
   * High temperature (e.g., 1.0) → more creative, diverse responses

2. **Top-k / Top-p (nucleus sampling)**

   * Limits the number of tokens considered at each step
   * Top-k: Only the k most probable tokens are sampled
   * Top-p: Smallest set of tokens whose cumulative probability ≥ p

---

## 8.2 Explain the difference between an embedding model and a generative model

* **Embedding Model:**

  * Converts text into **dense numerical vectors**
  * Used for similarity, clustering, search, and retrieval
  * Example: "king" → [0.21, 0.05, ..., 0.87]

* **Generative Model:**

  * Produces **new content** (text, code, etc.) from a prompt
  * Focuses on **sequence generation**
  * Example: GPT-4 generating a story

---

## 8.3 Name 2 commercial and 2 open source LLMs

### Commercial LLMs

* OpenAI GPT-4 / GPT-3.5
* Anthropic Claude

### Open Source LLMs

* LLaMA (Meta)
* Falcon

---

## 8.4 Explain what the 8B/12B/32B parameter model means

* B stands for **billion parameters**
* 8B → 8 billion parameters
* 12B → 12 billion parameters
* 32B → 32 billion parameters

### Implications

* More parameters → generally higher capacity
* Requires more **compute and memory**
* Can capture more complex patterns

---

## 8.5 Distinguish Co-Pilot and Agents

* **Co-Pilot:**

  * Context-aware code completion assistant
  * Suggests code snippets while programming
  * Example: GitHub Copilot

* **Agents:**

  * Autonomous systems that can perform **multi-step tasks**
  * Can interact with tools, APIs, or external systems
  * Example: AutoGPT performing research + email automation

---

## 8.6 Describe what prompting is in a few sentences

**Prompting** is the process of providing an **input or instruction to a language model** to elicit a desired output.

* Can be a question, instruction, or context
* Example: "Translate the following English text to French: 'Hello, how are you?'"
* Key to getting accurate and useful responses

---

## 8.7 Explain any 3 prompt engineering techniques using examples

1. **Chain-of-Thought Prompting**

   * Encourage step-by-step reasoning
   * Example: "Explain step by step how to solve 23 × 47."

2. **Few-Shot Prompting**

   * Provide a few examples in the prompt
   * Example: "Translate English to Spanish: 'Hello' → 'Hola', 'Thank you' → 'Gracias', 'Good morning' → ?"

3. **Zero-Shot Prompting**

   * Direct instruction without examples
   * Example: "Summarize the following paragraph in one sentence."

---

## 8.8 Explain why prompt engineering matters

* LLMs are **sensitive to input phrasing**
* Good prompts produce **accurate, relevant, and coherent outputs**
* Poor prompts lead to **irrelevant or hallucinated answers**
* Optimizes **efficiency** and reduces need for post-processing

---

## 8.9 Name typical use cases for CursorAI, GPT-4, vaderSentiment and LLaMa

* **CursorAI:** Code assistance and autocompletion
* **GPT-4:** Content generation, summarization, translation, reasoning
* **vaderSentiment:** Sentiment analysis on social media or text
* **LLaMa:** Research, experimentation, embeddings, fine-tuning

---

## 8.10 Distinguish Instruction Model, Reasoning Model, Mixture of Experts (MoE) Model and Base Model

| Model Type                     | Description                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------ |
| Instruction Model              | Trained to **follow human instructions**; good at tasks with clear prompts                 |
| Reasoning Model                | Optimized for **complex problem-solving** and logical reasoning                            |
| Mixture of Experts (MoE) Model | Uses **different expert sub-networks** for different inputs; selectively activates experts |
| Base Model                     | Core pretrained model **without instruction tuning**; can be fine-tuned or prompted        |

---