# QA

<details>
<summary>Root Mean Square Error — RMSE</summary>
A typical performance measure for regression problems is the root mean square error (RMSE). It gives an idea of how much error the system typically makes in its predictions.
</details>

---

<details>
<summary>Mean Absolute Error — MAE</summary>
If there are many outliers you might consider using MAE (also called average absolute deviation).
</details>

Note: RMSE is more sensitive to outliers than MAE. When outliers are extremely rare, RMSE can perform well and is often preferred.

---

<details>
<summary>TensorFlow</summary>
Open-source deep learning library developed by Google for machine learning and deep learning applications. It efficiently handles large-scale numerical computation and supports CPU and GPU acceleration. Commonly used for training and deploying ML models. 

TPU (Tensor Processing Unit) was designed from scratch by google specifically for tensor based operations that appear in deep learning workloads such as matrix multiplicaitons and convolutions. TPUs are specialized hardware accelerators dedicated only to machine learning tasks.
</details>

---

<details>
<summary>Keras</summary>
An open-source high-level neural networks API that runs on top of TensorFlow. Keras provides a simple, user-friendly interface for building and training neural networks.
</details>

---

<details>
<summary>NumPy</summary>
A fundamental library for numerical computation in Python. Provides the ndarray type for fast, vectorized array operations and is commonly used for data preprocessing in ML.
</details>

---

<details>
<summary>matplotlib</summary>
A data visualization library for creating plots and charts (line plots, bar charts, histograms, scatter plots, etc.). Works well with NumPy and pandas and is useful for visualizing data in ML workflows.
</details>

---

<details>
<summary>Loss</summary>
The loss (or cost) measures how well the model's predictions match the targets. It depends on model parameters (weights and biases) and is used to guide training via optimization. </br>
In regression an example of loss is meansquared error </br>  
In classification an example of loss is cross entropy loss
</details>

---

<details>
<summary>Regularization</summary>
Techniques (L1/L2, weight decay, etc.) that penalize model complexity to improve generalization and reduce overfitting.

limit tree depth for tree based models
</details>

---

<details>
<summary>Dropout</summary>
A stochastic regularization technique that randomly disables a fraction of neurons during training to reduce co-adaptation and improve generalization.

sets reandom y to 0
</details>

---

<details>
<summary>Early Stopping</summary>
Stop training when validation performance stops improving to avoid overfitting. Often combined with checkpoints to restore the best model.
</details>

---

<details>
<summary>Batch Normalization</summary>
Normalizes layer activations to have (approximately) zero mean and unit variance, which can speed up training and improve stability.

Normalize the output of each layer
resacle the data using mean plus standard deviation

</details>

---

<details>
<summary>BERT</summary>
Uses a denoising self-supervised pre-training task </br>
MLM (Masked Language Modeling) - in MLM random words in a sentence are replaced with a special token [MASK] and the model is trained to predict the missing words based on the surounding context </br>
Example:</br> 

```poweshell
input:"The [MASK] is barking loudly"
target: "dog"
```
</details>

---

<details>
<summary>Causal language modeling (CLM)</summary>
or autoregressive modeling </br>
In this setup the model is trained to predict the next word given all previous words in a sentence </br>
Example:</br>

```poweshell
input:"The dog is"
target: "barking"
```

</details>

---

<details>
<summary>BERT</summary>


</details>

---
🧠 Input Embedding
An input embedding is a way to convert raw input data (like words or image patches) into numerical vectors that a neural network can understand.

In NLP: Each word is mapped to a high-dimensional vector (e.g., using Word2Vec, GloVe, or learned embeddings).

In ViTs: Each image is split into patches (e.g., 16×16 pixels), flattened, and then passed through a linear projection to create a vector — this is the patch’s input embedding.

🔹 Purpose: To represent the semantic or visual content of each input unit in a format suitable for processing by the model.

📍 Position Embedding
Transformers process inputs as sequences, but they don’t inherently understand order or spatial layout. That’s where position embeddings come in.

They encode where each input is located in the sequence.

In NLP, they capture word order.

In ViTs, they encode the spatial position of each image patch.

In ViTs: Since patches are treated like tokens, position embeddings help the model know which patch came from which part of the image — top-left, center, bottom-right, etc.

🧩 Combined Input
In practice, the model adds both embeddings together:

Final Input = Input Embedding + Position Embedding

This combined vector is what gets fed into the Transformer layers.

nn dont know the adjencency of pixecls
dense newral network - 

convolutional newral network (CNN)
convolutional kernal
primary output of a convolutional network feature map
Max pool

Bias - variance tradeoff (algorithmic bias)


VGG16 architecture

Auto encoder



If training accuracy is very high but validation accuracy is much lower, the model is likely overfitting.
<img width="1102" height="629" alt="image" src="https://github.com/user-attachments/assets/3249a349-80f3-4f8e-b8ea-2886c94d6c23" />

<img width="1069" height="838" alt="image" src="https://github.com/user-attachments/assets/7a1da623-9c93-4ae6-8d79-db4b1e5778dc" />
<img width="1415" height="626" alt="image" src="https://github.com/user-attachments/assets/150e1552-9df8-409d-935e-65790bf18e40" />
<img width="847" height="475" alt="image" src="https://github.com/user-attachments/assets/27745490-33d9-4346-b186-271b13900ab8" />

