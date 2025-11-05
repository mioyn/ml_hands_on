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
</details>

---

<details>
<summary>Dropout</summary>
A stochastic regularization technique that randomly disables a fraction of neurons during training to reduce co-adaptation and improve generalization.
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
</details>

---

If training accuracy is very high but validation accuracy is much lower, the model is likely overfitting.
<img width="1102" height="629" alt="image" src="https://github.com/user-attachments/assets/3249a349-80f3-4f8e-b8ea-2886c94d6c23" />

<img width="1069" height="838" alt="image" src="https://github.com/user-attachments/assets/7a1da623-9c93-4ae6-8d79-db4b1e5778dc" />
<img width="1415" height="626" alt="image" src="https://github.com/user-attachments/assets/150e1552-9df8-409d-935e-65790bf18e40" />
