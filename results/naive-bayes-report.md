# Gaussian Naive Bayes Classification Report

## Classification Project - Waveform Dataset

---

## 1. Algorithm: Gaussian Naive Bayes

### Overview

Gaussian Naive Bayes (GNB) is a probabilistic classifier based on Bayes' theorem with the assumption that features are conditionally independent given the class label. It assumes that continuous features follow a Gaussian (normal) distribution within each class.

### Mathematical Foundation

The classifier applies Bayes' theorem:

$$P(C_k | X) = \frac{P(X | C_k) \cdot P(C_k)}{P(X)}$$

Where:

- $C_k$ is class $k$

- $X$ is the feature vector

- $P(C_k | X)$ is the posterior probability

- $P(X | C_k)$ is the likelihood (assumed Gaussian)

- $P(C_k)$ is the prior probability

### Key Assumptions

1. **Conditional Independence**: Features are independent given the class

2. **Gaussian Distribution**: Feature values within each class follow a normal distribution

3. **Equal Variance**: Features have the same variance across classes (can be relaxed)

### Advantages

- Fast training and prediction

- Works well with high-dimensional data

- Provides probability estimates

- Robust to irrelevant features

### Limitations

- Strong conditional independence assumption (often violated in practice)

- May underperform if feature distributions are highly non-Gaussian

- Assumes features are independent given class

## 2. Evaluation Methods

### 2.1 Accuracy

**Formula**: Accuracy = (TP + TN) / (TP + TN + FP + FN)

Measures the proportion of correct predictions out of all predictions. Simple metric but can be misleading with imbalanced datasets.

### 2.2 Balanced Accuracy

**Formula**: Balanced Accuracy = (Recall_class_0 + Recall_class_1 + ... + Recall_class_n) / n

Average of recall for each class. Provides a fair evaluation when classes are imbalanced by weighting each class equally regardless of its frequency.

### 2.3 Confusion Matrix

A matrix showing true vs predicted class labels:

- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications
Reveals which classes are confused with each other.

### 2.4 Precision, Recall, F1-Score

**Per-class metrics**:

- **Precision**: TP / (TP + FP) - How many predicted positives are actually positive

- **Recall (Sensitivity)**: TP / (TP + FN) - How many actual positives are found

- **F1-Score**: 2 *(Precision* Recall) / (Precision + Recall) - Harmonic mean

**Aggregations**:

- **Macro Average**: Unweighted mean across all classes

- **Weighted Average**: Mean weighted by class support (frequency)

## 3. Obtained Results

### 3.1 Dataset: X_scaled

#### Accuracy Metrics

- **Accuracy**: 0.8092

- **Balanced Accuracy**: 0.8084

#### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| 0 | 0.9768 | 0.5335 | 0.6901 | 1657 |
| 1 | 0.7662 | 0.9454 | 0.8464 | 1647 |
| 2 | 0.778 | 0.9463 | 0.854 | 1696 |
| macro avg | 0.8403 | 0.8084 | 0.7968 | 5000 |
| weighted avg | 0.84 | 0.8092 | 0.7972 | 5000 |

#### Predicted Class Distribution

- Class 0: 0 samples

- Class 1: 1 samples

- Class 2: 2 samples

#### Confusion Matrix

```
 True Class   0    1    2
          0 884  393  380
          1  12 1557   78
          2   9   82 1605
```

*(Row = True Class, Column = Predicted Class)*

### 3.2 Dataset: X_PCA

#### Accuracy Metrics

- **Accuracy**: 0.83

- **Balanced Accuracy**: 0.8301

#### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| 0 | 0.749 | 0.8769 | 0.8079 | 1657 |
| 1 | 0.881 | 0.8045 | 0.841 | 1647 |
| 2 | 0.8817 | 0.809 | 0.8438 | 1696 |
| macro avg | 0.8372 | 0.8301 | 0.8309 | 5000 |
| weighted avg | 0.8375 | 0.83 | 0.831 | 5000 |

#### Predicted Class Distribution

- Class 0: 0 samples

- Class 1: 1 samples

- Class 2: 2 samples

#### Confusion Matrix

```

 True Class    0    1    2
          0 1453  103  101
          1  239 1325   83
          2  248   76 1372

```

*(Row = True Class, Column = Predicted Class)*

## 4. Interpretation of Results

### Comparison Between X_scaled and X_PCA

- **X_scaled Accuracy**: 0.8092
- **X_PCA Accuracy**: 0.83

The **X_scaled** dataset (all 21 original features scaled to zero mean and unit variance) vs **X_PCA** dataset (principal component analysis reducing to 2 components) show different performance characteristics:

- **X_scaled**: Uses the full feature space with standardization. May capture more information but could be affected by noise and the curse of dimensionality.
- **X_PCA**: Uses only the first 2 principal components capturing maximum variance. Reduces dimensionality significantly, which can reduce overfitting but may lose important information.

### Class-Specific Performance

Examine the confusion matrix and per-class metrics to identify:

- Which classes are well-separated and easily classified
- Which classes are frequently confused with each other
- If the classifier shows bias toward any particular class

### Implications

1. **Model Reliability**: Higher accuracy and balanced accuracy indicate the model generalizes well across all classes.

2. **Dimensionality Trade-off**: Comparing X_scaled vs X_PCA reveals the importance of feature selection and dimensionality reduction.

3. **Feature Importance**: The waveform dataset has 21 features; PCA compression to 2 components shows what fraction of variance is captured.

4. **Multiclass Challenge**: With 3 classes, balanced accuracy helps ensure the model doesn't just perform well on the majority class.

## 5. Conclusion

Gaussian Naive Bayes provides a fast and interpretable baseline for the waveform classification task. The results show how the classifier performs with different preprocessing approaches (full scaled vs dimensionality-reduced). The detailed metrics allow for understanding both overall performance and class-specific strengths and weaknesses.
