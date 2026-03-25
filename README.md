# Waveform Clustering and Unsupervised Learning System

## Overview

---

This project implements a structured data mining pipeline for the preprocessing, transformation, clustering, and analysis of the [Waveform Database Generator (Version 1) dataset](https://archive.ics.uci.edu/dataset/107/waveform+database+generator+version+1).

The system is designed to support unsupervised learning tasks, where inherent structures in the data are discovered without the use of predefined class labels. The dataset consists of three waveform classes described by 21 numerical attributes, all of which include noise, making it suitable for evaluating clustering algorithms under realistic conditions.

The pipeline is organized into multiple stages:

1. Data Preprocessing: Cleaning, normalization, and transformation of raw data
2. Feature Preparation: Structuring the dataset for unsupervised learning
3. Clustering: Application of multiple clustering algorithms
4. Evaluation: Quantitative assessment using clustering validation metrics
5. Visualization and Analysis: Interpretation of discovered patterns

The project follows a modular architecture to ensure reproducibility, clarity, and separation of concerns across all stages of the data mining workflow.

---

## Dataset Description

The Waveform Database Generator (Version 1) dataset is a synthetic dataset widely used in machine learning research.

- Number of Classes: 3 (not used for unsupervised training)
- Number of Features: 21 numerical attributes
- Noise: All features include noise components
- Task Type: Unsupervised learning (clustering)

Although the dataset includes class labels, they are intentionally excluded during the clustering phase and may only be used for post-hoc evaluation and interpretation.

---

## Tools and Technologies

### Programming Language

Python 3.10+

---

## Development Environment Configuration

### 1. Repository Setup

Clone the repository and navigate to the project root:

```bash
git clone 
cd orcid-project
```

### 2. Python Virtual Environment

A virtual environment is recommended to isolate project dependencies.

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Dependency Installation

Install the required libraries using the provided requirements file:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```bash
data/
  ├── raw/               Original dataset
  └── preprocessed/      Cleaned and transformed data

results/                 Outputs (plots, metrics, cluster assignments)

src/
  ├── preprocessing/     Data cleaning, normalization, transformation
  ├── clustering/        Clustering algorithms
  │   ├── kmeans/
  │   ├── agglomerative/
  │   └── dbscan/
  ├── evaluation/        Clustering evaluation metrics
  └── utils/             Shared utilities (e.g., visualization, helpers)
  ```

  ---

## Data

In the project directory, the is a script `data/raw/data-loader.py` that is used to download the dataset for training purposes. This script creates two `csv` files: `X.csv` and `Y.csv` which represent the raw dataset without classfication labels and the classfification labels respectively.
