"""
Generate a detailed report on Gaussian Naive Bayes classification results.
"""
import os
import pandas as pd


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
EXCEL_FILE = os.path.join(RESULTS_DIR, 'naive-bayes-bitacora.xlsx')
REPORT_FILE = os.path.join(RESULTS_DIR, 'naive-bayes-report.md')


def read_excel_sheets():
    """Read all sheets from the Excel file."""
    sheets = {}
    xls = pd.ExcelFile(EXCEL_FILE)
    for sheet_name in xls.sheet_names:
        sheets[sheet_name] = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
    return sheets


def generate_report(sheets):
    """Generate a comprehensive markdown report."""
    report = []
    
    # Header
    report.append("# Gaussian Naive Bayes Classification Report\n")
    report.append("## Classification Project - Waveform Dataset\n")
    report.append("---\n\n")
    
    # 1. Algorithm Description
    report.append("## 1. Algorithm: Gaussian Naive Bayes\n\n")
    report.append("### Overview\n")
    report.append(
        "Gaussian Naive Bayes (GNB) is a probabilistic classifier based on Bayes' theorem "
        "with the assumption that features are conditionally independent given the class label. "
        "It assumes that continuous features follow a Gaussian (normal) distribution within each class.\n\n"
    )
    
    report.append("### Mathematical Foundation\n\n")
    report.append("The classifier applies Bayes' theorem:\n\n")
    report.append("$$P(C_k | X) = \\frac{P(X | C_k) \\cdot P(C_k)}{P(X)}$$\n\n")
    report.append("Where:\n")
    report.append("- $C_k$ is class $k$\n")
    report.append("- $X$ is the feature vector\n")
    report.append("- $P(C_k | X)$ is the posterior probability\n")
    report.append("- $P(X | C_k)$ is the likelihood (assumed Gaussian)\n")
    report.append("- $P(C_k)$ is the prior probability\n\n")
    
    report.append("### Key Assumptions\n")
    report.append("1. **Conditional Independence**: Features are independent given the class\n")
    report.append("2. **Gaussian Distribution**: Feature values within each class follow a normal distribution\n")
    report.append("3. **Equal Variance**: Features have the same variance across classes (can be relaxed)\n\n")
    
    report.append("### Advantages\n")
    report.append("- Fast training and prediction\n")
    report.append("- Works well with high-dimensional data\n")
    report.append("- Provides probability estimates\n")
    report.append("- Robust to irrelevant features\n\n")
    
    report.append("### Limitations\n")
    report.append("- Strong conditional independence assumption (often violated in practice)\n")
    report.append("- May underperform if feature distributions are highly non-Gaussian\n")
    report.append("- Assumes features are independent given class\n\n")
    
    # 2. Evaluation Metrics
    report.append("## 2. Evaluation Methods\n\n")
    
    report.append("### 2.1 Accuracy\n")
    report.append(
        "**Formula**: Accuracy = (TP + TN) / (TP + TN + FP + FN)\n\n"
        "Measures the proportion of correct predictions out of all predictions. "
        "Simple metric but can be misleading with imbalanced datasets.\n\n"
    )
    
    report.append("### 2.2 Balanced Accuracy\n")
    report.append(
        "**Formula**: Balanced Accuracy = (Recall_class_0 + Recall_class_1 + ... + Recall_class_n) / n\n\n"
        "Average of recall for each class. Provides a fair evaluation when classes are imbalanced "
        "by weighting each class equally regardless of its frequency.\n\n"
    )
    
    report.append("### 2.3 Confusion Matrix\n")
    report.append(
        "A matrix showing true vs predicted class labels:\n"
        "- **Diagonal**: Correct predictions\n"
        "- **Off-diagonal**: Misclassifications\n"
        "Reveals which classes are confused with each other.\n\n"
    )
    
    report.append("### 2.4 Precision, Recall, F1-Score\n\n")
    report.append("**Per-class metrics**:\n")
    report.append("- **Precision**: TP / (TP + FP) - How many predicted positives are actually positive\n")
    report.append("- **Recall (Sensitivity)**: TP / (TP + FN) - How many actual positives are found\n")
    report.append("- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean\n\n")
    
    report.append("**Aggregations**:\n")
    report.append("- **Macro Average**: Unweighted mean across all classes\n")
    report.append("- **Weighted Average**: Mean weighted by class support (frequency)\n\n")
    
    # 3. Results
    report.append("## 3. Obtained Results\n\n")
    
    for dataset_prefix in ['X_scaled', 'X_PCA']:
        report.append(f"### 3.{1 if dataset_prefix == 'X_scaled' else 2} Dataset: {dataset_prefix}\n\n")
        
        # Metrics Summary
        summary_sheet = f'{dataset_prefix}_metrics_summary'
        if summary_sheet in sheets:
            summary_df = sheets[summary_sheet]
            report.append("#### Accuracy Metrics\n\n")
            for _, row in summary_df.iterrows():
                report.append(f"- **{row['Metric']}**: {row['Value']}\n")
            report.append("\n")
        
        # Classification Report
        class_report_sheet = f'{dataset_prefix}_class_report'
        if class_report_sheet in sheets:
            class_report_df = sheets[class_report_sheet]
            report.append("#### Per-Class Performance\n\n")
            report.append("| Class | Precision | Recall | F1-Score | Support |\n")
            report.append("|-------|-----------|--------|----------|----------|\n")
            for _, row in class_report_df.iterrows():
                class_label = row['Class']
                prec = row['Precision']
                rec = row['Recall']
                f1 = row['F1-Score']
                supp = row['Support']
                report.append(f"| {class_label} | {prec} | {rec} | {f1} | {supp} |\n")
            report.append("\n")
        
        # Distribution
        dist_sheet = f'{dataset_prefix}_distribution'
        if dist_sheet in sheets:
            dist_df = sheets[dist_sheet]
            report.append("#### Predicted Class Distribution\n\n")
            # Handle potential column name variations
            col_names = dist_df.columns.tolist()
            class_col = next((c for c in col_names if 'class' in c.lower()), col_names[0])
            count_col = next((c for c in col_names if 'count' in c.lower()), col_names[1])
            
            for _, row in dist_df.iterrows():
                class_group = int(row[class_col])
                count = int(row[count_col])
                report.append(f"- Class {class_group}: {count} samples\n")
            report.append("\n")
        
        # Confusion Matrix
        cm_sheet = f'{dataset_prefix}_confusion_matrix'
        if cm_sheet in sheets:
            cm_df = sheets[cm_sheet]
            report.append("#### Confusion Matrix\n\n")
            report.append("```\n")
            report.append(cm_df.to_string(index=False))
            report.append("\n```\n\n")
            report.append("*(Row = True Class, Column = Predicted Class)*\n\n")
    
    # 4. Interpretation
    report.append("## 4. Interpretation of Results\n\n")
    
    report.append("### Comparison Between X_scaled and X_PCA\n\n")
    
    # Extract accuracy values
    x_scaled_acc = sheets['X_scaled_metrics_summary'].iloc[0]['Value']
    x_pca_acc = sheets['X_PCA_metrics_summary'].iloc[0]['Value']
    
    report.append(
        f"- **X_scaled Accuracy**: {x_scaled_acc}\n"
        f"- **X_PCA Accuracy**: {x_pca_acc}\n\n"
    )
    
    report.append(
        "The **X_scaled** dataset (all 21 original features scaled to zero mean and unit variance) "
        "vs **X_PCA** dataset (principal component analysis reducing to 2 components) show different "
        "performance characteristics:\n\n"
    )
    
    report.append(
        "- **X_scaled**: Uses the full feature space with standardization. May capture more information "
        "but could be affected by noise and the curse of dimensionality.\n"
        "- **X_PCA**: Uses only the first 2 principal components capturing maximum variance. "
        "Reduces dimensionality significantly, which can reduce overfitting but may lose important information.\n\n"
    )
    
    report.append("### Class-Specific Performance\n\n")
    
    report.append(
        "Examine the confusion matrix and per-class metrics to identify:\n"
        "- Which classes are well-separated and easily classified\n"
        "- Which classes are frequently confused with each other\n"
        "- If the classifier shows bias toward any particular class\n\n"
    )
    
    report.append("### Implications\n\n")
    
    report.append("1. **Model Reliability**: Higher accuracy and balanced accuracy indicate the model "
        "generalizes well across all classes.\n")
    report.append("2. **Dimensionality Trade-off**: Comparing X_scaled vs X_PCA reveals the importance "
        "of feature selection and dimensionality reduction.\n")
    report.append("3. **Feature Importance**: The waveform dataset has 21 features; PCA compression to 2 "
        "components shows what fraction of variance is captured.\n")
    report.append("4. **Multiclass Challenge**: With 3 classes, balanced accuracy helps ensure the model "
        "doesn't just perform well on the majority class.\n\n")
    
    # 5. Conclusion
    report.append("## 5. Conclusion\n\n")
    
    report.append(
        "Gaussian Naive Bayes provides a fast and interpretable baseline for the waveform classification task. "
        "The results show how the classifier performs with different preprocessing approaches "
        "(full scaled vs dimensionality-reduced). The detailed metrics allow for understanding "
        "both overall performance and class-specific strengths and weaknesses.\n"
    )
    
    return "\n".join(report)


def main():
    if not os.path.exists(EXCEL_FILE):
        print(f"Error: Excel file not found at {EXCEL_FILE}")
        return
    
    print(f"Reading results from: {EXCEL_FILE}")
    sheets = read_excel_sheets()
    
    print("Generating report...")
    report = generate_report(sheets)
    
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == '__main__':
    main()
