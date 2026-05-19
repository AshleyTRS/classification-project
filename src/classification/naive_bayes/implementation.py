import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    balanced_accuracy_score,
)


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'naive-bayes-bitacora.xlsx')


def load_features(dataset_name):
    path = os.path.join(DATA_DIR, 'preprocessed', dataset_name)
    df = pd.read_csv(path)
    return df


def load_labels():
    path = os.path.join(DATA_DIR, 'raw', 'y.csv')
    y = pd.read_csv(path, header=0)
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    return y


def classify_dataset(df, y):
    model = GaussianNB()
    model.fit(df.values, y.values)
    predictions = model.predict(df.values)
    return predictions


def compute_metrics(y_true, y_pred):
    """Compute classification metrics."""
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Balanced accuracy
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['classification_report'] = report
    
    return metrics


def format_metrics_for_excel(metrics):
    """Format metrics into DataFrames suitable for Excel."""
    formatted = {}
    
    # Accuracy and Balanced Accuracy
    summary_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Balanced Accuracy'],
        'Value': [
            f"{metrics['accuracy']:.4f}",
            f"{metrics['balanced_accuracy']:.4f}"
        ]
    })
    formatted['summary'] = summary_df
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    classes = sorted(set(range(cm.shape[0])))
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name = 'True Class'
    cm_df.columns.name = 'Predicted Class'
    formatted['confusion_matrix'] = cm_df.reset_index()
    
    # Classification Report
    report = metrics['classification_report']
    report_data = []
    for class_label in sorted([k for k in report.keys() if k.isdigit()]):
        class_metrics = report[class_label]
        report_data.append({
            'Class': int(class_label),
            'Precision': f"{class_metrics['precision']:.4f}",
            'Recall': f"{class_metrics['recall']:.4f}",
            'F1-Score': f"{class_metrics['f1-score']:.4f}",
            'Support': int(class_metrics['support'])
        })
    
    # Add macro and weighted averages
    report_data.append({
        'Class': 'macro avg',
        'Precision': f"{report['macro avg']['precision']:.4f}",
        'Recall': f"{report['macro avg']['recall']:.4f}",
        'F1-Score': f"{report['macro avg']['f1-score']:.4f}",
        'Support': int(report['macro avg']['support'])
    })
    report_data.append({
        'Class': 'weighted avg',
        'Precision': f"{report['weighted avg']['precision']:.4f}",
        'Recall': f"{report['weighted avg']['recall']:.4f}",
        'F1-Score': f"{report['weighted avg']['f1-score']:.4f}",
        'Support': int(report['weighted avg']['support'])
    })
    
    formatted['classification_report'] = pd.DataFrame(report_data)
    
    return formatted


def save_results_to_excel(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_prefix, dataset_result in results.items():
            # Classification results and distribution
            dataset_result['results'].to_excel(
                writer,
                sheet_name=f'{sheet_prefix}_results',
                index=False
            )
            dataset_result['distribution'].to_excel(
                writer,
                sheet_name=f'{sheet_prefix}_distribution',
                index=False
            )
            
            # Metrics
            dataset_result['metrics_summary'].to_excel(
                writer,
                sheet_name=f'{sheet_prefix}_metrics_summary',
                index=False
            )
            dataset_result['confusion_matrix'].to_excel(
                writer,
                sheet_name=f'{sheet_prefix}_confusion_matrix',
                index=False
            )
            dataset_result['classification_report'].to_excel(
                writer,
                sheet_name=f'{sheet_prefix}_class_report',
                index=False
            )

    print(f"Saved Naive Bayes output to: {output_file}")


def build_dataset_result(dataset_name, df, y):
    predictions = classify_dataset(df, y)
    
    # Classification results
    results_df = pd.DataFrame({
        'object': range(1, len(predictions) + 1),
        'true_class': y.values,
        'predicted_class': predictions,
    })
    
    # Distribution of predicted classes
    distribution_df = (
        results_df['predicted_class']
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={'index': 'class_group', 'predicted_class': 'count'})
    )
    
    # Compute and format metrics
    metrics = compute_metrics(y.values, predictions)
    formatted_metrics = format_metrics_for_excel(metrics)
    
    return {
        'results': results_df,
        'distribution': distribution_df,
        'metrics_summary': formatted_metrics['summary'],
        'confusion_matrix': formatted_metrics['confusion_matrix'],
        'classification_report': formatted_metrics['classification_report'],
    }


def main():
    datasets = ['X_scaled.csv', 'X_PCA.csv']
    y = load_labels()
    results = {}

    for dataset_name in datasets:
        df = load_features(dataset_name)
        results_key = dataset_name.replace('.csv', '')
        results[results_key] = build_dataset_result(dataset_name, df, y)

    save_results_to_excel(results, OUTPUT_FILE)


if __name__ == '__main__':
    main()
