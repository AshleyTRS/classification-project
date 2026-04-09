import pandas as pd
import numpy as np
import os
import sys

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
sys.path.insert(0, project_root)

from sklearn.cluster import DBSCAN
from src.evaluation.indexes import ClusteringEvaluator


# Las tres configuraciones a evaluar:
#   1. Óptima  : eps=2.76, min_samples=4  → 3 clusters, ~37 % ruido (coincide con 3 clases reales)
#   2. Config 2: eps=2.80, min_samples=4  → 2 clusters, ~32 % ruido (eps ligeramente mayor, fusiona más)
#   3. Config 3: eps=2.60, min_samples=3  → ~32 clusters, ~56 % ruido (eps menor, más fragmentado)
CONFIGS = [
    {"label": "dbscan_eps2.76_ms4", "eps": 2.76, "min_samples": 4},
    {"label": "dbscan_eps2.80_ms4", "eps": 2.80, "min_samples": 4},
    {"label": "dbscan_eps2.60_ms3", "eps": 2.60, "min_samples": 3},
]


def load_data(dataset, location):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    X_path = os.path.join(script_dir, '..', '..', '..', 'data', location, dataset)
    y_path = os.path.join(script_dir, '..', '..', '..', 'data', 'raw', 'y.csv')
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    return X, y


def run_dbscan(X, eps, min_samples):
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X.values)
    return labels


def compute_metrics(X, labels):
    """Calcula métricas excluyendo puntos de ruido (label == -1)."""
    mask = labels != -1
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    n_noise = int((labels == -1).sum())
    noise_pct = round(100 * n_noise / len(labels), 2)

    if mask.sum() == 0 or n_clusters < 2:
        return {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_pct": noise_pct,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "silhouette": None,
        }

    evaluator = ClusteringEvaluator(X.values[mask], labels[mask])
    results = evaluator.evaluate_all()
    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": noise_pct,
        "davies_bouldin": round(results["davies_bouldin"], 4),
        "calinski_harabasz": round(results["calinski_harabasz"], 4),
        "silhouette": round(results["silhouette"], 4) if results["silhouette"] is not None else None,
    }


def main():
    datasets = {"X_scaled.csv": "preprocessed"}

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', '..', 'results', 'bitacora', 'bitacora-dbscan.xlsx'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_metrics = []

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for dataset_name, location in datasets.items():
            X, y = load_data(dataset=dataset_name, location=location)

            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]

            results_df = X.copy()
            results_df["original_label"] = y.values

            for cfg in CONFIGS:
                labels = run_dbscan(X, cfg["eps"], cfg["min_samples"])
                results_df[cfg["label"]] = labels

                metrics = compute_metrics(X, labels)
                all_metrics.append({
                    "dataset": dataset_name,
                    "config": cfg["label"],
                    "eps": cfg["eps"],
                    "min_samples": cfg["min_samples"],
                    **metrics,
                })

                print(
                    f"[{dataset_name}] {cfg['label']}: "
                    f"{metrics['n_clusters']} clusters, "
                    f"{metrics['n_noise']} ruido ({metrics['noise_pct']}%), "
                    f"DB={metrics['davies_bouldin']}, "
                    f"CH={metrics['calinski_harabasz']}, "
                    f"Sil={metrics['silhouette']}"
                )

            sheet_name = dataset_name.replace(".csv", "")
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Hoja de métricas de evaluación
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_excel(writer, sheet_name="metricas_evaluacion", index=False)

    print(f"\nBitácora guardada en: {output_path}")


if __name__ == "__main__":
    main()
