import pandas as pd
import numpy as np
import os
import sys
from itertools import product

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
sys.path.insert(0, project_root)

from sklearn.cluster import DBSCAN
from src.evaluation.indexes import ClusteringEvaluator

EPS_VALUES = [0.5, 1.0, 1.5, 2.0, 2.4, 2.76, 3.0, 3.5, 4.5, 6.0]
MIN_SAMPLES_VALUES = [2, 3, 4, 5, 10]

CONFIGS = [
    {"eps": eps, "min_samples": ms}
    for eps, ms in product(EPS_VALUES, MIN_SAMPLES_VALUES)
]


def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    X_path = os.path.join(script_dir, '..', '..', '..', 'data', 'preprocessed', 'X_scaled.csv')
    X = pd.read_csv(X_path)
    return X


def compute_metrics(X, labels):
    mask = labels != -1
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    n_noise = int((labels == -1).sum())
    noise_pct = round(100 * n_noise / len(labels), 2)

    cluster_sizes = []
    cluster_sizes_str = ""
    cluster_min = None
    cluster_max = None
    cluster_mean = None
    cluster_std = None
    balance_ratio = None

    if n_clusters > 0:
        for cid in sorted(set(labels[mask])):
            cluster_sizes.append(int(np.sum(labels == cid)))
        cluster_sizes_str = ",".join(str(s) for s in sorted(cluster_sizes, reverse=True))
        cluster_min = min(cluster_sizes)
        cluster_max = max(cluster_sizes)
        cluster_mean = round(float(np.mean(cluster_sizes)), 2)
        cluster_std = round(float(np.std(cluster_sizes)), 2)
        balance_ratio = round(cluster_max / cluster_min, 2) if cluster_min > 0 else None

    if mask.sum() == 0 or n_clusters < 2:
        return {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_pct": noise_pct,
            "cluster_min_size": cluster_min,
            "cluster_max_size": cluster_max,
            "cluster_mean_size": cluster_mean,
            "cluster_std_size": cluster_std,
            "balance_ratio": balance_ratio,
            "cluster_sizes": cluster_sizes_str,
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
        "cluster_min_size": cluster_min,
        "cluster_max_size": cluster_max,
        "cluster_mean_size": cluster_mean,
        "cluster_std_size": cluster_std,
        "balance_ratio": balance_ratio,
        "cluster_sizes": cluster_sizes_str,
        "davies_bouldin": round(results["davies_bouldin"], 4),
        "calinski_harabasz": round(results["calinski_harabasz"], 4),
        "silhouette": round(results["silhouette"], 4) if results["silhouette"] is not None else None,
    }


def main():
    X = load_data()
    print(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características.")
    print(f"Corriendo {len(CONFIGS)} iteraciones...\n")

    rows = []
    for i, cfg in enumerate(CONFIGS, start=1):
        labels = DBSCAN(eps=cfg["eps"], min_samples=cfg["min_samples"]).fit_predict(X.values)
        metrics = compute_metrics(X, labels)

        row = {
            "iteracion": i,
            "dataset": "X_scaled.csv",
            "eps": cfg["eps"],
            "min_samples": cfg["min_samples"],
            **metrics,
        }
        rows.append(row)

        print(
            f"[{i:02d}] eps={cfg['eps']}, ms={cfg['min_samples']:2d} -> "
            f"{metrics['n_clusters']} clusters, "
            f"ruido={metrics['noise_pct']}%, "
            f"balance_ratio={metrics['balance_ratio']}, "
            f"sizes=[{metrics['cluster_sizes']}]",
            flush=True
        )

    results_df = pd.DataFrame(rows)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', '..', 'results', 'bitacora', 'bitacora-dbscan.xlsx'
    )
    sheet_name = "iteraciones_50"

    if os.path.exists(output_path):
        with pd.ExcelFile(output_path, engine="openpyxl") as xls:
            if sheet_name in xls.sheet_names:
                existing_df = pd.read_excel(xls, sheet_name=sheet_name)
                results_df = pd.concat([existing_df, results_df], ignore_index=True)

    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        results_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\nResultados guardados en hoja '{sheet_name}': {output_path}")
    print(f"\n--- Resumen rápido ---")
    balanced = results_df[
        (results_df["n_clusters"] >= 2) &
        (results_df["balance_ratio"].notna()) &
        (results_df["balance_ratio"] <= 10)
    ].sort_values("balance_ratio")
    if balanced.empty:
        print("No se encontraron configs con balance_ratio <= 10.")
    else:
        print(balanced[["iteracion", "eps", "min_samples", "n_clusters", "noise_pct", "balance_ratio", "cluster_sizes"]].to_string(index=False))


if __name__ == "__main__":
    main()
