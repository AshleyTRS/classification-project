import pandas as pd
import numpy as np
import os
import sys

from sklearn.cluster import DBSCAN

# --- Project path setup (same as your script) ---
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
sys.path.insert(0, project_root)

from src.evaluation.indexes import ClusteringEvaluator


# ---------------- CONFIG ----------------
EPS_MIN = 0.1
EPS_MAX = 8.0
EPS_TOL = 0.05     # stopping condition for binary search
MAX_DEPTH = 8      # how many recursive splits

MIN_SAMPLES_VALUES = [800, 1000, 1500, 1600, 2000,]


def load_data():
    """
    Loads training set
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    X_path = os.path.join(script_dir, '..', '..', '..', 'data', 'preprocessed', 'X_PCA.csv')
    X = pd.read_csv(X_path)
    dataset_name = os.path.basename(X_path)
    return X, dataset_name


def compute_metrics(X, labels):
    """
    DBSCAN metrics
    """
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


def score_configuration(row):
    """
    Score DBSCAN results with a preference of n_clusters = 3 (original data class labels)
    """
    if row["n_clusters"] < 2:
        return -np.inf

    score = 0

    # Prefer 3 clusters
    if row["n_clusters"] == 3:
        score += 100
    else:
        score -= abs(3 - row["n_clusters"]) * 20

    # Penalize noise
    score -= row["noise_pct"] * 0.5

    # Balance
    if row["balance_ratio"] is not None:
        score -= row["balance_ratio"] * 2

    # Metrics
    if row["silhouette"] is not None:
        score += row["silhouette"] * 50

    if row["davies_bouldin"] is not None:
        score -= row["davies_bouldin"] * 10

    if row["calinski_harabasz"] is not None:
        score += row["calinski_harabasz"] * 0.01

    return score


def binary_eps_search(X, min_samples, eps_min, eps_max, depth, results, visited):
    """
    Perform binary search sweep to find best 3_clusters params
    """
    if depth == 0 or abs(eps_max - eps_min) < EPS_TOL:
        return

    mid = round((eps_min + eps_max) / 2, 4)

    if (mid, min_samples) in visited:
        return

    visited.add((mid, min_samples))

    labels = DBSCAN(eps=mid, min_samples=min_samples).fit_predict(X.values)
    metrics = compute_metrics(X, labels)

    row = {
        "eps": mid,
        "min_samples": min_samples,
        **metrics
    }

    results.append(row)

    print(f"eps={mid:.4f}, ms={min_samples} -> {metrics['n_clusters']} clusters")

    # Heuristic branching:
    if metrics["n_clusters"] > 3:
        # Too many clusters → increase eps
        binary_eps_search(X, min_samples, mid, eps_max, depth - 1, results, visited)
    elif metrics["n_clusters"] < 3:
        # Too few clusters → decrease eps
        binary_eps_search(X, min_samples, eps_min, mid, depth - 1, results, visited)
    else:
        # Found 3 clusters → explore both sides
        binary_eps_search(X, min_samples, eps_min, mid, depth - 1, results, visited)
        binary_eps_search(X, min_samples, mid, eps_max, depth - 1, results, visited)

def main():
    X, dataset_name = load_data()
    print(f"Datos cargados: {X.shape}")

    all_results = []

    for ms in MIN_SAMPLES_VALUES:
        print(f"\n=== Searching for min_samples={ms} ===")
        visited = set()
        binary_eps_search(X, ms, EPS_MIN, EPS_MAX, MAX_DEPTH, all_results, visited)

    results_df = pd.DataFrame(all_results)
    results_df.insert(0, "dataset", dataset_name)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', '..', 'results', 'bitacora', 'bitacora-dbscan.xlsx'
    )

    sheet_name = "iteraciones_50"
    start_iteration = 1

    if os.path.exists(output_path):
        with pd.ExcelFile(output_path, engine="openpyxl") as xls:
            if sheet_name in xls.sheet_names:
                existing_df = pd.read_excel(xls, sheet_name=sheet_name)
                if "iteracion" in existing_df.columns and not existing_df["iteracion"].empty:
                    start_iteration = int(existing_df["iteracion"].max()) + 1
                results_df.insert(1, "iteracion", range(start_iteration, start_iteration + len(results_df)))
                results_df = pd.concat([existing_df, results_df], ignore_index=True)
            else:
                results_df.insert(1, "iteracion", range(start_iteration, start_iteration + len(results_df)))
    else:
        results_df.insert(1, "iteracion", range(start_iteration, start_iteration + len(results_df)))

    # Scoring
    results_df["score"] = results_df.apply(score_configuration, axis=1)

    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        results_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("\nResultados agregados correctamente (sin sobrescribir).")

    best = results_df.sort_values("score", ascending=False).head(10)

    print("\nTop 10 configuraciones:")
    print(best[[
        "eps", "min_samples", "n_clusters",
        "noise_pct", "balance_ratio", "score"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()