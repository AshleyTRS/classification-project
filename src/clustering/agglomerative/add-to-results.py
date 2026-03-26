import pandas as pd
import os
import sys
from scipy.cluster.hierarchy import linkage, fcluster

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
sys.path.insert(0, project_root)

from src.evaluation.indexes import ClusteringEvaluator
from src.clustering.agglomerative.implementation import AgglomerativeModel


def load_data(dataset, location):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    X_path = os.path.join(script_dir, '..', '..', '..', 'data', location, dataset)
    y_path = os.path.join(script_dir, '..', '..', '..', 'data', 'raw', 'y.csv')

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    return X, y


def run_agglomerative(X, n_clusters, dataset, k_num):
    model = AgglomerativeModel(n_clusters=n_clusters)
    labels = model.train(X)
    evaluator = ClusteringEvaluator(X, labels)
    evaluator.save_results(dataset, k_num)
    return labels


def run_dendrogram_clustering(X,  dataset, k_num):
    Z = linkage(X, method='ward')

    # Choose threshold create clusters based on distance
    labels = fcluster(Z, t=10, criterion='distance')

    evaluator = ClusteringEvaluator(X, labels)
    evaluator.save_results(dataset, k_num)

    return labels - 1


def main():
    datasets = {'X_scaled.csv': 'preprocessed', 'X_PCA.csv': 'preprocessed', 'X.csv': 'raw'}

    with pd.ExcelWriter(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'results', 'bitacora', 'bitacora-aglomerativo.xlsx')) as writer:
        for dataset_name, location in datasets.items():
            # Load data
            X_scaled, y = load_data(dataset=dataset_name, location=location)

            # original class labels
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]

            # run clustering for dataset
            labels_k3 = run_agglomerative(X_scaled, 3, dataset_name, "3")
            labels_k4 = run_agglomerative(X_scaled, 4, dataset_name, "4")
            labels_dendro = run_dendrogram_clustering(X_scaled, dataset_name, "dendogram")

            # build result dataframe
            results_df = X_scaled.copy()

            results_df['original_label'] = y
            results_df['agglo_k3'] = labels_k3
            results_df['agglo_k4'] = labels_k4
            results_df['agglo_dendrogram'] = labels_dendro

            # Generate sheet name from dataset
            sheet_name = dataset_name.replace('.csv', '')

            os.makedirs(os.path.dirname(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'results', 'bitacora')), exist_ok=True)

            results_df.to_excel(writer, sheet_name=sheet_name, index=False)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'results', 'bitacora', 'bitacora-aglomerativo.xlsx')
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()