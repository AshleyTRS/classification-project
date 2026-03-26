import pandas as pd
import os
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster


def load_data(dataset, location):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    X_path = os.path.join(script_dir, '..', '..', '..', 'data', location, dataset)
    y_path = os.path.join(script_dir, '..', '..', '..', 'data', 'raw', 'y.csv')

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    return X, y


def run_agglomerative(X, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(X)


def run_dendrogram_clustering(X):
    Z = linkage(X, method='ward')

    # Tchoose threshold create clusters based on distance
    labels = fcluster(Z, t=10, criterion='distance')

    # Make zero-indexed
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
            labels_k3 = run_agglomerative(X_scaled, 3)
            labels_k4 = run_agglomerative(X_scaled, 4)
            labels_dendro = run_dendrogram_clustering(X_scaled)

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