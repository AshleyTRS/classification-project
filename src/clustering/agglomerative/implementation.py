import pandas as pd
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
class AgglomerativeModel:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.labels = None

    def train(self, X):
        model = AgglomerativeClustering(n_clusters=self.n_clusters)
        self.labels = model.fit_predict(X)
        return self.labels

    def print_clusters(self):
        if self.labels is None:
            print("Model has not been trained yet.")
            return

        print("\nCluster summary:")

        for cluster_id in range(self.n_clusters):
            indices = np.where(self.labels == cluster_id)[0]

            print(f"\nCluster {cluster_id}:")
            print(f"Number of points: {len(indices)}")
            print(f"Indices: {indices}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scaled_path = os.path.join(script_dir, '..', '..', '..', 'data', 'preprocessed', 'X_scaled.csv')
    
    X_scaled = pd.read_csv(scaled_path)

    model = AgglomerativeModel(n_clusters=4)

    model.train(X_scaled)
    model.print_clusters()


if __name__ == "__main__":
    main()