import numpy as np
import os
from sklearn.metrics import (
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_score
)

class ClusteringEvaluator:
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels

    def compute_db_index(self):
        return davies_bouldin_score(self.X, self.labels)

    def compute_ch_index(self):
        return calinski_harabasz_score(self.X, self.labels)

    def compute_silhouette_score(self):
        # Silhouette requires at least 2 clusters
        if len(set(self.labels)) < 2:
            return None
        return silhouette_score(self.X, self.labels)

    def evaluate_all(self):
        results = {
            "davies_bouldin": self.compute_db_index(),
            "calinski_harabasz": self.compute_ch_index(),
            "silhouette": self.compute_silhouette_score()
        }
        return results

    def save_results(self, dataset_name, k_num):
        file_name = dataset_name + "_k_" + k_num + "_eval_index.txt"
        results = self.evaluate_all()

        # Create output directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '..', '..', 'results', 'bitacora')
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, file_name)

        # Write results to file
        with open(output_path, 'w') as f:
            f.write("Evaluation Metrics:\n")
            f.write(f"Davies-Bouldin: {results['davies_bouldin']:.4f}\n")
            f.write(f"Calinski-Harabasz : {results['calinski_harabasz']:.4f}\n")
            if results['silhouette'] is not None:
                f.write(f"Silhouette Score: {results['silhouette']:.4f}\n")
            else:
                f.write("Silhouette Score: N/A\n")

        print(f"Results saved to: {output_path}")