import pandas as pd
import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class DBSCANModel:
    def __init__(self, eps=2.76, min_samples=4):
        """
        Inicializa el modelo DBSCAN.

        Args:
            eps (float): Radio de vecindad. Valor recomendado para Waveform (21D escalado): 2.76.
                         Si es None, se estima automáticamente via k-distance (resultado variable).
            min_samples (int): Número mínimo de puntos para formar un cluster denso.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.core_sample_indices = None
        self.n_clusters_ = 0
        self.n_noise_ = 0

    def train(self, X):
        """
        Entrena el modelo DBSCAN en los datos X.
        
        Args:
            X (array-like): Datos preprocesados (shape n_samples x n_features).
        
        Returns:
            array: Etiquetas de cluster para cada punto (-1 indica ruido).
        """
        # Si eps no está definido, estimarlo usando la gráfica k‑distance
        if self.eps is None:
            self.eps = self._estimate_eps(X, k=self.min_samples - 1)
            print(f"eps estimado automáticamente: {self.eps:.4f}")
        
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = model.fit_predict(X)
        self.core_sample_indices = model.core_sample_indices_
        
        # Contar clusters y ruido
        unique_labels = set(self.labels)
        self.n_clusters_ = len([l for l in unique_labels if l != -1])
        self.n_noise_ = list(self.labels).count(-1)
        
        return self.labels

    def _estimate_eps(self, X, k=4):
        """
        Estima un valor razonable para eps mediante la gráfica k‑distance.
        
        Args:
            X (array-like): Datos.
            k (int): Número de vecinos más cercanos (generalmente min_samples-1).
        
        Returns:
            float: eps sugerido (percentil 90 de las distancias al k‑ésimo vecino).
        """
        neigh = NearestNeighbors(n_neighbors=k + 1)
        neigh.fit(X)
        distances, _ = neigh.kneighbors(X)
        k_distances = distances[:, -1]  # distancia al k‑ésimo vecino
        k_distances_sorted = np.sort(k_distances)
        
        # Percentil 40 identifica el codo de la curva para datasets de alta dimensión
        # con clases solapadas (ej. Waveform 21D). Percentiles altos (~90) fusionan
        # todo en un único cluster en este tipo de datos.
        eps_estimate = np.percentile(k_distances_sorted, 40)
        
        # Opcional: graficar k‑distance para inspección visual (descomentar si se tiene matplotlib)
        # if False:  # cambiar a True para debug
        #     import matplotlib.pyplot as plt
        #     plt.plot(k_distances_sorted)
        #     plt.axhline(y=eps_estimate, color='r', linestyle='--', label=f'eps estimado = {eps_estimate:.3f}')
        #     plt.xlabel('Puntos ordenados')
        #     plt.ylabel(f'Distancia al {k}‑ésimo vecino')
        #     plt.title('Gráfica k‑distance para estimar eps')
        #     plt.legend()
        #     plt.show()
        
        return eps_estimate

    def print_clusters(self):
        """
        Imprime un resumen de los clusters encontrados.
        """
        if self.labels is None:
            print("El modelo aún no ha sido entrenado.")
            return

        print(f"\nDBSCAN - Resumen:")
        print(f"  Número de clusters encontrados: {self.n_clusters_}")
        print(f"  Número de puntos de ruido (outliers): {self.n_noise_}")
        print(f"  Parámetros: eps={self.eps:.4f}, min_samples={self.min_samples}")
        
        if self.n_clusters_ > 0:
            print("\n  Distribución de puntos por cluster:")
            for cluster_id in range(self.n_clusters_):
                count = np.sum(self.labels == cluster_id)
                print(f"    Cluster {cluster_id}: {count} puntos")
        else:
            print("  No se encontraron clusters densos con los parámetros actuales.")
        
        if self.n_noise_ > 0:
            print(f"\n  Ruido (cluster -1): {self.n_noise_} puntos")

    def get_cluster_indices(self):
        """
        Devuelve un diccionario con los índices de cada cluster.
        
        Returns:
            dict: {cluster_id: lista_de_índices}
        """
        if self.labels is None:
            return {}
        
        clusters = {}
        for cluster_id in range(self.n_clusters_):
            clusters[cluster_id] = np.where(self.labels == cluster_id)[0].tolist()
        
        if self.n_noise_ > 0:
            clusters[-1] = np.where(self.labels == -1)[0].tolist()
        
        return clusters


def main():
    """
    Ejemplo de uso: carga los datos escalados, entrena DBSCAN y muestra resultados.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scaled_path = os.path.join(script_dir, '..', '..', '..', 'data', 'preprocessed', 'X_scaled.csv')
    
    # Cargar datos escalados
    X_scaled = pd.read_csv(scaled_path)
    print(f"Datos cargados: {X_scaled.shape[0]} muestras, {X_scaled.shape[1]} características.")
    
    # Parámetros: ajustar según el dataset
    # Para waveform (21D, ruido moderado) empezar con min_samples=5 y eps estimado automáticamente
    model = DBSCANModel(eps=2.76, min_samples=4)
    
    # Entrenar
    labels = model.train(X_scaled)
    
    # Mostrar resumen
    model.print_clusters()
    
    # Evaluación (opcional)
    from src.evaluation.indexes import ClusteringEvaluator
    
    # Excluir puntos de ruido (label=-1) de las métricas de evaluación
    valid_mask = labels != -1
    evaluator = ClusteringEvaluator(X_scaled.values[valid_mask], labels[valid_mask])
    results = evaluator.evaluate_all()
    print("\nMétricas de evaluación:")
    print(f"  Davies‑Bouldin: {results['davies_bouldin']:.4f}")
    print(f"  Calinski‑Harabasz: {results['calinski_harabasz']:.4f}")
    if results['silhouette'] is not None:
        print(f"  Silhouette: {results['silhouette']:.4f}")
    else:
        print("  Silhouette: N/A (solo un cluster o solo ruido)")


if __name__ == "__main__":
    main()