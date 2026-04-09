import pandas as pd
from sklearn.cluster import KMeans


X = pd.read_csv("data/preprocessed/X_scaled.csv")


k = 4

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)


labels = kmeans.fit_predict(X)


print("Clusters asignados:")
print(labels)

print("\nCentroides:")
print(kmeans.cluster_centers_)

# 💾 Guardar resultados
X["cluster"] = labels
X.to_csv("results_kmeans.csv", index=False)

print("\nArchivo generado: results_kmeans.csv")
