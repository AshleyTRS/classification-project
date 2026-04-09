import pandas as pd
from sklearn.cluster import KMeans

# 📂 Cargar datos escalados
X = pd.read_csv("data/preprocessed/X_scaled.csv")

# ⚙️ Número de clusters
k = 3  # cámbialo si quieres

# 🤖 Modelo
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

# 🚀 Entrenar
labels = kmeans.fit_predict(X)

# 📊 Mostrar resultados básicos
print("Clusters asignados:")
print(labels)

print("\nCentroides:")
print(kmeans.cluster_centers_)

# 💾 Guardar resultados
X["cluster"] = labels
X.to_csv("results_kmeans.csv", index=False)

print("\nArchivo generado: results_kmeans.csv")
