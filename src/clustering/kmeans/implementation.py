import pandas as pd
from sklearn.cluster import KMeans

def run_kmeans(k=2):
    
    X = pd.read_csv("data/preprocessed/X_scaled.csv")

    
    model = KMeans(n_clusters=k, random_state=42, n_init=10)

    
    labels = model.fit_predict(X)

    
    X["cluster"] = labels
    output_path = f"results/bitacora/kmeans_k{k}.csv"
    X.to_csv(output_path, index=False)

    print(f"KMeans ejecutado con k={k}")
    print(f"Resultados guardados en {output_path}")

    return model.cluster_centers_
