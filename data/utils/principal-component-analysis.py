import os
from sklearn.decomposition import PCA
import pandas as pd

def transform_data(X_scaled):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca

def save_transformed_data(X_scaled, path=None):
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, '..', 'preprocessed', 'X_PCA.csv')
    pd.DataFrame(X_scaled).to_csv(path, index=False)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scaled_path = os.path.join(script_dir, '..', 'preprocessed', 'X_scaled.csv')
    X_scaled = pd.read_csv(scaled_path)

    X_pca = transform_data(X_scaled)
    save_transformed_data(X_pca)
    print("ok")

if __name__ == "__main__":
    main()