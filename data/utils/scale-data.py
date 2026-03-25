import os
from sklearn.preprocessing import StandardScaler
import pandas as pd

def save_scaled_data(X_scaled, path=None):
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, '..', 'preprocessed', 'X_scaled.csv')
    pd.DataFrame(X_scaled).to_csv(path, index=False)

def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, '..', 'raw', 'X.csv')
    X = pd.read_csv(raw_path)

    X_scaled, scaler = scale_data(X)
    save_scaled_data(X_scaled)

if __name__ == "__main__":
    main()
    