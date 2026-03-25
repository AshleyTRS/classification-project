from ucimlrepo import fetch_ucirepo
import os

def save_raw_data(X, y, path="data/raw/"):
    os.makedirs(path, exist_ok=True)
    
    X.to_csv(os.path.join(path, "X.csv"), index=False)
    y.to_csv(os.path.join(path, "y.csv"), index=False) # this data will be ignored during unsupervised learning

def load_waveform_data():
    dataset = fetch_ucirepo(id=107)
    
    X = dataset.data.features # input data for unsupervised training
    y = dataset.data.targets  # data labels/targets/classes will not be used for training
    
    return X, y

def main():
    X, y = load_waveform_data()
    save_raw_data(X, y)

if __name__ == "__main__":
    main()