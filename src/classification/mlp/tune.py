import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE = "../../../"
raw = pd.read_csv(BASE + "data/preprocessed/X_scaled.csv", header=None)
if (raw.iloc[0].values == np.arange(raw.shape[1])).all():
    raw = raw.iloc[1:]
X = raw.values.astype(float)
y = pd.read_csv(BASE + "data/raw/y.csv").values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

configs = [
    {"hidden_layer_sizes": (256, 128, 64), "activation": "relu",  "max_iter": 1000, "early_stopping": False},
    {"hidden_layer_sizes": (256, 128, 64), "activation": "tanh",  "max_iter": 1000, "early_stopping": False},
    {"hidden_layer_sizes": (512, 256, 128),"activation": "relu",  "max_iter": 1000, "early_stopping": False},
    {"hidden_layer_sizes": (128, 64, 32),  "activation": "tanh",  "max_iter": 1000, "early_stopping": False},
    {"hidden_layer_sizes": (256, 128),     "activation": "tanh",  "max_iter": 1000, "early_stopping": False},
]

best = 0
for cfg in configs:
    m = MLPClassifier(solver="adam", random_state=42, **cfg)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    tag = " ** MEJOR **" if acc > best else ""
    if acc > best:
        best = acc
    print(f"capas={cfg['hidden_layer_sizes']} act={cfg['activation']}  acc={acc:.4f}  iters={m.n_iter_}{tag}")

print(f"\nMejor: {best:.4f}")
