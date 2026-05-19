import numpy as np
import pandas as pd
import sys
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..", "..", "..")

RAW_X      = os.path.join(BASE, "data", "raw", "X.csv")
SCALED_X   = os.path.join(BASE, "data", "preprocessed", "X_scaled.csv")
PCA_X      = os.path.join(BASE, "data", "preprocessed", "X_PCA.csv")
LABELS_Y   = os.path.join(BASE, "data", "raw", "y.csv")

RESULTS_DIR = os.path.join(BASE, "results", "bitacora")


def load_data(dataset: str = "scaled"):
    """
    dataset: 'raw' | 'scaled' | 'pca'
    Devuelve (X: ndarray, y: ndarray)
    """
    y = pd.read_csv(LABELS_Y).values.ravel()

    if dataset == "raw":
        X = pd.read_csv(RAW_X).values
    elif dataset == "pca":
        raw = pd.read_csv(PCA_X, header=None)
        # Primera fila es índice de columnas guardado como dato → la quitamos
        try:
            float(raw.iloc[0, 0])
            is_index_row = (raw.iloc[0].values == np.arange(raw.shape[1])).all()
        except Exception:
            is_index_row = False
        if is_index_row:
            raw = raw.iloc[1:]
        X = raw.values.astype(float)
    else:  # scaled (default)
        raw = pd.read_csv(SCALED_X, header=None)
        try:
            is_index_row = (raw.iloc[0].values == np.arange(raw.shape[1])).all()
        except Exception:
            is_index_row = False
        if is_index_row:
            raw = raw.iloc[1:]
        X = raw.values.astype(float)

    assert len(X) == len(y), f"Shapes no coinciden: X={len(X)}, y={len(y)}"
    return X, y


def train_mlp(X_train, y_train, hidden_layers=(100, 50), max_iter=500):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Clase 0", "Clase 1", "Clase 2"])
    cm = confusion_matrix(y_test, y_pred)
    return acc, report, cm, y_pred


def save_results(dataset_name: str, acc: float, report: str, cm, hidden_layers):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    arch_str = "-".join(str(n) for n in hidden_layers)
    fname = f"mlp_{dataset_name}_arch{arch_str}.txt"
    path = os.path.join(RESULTS_DIR, fname)

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== MLP — Red Neuronal Supervisada ===\n")
        f.write(f"Dataset      : {dataset_name}\n")
        f.write(f"Arquitectura : {hidden_layers}\n")
        f.write(f"Exactitud    : {acc:.4f}  ({acc*100:.2f}%)\n\n")
        f.write("Reporte por clase:\n")
        f.write(report + "\n")
        f.write("Matriz de confusión:\n")
        f.write(str(cm) + "\n")

    print(f"Resultados guardados: {path}")
    return path


def run(dataset: str = "scaled", hidden_layers=(100, 50), test_size: float = 0.2):
    print(f"\n{'='*55}")
    print(f" MLP  |  dataset={dataset}  |  capas={hidden_layers}")
    print(f"{'='*55}")

    X, y = load_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    print("Entrenando...")
    model = train_mlp(X_train, y_train, hidden_layers=hidden_layers)
    print(f"Iteraciones: {model.n_iter_}")

    acc, report, cm, _ = evaluate(model, X_test, y_test)

    print(f"\nExactitud: {acc:.4f}  ({acc*100:.2f}%)")
    print("\nReporte por clase:")
    print(report)
    print("Matriz de confusión:")
    print(cm)

    save_results(dataset, acc, report, cm, hidden_layers)
    return acc, model


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "scaled"

    configs = [
        ((100, 50),),
        ((128, 64, 32),),
        ((200, 100),),
    ]

    best_acc = 0
    best_cfg = None
    for (layers,) in configs:
        acc, _ = run(dataset=dataset, hidden_layers=layers)
        if acc > best_acc:
            best_acc = acc
            best_cfg = layers

    print(f"\n>>> Mejor configuración: capas={best_cfg}  exactitud={best_acc:.4f}")
