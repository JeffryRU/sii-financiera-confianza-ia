"""
Ejemplo de DEEP LEARNING – Autoencoder para detección de anomalías (fraude)
Caso: Financiera Confianza Perú

Idea
----
Un autoencoder es una red neuronal profunda que aprende a COMPRIMIR y RECONSTRUIR
las solicitudes legítimas. Como nunca vio fraude durante el entrenamiento, reconstruye
mal las solicitudes atípicas: el error de reconstrucción funciona como puntaje de anomalía.
Ventaja frente al modelo supervisado: detecta patrones de fraude NUEVOS que aún no
tienen etiqueta (fraude de "día cero").

Arquitectura (PyTorch):
    entrada(25) -> 32 -> 16 -> 6 (cuello de botella) -> 16 -> 32 -> salida(25)
    activación ReLU, pérdida MSE, optimizador Adam, early stopping sobre validación.

Ejecución:
    python 03_dl_autoencoder/dl_autoencoder.py
Salidas:
    figures/dl_*.png, resultados/dl_metricas.json, resultados/dl_autoencoder.pt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from comun.datos_sinteticos import VARIABLES_CATEGORICAS, VARIABLES_NUMERICAS, generar_solicitudes  # noqa: E402

FIG = RAIZ / "figures"; RES = RAIZ / "resultados"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
SEMILLA = 42
torch.manual_seed(SEMILLA); np.random.seed(SEMILLA)


class Autoencoder(nn.Module):
    """Red codificador-decodificador simétrica."""

    def __init__(self, d: int, latente: int = 6):
        super().__init__()
        self.codificador = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(),
                                         nn.Linear(16, latente))
        self.decodificador = nn.Sequential(nn.Linear(latente, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(),
                                           nn.Linear(32, d))

    def forward(self, x):
        return self.decodificador(self.codificador(x))


def error_reconstruccion(modelo: nn.Module, X: np.ndarray) -> np.ndarray:
    modelo.eval()
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32)
        return ((modelo(t) - t) ** 2).mean(dim=1).numpy()


def main(epocas: int = 120, paciencia: int = 12) -> None:
    df = generar_solicitudes(n=20000, semilla=SEMILLA)
    X = df[VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS]; y = df.fraude.values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEMILLA)

    # Normalización robusta: cuantiles -> distribución normal (maneja colas largas de montos)
    pre = ColumnTransformer([
        ("num", QuantileTransformer(output_distribution="normal", random_state=SEMILLA), VARIABLES_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), VARIABLES_CATEGORICAS)])
    legit = X_tr[y_tr == 0]
    pre.fit(legit)
    A = pre.transform(legit).astype("float32")
    A_tr, A_val = train_test_split(A, test_size=0.15, random_state=SEMILLA)
    B_te = pre.transform(X_te).astype("float32")

    modelo = Autoencoder(A.shape[1])
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3, weight_decay=1e-5)
    perdida = nn.MSELoss()
    cargador = torch.utils.data.DataLoader(torch.tensor(A_tr), batch_size=256, shuffle=True)
    hist_tr, hist_val, mejor, sin_mejora, mejor_estado = [], [], np.inf, 0, None
    for ep in range(epocas):
        modelo.train(); acum = 0.0
        for lote in cargador:
            opt.zero_grad(); l = perdida(modelo(lote), lote); l.backward(); opt.step()
            acum += l.item() * len(lote)
        hist_tr.append(acum / len(A_tr))
        val = float(error_reconstruccion(modelo, A_val).mean()); hist_val.append(val)
        if val < mejor - 1e-5:
            mejor, sin_mejora = val, 0
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    modelo.load_state_dict(mejor_estado)
    print(f"Entrenamiento detenido en la época {len(hist_tr)} (mejor MSE validación = {mejor:.4f})")

    err = error_reconstruccion(modelo, B_te)
    umbral = float(np.quantile(error_reconstruccion(modelo, A_val), 0.99))  # 1 % de falsas alarmas esperado
    pred = (err > umbral).astype(int)
    fpr, tpr, _ = roc_curve(y_te, err)
    met = {"roc_auc": float(roc_auc_score(y_te, err)), "pr_auc": float(average_precision_score(y_te, err)),
           "recall_fpr1": float(np.interp(0.01, fpr, tpr)),
           "umbral_p99": umbral, "recall_umbral": float(pred[y_te == 1].mean()),
           "falsas_alarmas_umbral": float(pred[y_te == 0].mean()), "epocas": len(hist_tr),
           "parametros": int(sum(p.numel() for p in modelo.parameters()))}
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in met.items()})

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist_tr, label="Entrenamiento"); ax[0].plot(hist_val, label="Validación")
    ax[0].set(title="Curva de aprendizaje del autoencoder", xlabel="Época", ylabel="MSE"); ax[0].legend()
    bins = np.logspace(np.log10(err.min() + 1e-4), np.log10(err.max()), 50)
    ax[1].hist(err[y_te == 0], bins=bins, alpha=.6, density=True, label="Legítimas")
    ax[1].hist(err[y_te == 1], bins=bins, alpha=.6, density=True, label="Fraude")
    ax[1].axvline(umbral, c="k", ls="--", label="Umbral (p99 legítimas)")
    ax[1].set_xscale("log")
    ax[1].set(title=f"Error de reconstrucción (ROC-AUC {met['roc_auc']:.3f})", xlabel="MSE por solicitud (log)",
              ylabel="Densidad"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "dl_autoencoder.png", dpi=160); plt.close(fig)

    torch.save(modelo.state_dict(), RES / "dl_autoencoder.pt")
    (RES / "dl_metricas.json").write_text(json.dumps(met, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
