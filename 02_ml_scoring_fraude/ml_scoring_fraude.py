"""
Componente 2 – Modelo de Machine Learning integrado al ERP (Core financiero)
Caso: Financiera Confianza Perú

Objetivo
--------
1. Detectar fraude en solicitudes digitales de microcrédito (clasificación supervisada
   + detección de anomalías no supervisada).
2. Calcular un scoring crediticio dinámico (probabilidad de mora > 90 días convertida a
   un puntaje 300–850) que se recalcula cada vez que cambian los datos del cliente.
3. Combinar ambos en una DECISIÓN que el ERP puede consumir por API:
   APROBAR / REVISION_MANUAL / RECHAZAR_Y_ALERTAR.

Comparación con la situación actual: se reproduce una "regla estática" típica
(p. ej. rechazar si score_buro < 500 o si ratio cuota/ingreso > 0.45 o si la
coincidencia biométrica < 0.70) para medir cuánto fraude deja pasar.

Ejecución:
    python 02_ml_scoring_fraude/ml_scoring_fraude.py
Salidas:
    figures/ml_*.png, resultados/ml_metricas.json, resultados/ml_modelo_fraude.joblib
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve, recall_score,
                             roc_auc_score, roc_curve, precision_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from comun.datos_sinteticos import VARIABLES_CATEGORICAS, VARIABLES_NUMERICAS, generar_solicitudes  # noqa: E402

FIG = RAIZ / "figures"; RES = RAIZ / "resultados"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
SEMILLA = 42


def preprocesador(escalar: bool) -> ColumnTransformer:
    """One-hot para categóricas; escalado opcional (necesario para regresión logística)."""
    num = StandardScaler() if escalar else "passthrough"
    return ColumnTransformer([
        ("num", num, VARIABLES_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), VARIABLES_CATEGORICAS),
    ])


def regla_estatica(df: pd.DataFrame) -> np.ndarray:
    """Regla fija que hoy usa la financiera (situación AS-IS). 1 = marcar como sospechoso."""
    return ((df.score_buro < 500) | (df.ratio_cuota_ingreso > 0.60)
            | (df.coincidencia_biometrica < 0.70)).astype(int).values


def recall_a_fpr(y, p, fpr_objetivo=0.01) -> float:
    """Recall (fraude detectado) cuando solo se tolera 1 % de falsas alarmas."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_objetivo, fpr, tpr))


def prob_a_score(pd_: np.ndarray, base=600, odds_base=20, pdo=40) -> np.ndarray:
    """Convierte probabilidad de mora (PD) en puntaje tipo buró 300-850.
    Convención 'points to double the odds' (PDO): 600 puntos = odds 20:1, +40 puntos duplican las odds."""
    pd_ = np.clip(pd_, 1e-4, 1 - 1e-4)
    odds = (1 - pd_) / pd_
    factor = pdo / np.log(2)
    offset = base - factor * np.log(odds_base)
    return np.clip(offset + factor * np.log(odds), 300, 850).round()


def decidir(p_fraude: float, score: float, u_alto=0.50, u_rev=0.08, score_min=520) -> str:
    """Política de decisión que el ERP ejecuta con la respuesta del servicio de IA."""
    if p_fraude >= u_alto:
        return "RECHAZAR_Y_ALERTAR"
    if p_fraude >= u_rev or score < score_min:
        return "REVISION_MANUAL"
    return "APROBAR"


def main() -> None:
    df = generar_solicitudes(n=20000, semilla=SEMILLA)
    df.to_csv(RAIZ / "resultados" / "solicitudes_sinteticas.csv", index=False)
    X = df[VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS]
    y = df.fraude
    X_tr, X_te, y_tr, y_te, df_tr, df_te = train_test_split(X, y, df, test_size=0.3, stratify=y, random_state=SEMILLA)

    # ------------------------------------------------------------------ FRAUDE
    modelos = {
        "Regresión logística": Pipeline([("pre", preprocesador(True)),
                                         ("m", LogisticRegression(max_iter=2000))]),
        "Random Forest": Pipeline([("pre", preprocesador(False)),
                                   ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                                n_jobs=-1,
                                                                random_state=SEMILLA))]),
        "Gradient Boosting": Pipeline([("pre", preprocesador(False)),
                                       ("m", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                            random_state=SEMILLA))]),
    }
    metricas, probs = {}, {}
    for nombre, mod in modelos.items():
        mod.fit(X_tr, y_tr)
        p = mod.predict_proba(X_te)[:, 1]
        probs[nombre] = p
        metricas[nombre] = {"roc_auc": roc_auc_score(y_te, p), "pr_auc": average_precision_score(y_te, p),
                            "recall_fpr1": recall_a_fpr(y_te, p)}

    # Detección de anomalías NO supervisada (no necesita etiquetas de fraude)
    iso = Pipeline([("pre", preprocesador(True)),
                    ("m", IsolationForest(n_estimators=300, contamination="auto", random_state=SEMILLA))])
    iso.fit(X_tr[y_tr == 0])
    p_iso = -iso.decision_function(X_te)
    probs["Isolation Forest (no supervisado)"] = p_iso
    metricas["Isolation Forest (no supervisado)"] = {"roc_auc": roc_auc_score(y_te, p_iso),
                                                    "pr_auc": average_precision_score(y_te, p_iso),
                                                    "recall_fpr1": recall_a_fpr(y_te, p_iso)}

    # Regla estática actual
    r = regla_estatica(df_te)
    metricas["Regla estática actual"] = {"recall": float(recall_score(y_te, r)),
                                         "precision": float(precision_score(y_te, r, zero_division=0)),
                                         "tasa_alertas": float(r.mean())}

    mejor = max((k for k in modelos), key=lambda k: metricas[k]["pr_auc"])
    print("Mejor modelo de fraude:", mejor)
    for k, v in metricas.items():
        print(f"  {k:38s}", {m: round(x, 3) for m, x in v.items()})

    # ------------------------------------------------------------------ SCORING (mora)
    leg_tr, leg_te = df_tr[df_tr.fraude == 0], df_te[df_te.fraude == 0]
    mod_pd = Pipeline([("pre", preprocesador(False)),
                       ("m", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, random_state=SEMILLA))])
    mod_pd.fit(leg_tr[VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS], leg_tr.mora_90d)
    pd_te = mod_pd.predict_proba(leg_te[VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS])[:, 1]
    auc_pd = roc_auc_score(leg_te.mora_90d, pd_te)
    gini = 2 * auc_pd - 1
    print(f"Scoring de mora: AUC={auc_pd:.3f}  Gini={gini:.3f}")

    # Ejemplo de scoring DINÁMICO: el mismo cliente antes y después de 6 meses de buen pago
    cliente = leg_te[(leg_te.max_dias_atraso_12m > 40) & (leg_te.n_entidades_sbs >= 2)].iloc[[0]].copy()
    antes = cliente[VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS]
    despues = antes.copy()
    despues["max_dias_atraso_12m"] = 0
    despues["score_buro"] = min(850, float(despues["score_buro"].iloc[0]) + 35)
    despues["cliente_recurrente"] = 1
    despues["n_entidades_sbs"] -= 1  # canceló una deuda en otra entidad
    despues["antiguedad_negocio_meses"] += 6
    s_antes = prob_a_score(mod_pd.predict_proba(antes)[:, 1])[0]
    s_desp = prob_a_score(mod_pd.predict_proba(despues)[:, 1])[0]
    print(f"Score dinámico del cliente {cliente.id_solicitud.iloc[0]}: {s_antes:.0f} -> {s_desp:.0f}")

    # ------------------------------------------------------------------ DECISIÓN integrada
    mod_f = modelos[mejor]
    pf_all = mod_f.predict_proba(X_te)[:, 1]
    score_all = prob_a_score(mod_pd.predict_proba(X_te)[:, 1])
    decisiones = pd.Series([decidir(a, b) for a, b in zip(pf_all, score_all)], index=X_te.index)
    tabla_dec = pd.crosstab(decisiones, y_te.map({0: "legítima", 1: "fraude"}))
    print(tabla_dec)

    # ------------------------------------------------------------------ IMPORTANCIA de variables
    imp = permutation_importance(mod_f, X_te, y_te, n_repeats=5, random_state=SEMILLA,
                                 scoring="average_precision", n_jobs=-1)
    imp_s = pd.Series(imp.importances_mean, index=X_te.columns).sort_values(ascending=True).tail(10)

    # ------------------------------------------------------------------ FIGURAS
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for nombre, p in probs.items():
        fpr, tpr, _ = roc_curve(y_te, p); pr, rc, _ = precision_recall_curve(y_te, p)
        ax[0].plot(fpr, tpr, label=f"{nombre} (AUC {metricas[nombre]['roc_auc']:.3f})")
        ax[1].plot(rc, pr, label=f"{nombre} (AP {metricas[nombre]['pr_auc']:.3f})")
    rr = metricas["Regla estática actual"]
    ax[0].scatter([r[y_te.values == 0].mean()], [rr["recall"]], c="k", marker="x", s=60, zorder=5, label="Regla estática actual")
    ax[1].scatter([rr["recall"]], [rr["precision"]], c="k", marker="x", s=60, zorder=5, label="Regla estática actual")
    ax[0].plot([0, 1], [0, 1], "--", c="grey", lw=.8)
    ax[0].set(title="Curva ROC – detección de fraude", xlabel="Tasa de falsos positivos", ylabel="Recall (fraude detectado)")
    ax[1].set(title="Curva Precisión-Recall", xlabel="Recall", ylabel="Precisión")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "ml_curvas_fraude.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(imp_s.index, imp_s.values, color="#4a6fa5")
    ax.set(title=f"Importancia por permutación – {mejor}", xlabel="Caída en PR-AUC al permutar")
    fig.tight_layout(); fig.savefig(FIG / "ml_importancia_variables.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sc_leg = prob_a_score(pd_te)
    ax.hist(sc_leg[leg_te.mora_90d.values == 0], bins=40, alpha=.65, label="Pagó", density=True)
    ax.hist(sc_leg[leg_te.mora_90d.values == 1], bins=40, alpha=.65, label="Mora > 90 días", density=True)
    ax.set(title=f"Distribución del score dinámico (Gini = {gini:.2f})", xlabel="Score (300–850)", ylabel="Densidad")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "ml_distribucion_score.png", dpi=160); plt.close(fig)

    joblib.dump({"fraude": mod_f, "pd": mod_pd}, RES / "ml_modelos.joblib")
    salida = {"fraude": metricas, "mejor_modelo_fraude": mejor,
              "scoring": {"auc": auc_pd, "gini": gini, "ejemplo_dinamico": {"antes": s_antes, "despues": s_desp}},
              "decisiones": tabla_dec.to_dict(),
              "tasa_fraude": float(y.mean()), "tasa_mora": float(df.mora_90d.mean()), "n": len(df)}
    (RES / "ml_metricas.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
