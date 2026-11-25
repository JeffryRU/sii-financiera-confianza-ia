"""
Componente 1 – Minería de procesos: Descubrimiento, Conformidad y Mejora
Caso: Financiera Confianza Perú – proceso de evaluación y aprobación de microcréditos

Problema: el proceso estándar debería durar 24 h, pero los event logs de auditoría
muestran que ~35 % de los expedientes siguen rutas no autorizadas o retrocesos
manuales entre analistas.

Este script:
 1. Genera un event log SINTÉTICO (formato XES/CSV: case_id, actividad, timestamp,
    recurso) con 2 000 expedientes que reproduce esa situación.
 2. DESCUBRIMIENTO: obtiene el grafo de flujo directo (DFG) y un modelo con Inductive Miner.
 3. CONFORMIDAD: compara cada expediente contra el MODELO NORMATIVO (el proceso que
    manda el manual de créditos) mediante token-based replay y clasifica las desviaciones.
 4. MEJORA: mide el impacto en tiempo de ciclo, cuellos de botella y analistas con más
    retrocesos, y exporta indicadores para el tablero.

Ejecución:
    python 01_mineria_procesos/mineria_procesos.py
Salidas:
    resultados/event_log_creditos.csv, resultados/pm_metricas.json, figures/pm_*.png
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pm4py

RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / "figures"; RES = RAIZ / "resultados"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
SEMILLA = 7

# Proceso normativo (manual de créditos). Duración media de cada paso en horas.
NORMATIVO = [
    ("Registrar solicitud", 0.3),
    ("Validar identidad RENIEC", 0.4),
    ("Consultar central de riesgos SBS", 0.5),
    ("Evaluar expediente", 4.0),
    ("Verificar negocio en campo", 8.0),
    ("Decidir en comité de créditos", 6.0),
    ("Aprobar crédito", 1.0),
    ("Desembolsar", 2.0),
]
ANALISTAS = [f"analista_{i:02d}" for i in range(1, 13)]
ASESORES = [f"asesor_{i:02d}" for i in range(1, 31)]


def _recurso(act: str, rng, analista: str, asesor: str) -> str:
    if act in ("Evaluar expediente", "Devolver a asesor", "Reasignar analista"):
        return analista
    if act in ("Registrar solicitud", "Verificar negocio en campo", "Completar documentos"):
        return asesor
    if act == "Decidir en comité de créditos":
        return "comite"
    return "sistema"


def generar_event_log(n_casos: int = 2000, tasa_desvio: float = 0.35) -> pd.DataFrame:
    """Simula el event log del core financiero. Tipos de desviación inyectados:
       - RETROCESO: el analista devuelve el expediente al asesor una o más veces.
       - REASIGNACION: el expediente pasa de un analista a otro (ping-pong).
       - OMITE_COMITE: se aprueba sin pasar por comité (ruta no autorizada).
       - OMITE_CAMPO: se evalúa sin verificación de campo.
       - DESEMBOLSO_ANTICIPADO: se desembolsa antes de la aprobación formal.
    """
    rng = np.random.default_rng(SEMILLA)
    filas = []
    inicio = datetime(2026, 9, 1, 8, 0)
    pesos = {"RETROCESO": .34, "REASIGNACION": .24, "OMITE_COMITE": .18, "OMITE_CAMPO": .14, "DESEMBOLSO_ANTICIPADO": .10}
    for i in range(n_casos):
        caso = f"EXP-{i + 1:05d}"
        t = inicio + timedelta(hours=float(rng.uniform(0, 24 * 60)))
        analista = rng.choice(ANALISTAS, p=_pesos_analistas())
        asesor = rng.choice(ASESORES)
        traza = [a for a, _ in NORMATIVO]
        desvio = "NINGUNO"
        if rng.random() < tasa_desvio:
            desvio = rng.choice(list(pesos), p=list(pesos.values()))
            k = traza.index("Evaluar expediente")
            if desvio == "RETROCESO":
                for _ in range(rng.integers(1, 4)):
                    traza[k + 1:k + 1] = ["Devolver a asesor", "Completar documentos", "Evaluar expediente"]
            elif desvio == "REASIGNACION":
                traza[k + 1:k + 1] = ["Reasignar analista", "Evaluar expediente"] * int(rng.integers(1, 3))
            elif desvio == "OMITE_COMITE":
                traza.remove("Decidir en comité de créditos")
            elif desvio == "OMITE_CAMPO":
                traza.remove("Verificar negocio en campo")
            elif desvio == "DESEMBOLSO_ANTICIPADO":
                traza.remove("Desembolsar"); traza.insert(traza.index("Aprobar crédito"), "Desembolsar")
        dur = dict(NORMATIVO) | {"Devolver a asesor": 6.0, "Completar documentos": 14.0, "Reasignar analista": 10.0}
        for act in traza:
            filas.append({"case_id": caso, "actividad": act, "timestamp": t,
                          "recurso": _recurso(act, rng, analista, asesor), "desvio_real": desvio})
            t += timedelta(hours=float(rng.gamma(2.0, dur[act] / 2.0)))
    return pd.DataFrame(filas)


def _pesos_analistas():
    w = np.ones(len(ANALISTAS)); w[[2, 7]] = 2.2  # dos analistas con más carga
    return w / w.sum()


def modelo_normativo():
    """Petri net del proceso que exige el manual (se descubre de una traza ideal)."""
    ideal = pd.DataFrame([{"case:concept:name": "IDEAL", "concept:name": a,
                           "time:timestamp": datetime(2026, 1, 1) + timedelta(hours=j)}
                          for j, (a, _) in enumerate(NORMATIVO)])
    return pm4py.discover_petri_net_inductive(ideal)


def main() -> None:
    df = generar_event_log()
    df.to_csv(RES / "event_log_creditos.csv", index=False)
    log = pm4py.format_dataframe(df, case_id="case_id", activity_key="actividad", timestamp_key="timestamp")
    log["org:resource"] = df["recurso"].values

    # ---------------- 1. DESCUBRIMIENTO
    dfg, ini, fin = pm4py.discover_dfg(log)
    pm4py.save_vis_dfg(dfg, ini, fin, str(FIG / "pm_dfg_descubierto.png"))
    red_desc, im_d, fm_d = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)
    pm4py.save_vis_petri_net(red_desc, im_d, fm_d, str(FIG / "pm_modelo_descubierto.png"))
    variantes = pm4py.get_variants(log)
    print(f"Expedientes: {log['case:concept:name'].nunique()} | Eventos: {len(log)} | Variantes: {len(variantes)}")

    # ---------------- 2. CONFORMIDAD contra el modelo normativo
    red, im, fm = modelo_normativo()
    pm4py.save_vis_petri_net(red, im, fm, str(FIG / "pm_modelo_normativo.png"))
    replay = pm4py.conformance_diagnostics_token_based_replay(log, red, im, fm)
    casos = log.groupby("case:concept:name", sort=False).agg(
        inicio=("time:timestamp", "min"), fin=("time:timestamp", "max"),
        desvio_real=("desvio_real", "first"),
        traza=("concept:name", lambda s: " > ".join(s))).reset_index()
    # pm4py devuelve el replay en el mismo orden que los casos del log
    orden = log.drop_duplicates("case:concept:name")["case:concept:name"].tolist()
    ajuste = pd.DataFrame({"case:concept:name": orden,
                           "conforme": [r["trace_is_fit"] for r in replay],
                           "fitness": [r["trace_fitness"] for r in replay]})
    casos = casos.merge(ajuste, on="case:concept:name")
    casos["horas"] = (casos.fin - casos.inicio).dt.total_seconds() / 3600
    fitness_global = pm4py.fitness_token_based_replay(log, red, im, fm)
    pct_no_conf = 100 * (1 - casos.conforme.mean())
    print(f"No conformes: {pct_no_conf:.1f} % | fitness del log: {fitness_global['log_fitness']:.3f}")

    # Clasificación de la desviación a partir de la traza observada (no del dato "real")
    def clasificar(t: str) -> str:
        if "Devolver a asesor" in t: return "Retroceso a asesor"
        if "Reasignar analista" in t: return "Reasignación entre analistas"
        if "Decidir en comité" not in t: return "Aprobación sin comité"
        if "Verificar negocio" not in t: return "Sin verificación de campo"
        if t.index("Desembolsar") < t.index("Aprobar crédito"): return "Desembolso antes de aprobar"
        return "Conforme"
    casos["tipo_desvio"] = casos.traza.apply(clasificar)
    exactitud = (casos.conforme == (casos.desvio_real == "NINGUNO")).mean()

    # ---------------- 3. MEJORA: impacto en tiempos y recursos
    resumen = casos.groupby("tipo_desvio").agg(expedientes=("traza", "size"), horas_mediana=("horas", "median"),
                                               horas_p90=("horas", lambda s: s.quantile(.9))).sort_values("expedientes", ascending=False)
    resumen["pct"] = 100 * resumen.expedientes / len(casos)
    sla24 = {"conformes_dentro_24h_pct": float(100 * (casos[casos.conforme].horas <= 24).mean()),
             "no_conformes_dentro_24h_pct": float(100 * (casos[~casos.conforme].horas <= 24).mean())}
    retro = df[df.actividad.isin(["Devolver a asesor", "Reasignar analista"])].groupby("recurso").size()
    carga = df[df.actividad == "Evaluar expediente"].groupby("recurso").case_id.nunique()
    tasa_retro = (retro / carga).fillna(0).sort_values(ascending=False)
    # tiempo de espera medio ANTES de cada actividad (cuellos de botella)
    d = df.sort_values(["case_id", "timestamp"]).copy()
    d["espera_h"] = d.groupby("case_id").timestamp.diff().dt.total_seconds() / 3600
    espera = d.groupby("actividad").espera_h.mean().dropna().sort_values(ascending=False)
    print(resumen.round(1)); print(sla24)

    # ---------------- Figuras
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    r2 = resumen.drop("Conforme")
    ax[0].barh(r2.index, r2.pct, color="#b5532c"); ax[0].invert_yaxis()
    ax[0].set(title=f"Expedientes no conformes: {pct_no_conf:.1f} %", xlabel="% del total de expedientes", xlim=(0, r2.pct.max() * 1.18))
    for i, v in enumerate(r2.pct): ax[0].text(v + .2, i, f"{v:.1f} %", va="center", fontsize=8)
    orden_box = list(resumen.index)
    ax[1].boxplot([casos[casos.tipo_desvio == k].horas for k in orden_box], vert=False, showfliers=False)
    ax[1].set_yticks(range(1, len(orden_box) + 1), orden_box, fontsize=8)
    ax[1].axvline(24, c="k", ls="--", lw=1); ax[1].text(25, .6, "Meta 24 h", fontsize=8)
    ax[1].set(title="Tiempo de ciclo por tipo de ruta", xlabel="Horas desde registro hasta desembolso")
    fig.tight_layout(); fig.savefig(FIG / "pm_conformidad.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))
    e = espera.head(8)[::-1]; ax[0].barh(e.index, e.values, color="#4a6fa5")
    ax[0].set(title="Espera media antes de cada actividad", xlabel="Horas")
    t = tasa_retro.head(8)[::-1]; ax[1].barh(t.index, t.values, color="#7a5c99")
    ax[1].set(title="Retrocesos/reasignaciones por expediente evaluado", xlabel="Eventos por expediente")
    fig.tight_layout(); fig.savefig(FIG / "pm_cuellos_botella.png", dpi=160); plt.close(fig)

    salida = {"expedientes": int(len(casos)), "eventos": int(len(df)), "variantes": len(variantes),
              "pct_no_conformes": pct_no_conf, "log_fitness": fitness_global["log_fitness"],
              "exactitud_deteccion": float(exactitud), "resumen": resumen.round(2).reset_index().to_dict("records"),
              "sla24": sla24, "espera_media_h": espera.round(2).to_dict(),
              "top_analistas_retroceso": tasa_retro.head(5).round(3).to_dict()}
    (RES / "pm_metricas.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    casos.drop(columns=["inicio", "fin"]).to_csv(RES / "pm_conformidad_por_expediente.csv", index=False)


if __name__ == "__main__":
    main()
