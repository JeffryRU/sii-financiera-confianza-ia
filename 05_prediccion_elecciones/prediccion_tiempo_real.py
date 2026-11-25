"""
Predicción en tiempo real ("nowcast") de la segunda vuelta presidencial Perú 2026
combinando encuestas + análisis de sentimiento en redes sociales.

Por qué este diseño (y no solo contar tuits):
  * Las redes sociales NO son una muestra representativa del electorado peruano
    (sobrerrepresentan zonas urbanas, Lima y jóvenes). Contar menciones o "likes" falla.
  * Por eso el modelo usa las ENCUESTAS como ancla (prior) y usa las redes solo para medir
    el CAMBIO de la conversación desde la última encuesta (señal de tendencia), lo que
    cancela el sesgo constante de la plataforma.

Modelo:
  1. Sentimiento/postura por publicación (sentimiento.py: RoBERTuito o léxico de respaldo).
  2. Índice social s_t (ventana móvil de 24 h) = (pos_Fujimori + neg_Sánchez) / (menciones polarizadas).
  3. Prior de encuestas p_enc(t): promedio ponderado por recencia (vida media de 2 días) en escala logit.
  4. Nowcast: logit p_t = logit p_enc(t) + β · [logit s_t − logit s_ref], con s_ref = índice social
     alrededor de la fecha de la última encuesta y β = 0.6 (encogimiento por ruido de la señal).
  5. Incertidumbre: Monte Carlo (10 000 simulaciones) con error de encuesta σ = 1.5 pts y el error
     muestral del índice social -> probabilidad de victoria e intervalo del 80 %.

Uso:
  python prediccion_tiempo_real.py            # DEMO con flujo simulado del 1 al 7 de junio de 2026
  python prediccion_tiempo_real.py --vivo     # EN VIVO con la API de X (requiere X_BEARER_TOKEN)
Salidas: ../figures/elecciones_nowcast.png, ../resultados/elecciones_nowcast.csv y .json
"""
from __future__ import annotations

import json
import sys
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from colector import ColectorSimulado, ColectorX  # noqa: E402
from sentimiento import AnalizadorSentimiento  # noqa: E402

RAIZ = AQUI.parent
FIG = RAIZ / "figures"; RES = RAIZ / "resultados"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)

BETA, VIDA_MEDIA_DIAS, SIGMA_ENCUESTA, VENTANA_H, UMBRAL_POL = 0.6, 2.0, 0.015, 24, 0.2
logit = lambda p: np.log(p / (1 - p))
inv = lambda z: 1 / (1 + np.exp(-z))


def cargar_encuestas() -> pd.DataFrame:
    e = pd.read_csv(AQUI / "data" / "encuestas_2v_2026.csv", parse_dates=["fecha"])
    e["p_k"] = e.fujimori_validos / (e.fujimori_validos + e.sanchez_validos)
    e["fecha"] = e.fecha + pd.Timedelta(hours=20)   # hora aproximada de publicación/cierre de campo
    return e.sort_values("fecha")


def prior_encuestas(enc: pd.DataFrame, t: datetime) -> tuple[float, datetime | None]:
    v = enc[enc.fecha <= t]
    if v.empty:
        return np.nan, None
    edad = (t - v.fecha).dt.total_seconds() / 86400
    w = 0.5 ** (edad / VIDA_MEDIA_DIAS)
    return float(inv(np.average(logit(v.p_k), weights=w))), v.fecha.iloc[-1].to_pydatetime()


class Nowcaster:
    def __init__(self, encuestas: pd.DataFrame, analizador: AnalizadorSentimiento):
        self.enc, self.an = encuestas, analizador
        self.ventana: deque = deque()          # (fecha, voto_implicito_k: 1/0)
        self.historial_social: list[tuple[datetime, float, int]] = []
        self.filas = []

    def ingerir(self, publicaciones: list[dict]) -> None:
        for p in publicaciones:
            for cand, pol in self.an.analizar(p["texto"]).items():
                if abs(pol) < UMBRAL_POL:
                    continue
                # positivo hacia K o negativo hacia S => señal pro-K
                pro_k = (cand == "fujimori") == (pol > 0)
                self.ventana.append((p["fecha"], int(pro_k)))

    def actualizar(self, t: datetime, rng=np.random.default_rng(0)) -> dict:
        while self.ventana and self.ventana[0][0] < t - timedelta(hours=VENTANA_H):
            self.ventana.popleft()
        n = len(self.ventana)
        s = (sum(v for _, v in self.ventana) + 1) / (n + 2)            # suavizado de Laplace
        self.historial_social.append((t, s, n))
        p_enc, f_ult = prior_encuestas(self.enc, t)
        # referencia social: promedio del índice ±12 h alrededor de la última encuesta
        ref = [x for (ti, x, _) in self.historial_social if f_ult and abs((ti - f_ult).total_seconds()) <= 12 * 3600]
        s_ref = float(np.mean(ref)) if ref else s
        z = logit(p_enc) + BETA * (logit(s) - logit(s_ref))
        # Monte Carlo
        sd_social = BETA * np.sqrt(1 / (n * s * (1 - s))) if n else 0.2
        sd_enc = SIGMA_ENCUESTA / (p_enc * (1 - p_enc))
        sims = inv(z + rng.normal(0, np.hypot(sd_enc, sd_social), 10000))
        fila = {"hora": t, "n_ventana": n, "indice_social": s, "indice_ref": s_ref, "prior_encuestas": p_enc,
                "nowcast_fujimori": float(inv(z)), "p10": float(np.quantile(sims, .1)),
                "p90": float(np.quantile(sims, .9)), "prob_victoria_fujimori": float((sims > .5).mean())}
        self.filas.append(fila)
        return fila


def demo() -> None:
    enc = cargar_encuestas()
    # Apoyo latente SIMULADO: se interpola entre los simulacros (no usa el resultado oficial)
    xs = np.array([d.timestamp() for d in enc.fecha]); ys = enc.p_k.values

    def apoyo(t: datetime) -> float:
        return float(np.interp(t.timestamp(), xs, ys))

    inicio, fin = datetime(2026, 5, 31, 0), datetime(2026, 6, 7, 16)   # 16:00 = cierre de mesas
    col = ColectorSimulado(inicio, fin, apoyo, sesgo_plataforma=0.03)
    an = AnalizadorSentimiento()
    nc = Nowcaster(enc, an)
    hora = inicio
    for lote in col.flujo():
        nc.ingerir(lote)
        hora += timedelta(hours=1)
        if hora >= datetime(2026, 5, 31, 21):
            f = nc.actualizar(hora)
            if hora.hour in (0, 12) or hora == fin:
                print(f"{hora:%d/%m %H:%M}  n={f['n_ventana']:4d}  social={f['indice_social']:.3f}  "
                      f"encuestas={f['prior_encuestas']:.3f}  NOWCAST={f['nowcast_fujimori']:.3%} "
                      f"[{f['p10']:.1%}–{f['p90']:.1%}]  P(gana KF)={f['prob_victoria_fujimori']:.0%}")
    guardar(nc, enc, an.backend)


def guardar(nc: Nowcaster, enc: pd.DataFrame, backend: str) -> None:
    df = pd.DataFrame(nc.filas)
    df.to_csv(RES / "elecciones_nowcast.csv", index=False)
    real = pd.read_csv(AQUI / "data" / "resultado_onpe_2v_2026.csv")
    real_k = float(real.loc[0, "votos_validos_pct"]) / 100
    ult = df.iloc[-1]
    ult_enc = enc.iloc[-1]
    salida = {"modo": "demo_simulado", "backend_sentimiento": backend,
              "nowcast_final_fujimori_pct": 100 * ult.nowcast_fujimori,
              "intervalo80": [100 * ult.p10, 100 * ult.p90], "prob_victoria_fujimori": ult.prob_victoria_fujimori,
              "resultado_onpe_fujimori_pct": 100 * real_k,
              "error_absoluto_pts": 100 * abs(ult.nowcast_fujimori - real_k),
              "error_ultima_encuesta_pts": abs(ult_enc.fujimori_validos - 100 * real_k),
              "error_solo_redes_pts": 100 * abs(ult.indice_social - real_k),
              "ganador_predicho": "Keiko Fujimori" if ult.nowcast_fujimori > .5 else "Roberto Sánchez",
              "ganador_real": "Keiko Fujimori"}
    (RES / "elecciones_nowcast.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(salida, indent=2, ensure_ascii=False, default=float))

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.fill_between(df.hora, 100 * df.p10, 100 * df.p90, alpha=.18, color="#e07b00", label="Intervalo 80 %")
    ax.plot(df.hora, 100 * df.nowcast_fujimori, color="#e07b00", lw=2, label="Nowcast (encuestas + redes)")
    ax.plot(df.hora, 100 * df.indice_social, color="grey", lw=1, alpha=.7, label="Índice social bruto (solo redes)")
    ax.plot(df.hora, 100 * df.prior_encuestas, color="#3b6ea5", lw=1, ls="--", label="Promedio de encuestas")
    ax.scatter(enc.fecha, 100 * enc.p_k, color="#3b6ea5", zorder=5, label="Simulacros (votos válidos)")
    ax.axhline(100 * real_k, color="k", lw=1, ls=":", label=f"Resultado ONPE/JNE: {100 * real_k:.3f} %")
    ax.axhline(50, color="#b22", lw=.7)
    ax.set(title="Keiko Fujimori – % de votos válidos estimado en tiempo real (segunda vuelta 2026)",
           ylabel="% votos válidos Fujimori", xlim=(df.hora.min() - timedelta(hours=30), df.hora.max()))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.legend(fontsize=7.5, loc="lower left", ncol=3, bbox_to_anchor=(0, -0.32)); ax.set_ylim(46, 59)
    ax.text(0.01, 0.97, "Flujo de redes SIMULADO (modo demo). En vivo: --vivo con X_BEARER_TOKEN",
            transform=ax.transAxes, fontsize=7.5, color="#555", va="top")
    fig.tight_layout(); fig.savefig(FIG / "elecciones_nowcast.png", dpi=160); plt.close(fig)


def vivo() -> None:
    enc = cargar_encuestas()
    nc = Nowcaster(enc, AnalizadorSentimiento("auto"))
    for lote in ColectorX().flujo(cada_seg=300):
        nc.ingerir(lote)
        f = nc.actualizar(datetime.now(tz=lote[0]["fecha"].tzinfo if lote else None).replace(tzinfo=None))
        print(f"{f['hora']:%H:%M} NOWCAST Fujimori={f['nowcast_fujimori']:.2%} P(gana)={f['prob_victoria_fujimori']:.0%}")


if __name__ == "__main__":
    vivo() if "--vivo" in sys.argv else demo()
