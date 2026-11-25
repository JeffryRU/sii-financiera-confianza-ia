"""
Generador de datos sintéticos de solicitudes de microcrédito digital
para el caso "Financiera Confianza Perú".

IMPORTANTE: los datos son FICTICIOS. Se generan con reglas estadísticas
que imitan patrones conocidos de fraude digital (reutilización de
dispositivo, líneas celulares nuevas, baja coincidencia biométrica,
solicitudes nocturnas, ubicación lejana a la agencia) y de morosidad
(alta relación cuota/ingreso, atrasos previos, bajo puntaje de buró).
No contienen datos de personas reales (Ley N.° 29733).

Uso:
    from comun.datos_sinteticos import generar_solicitudes
    df = generar_solicitudes(n=20000, semilla=42)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CANALES = ["app_movil", "web", "asesor_campo"]
RUBROS = ["bodega", "textil", "agro", "transporte", "comida", "servicios"]


def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generar_solicitudes(n: int = 20000, semilla: int = 42) -> pd.DataFrame:
    """Genera `n` solicitudes de crédito con etiquetas `fraude` y `mora_90d`.

    Columnas de entrada (variables que el ERP/CRM ya registra o puede registrar):
        monto_solicitado, plazo_meses, ingreso_mensual_declarado, cuota_estimada,
        ratio_cuota_ingreso, antiguedad_negocio_meses, edad, score_buro,
        n_entidades_sbs, max_dias_atraso_12m, solicitudes_dispositivo_24h,
        antiguedad_linea_dias, distancia_agencia_km, coincidencia_biometrica,
        hora_solicitud, canal, rubro, cambios_datos_7d, cliente_recurrente

    Etiquetas (salidas esperadas):
        fraude   -> 1 si la solicitud resultó fraudulenta (suplantación, mula, etc.)
        mora_90d -> 1 si el crédito cayó en mora > 90 días (solo créditos legítimos)
    """
    rng = np.random.default_rng(semilla)

    canal = rng.choice(CANALES, size=n, p=[0.55, 0.15, 0.30])
    rubro = rng.choice(RUBROS, size=n)
    edad = rng.integers(21, 68, size=n)
    cliente_recurrente = rng.random(n) < 0.45
    antig_negocio = np.clip(rng.gamma(2.2, 18, size=n), 1, 300).round()
    ingreso = np.clip(rng.lognormal(7.4, 0.45, size=n), 600, 15000).round(-1)
    monto = np.clip(rng.lognormal(8.3, 0.6, size=n), 300, 60000).round(-1)
    plazo = rng.choice([6, 9, 12, 18, 24, 36], size=n, p=[.12, .13, .35, .2, .15, .05])
    tasa_mensual = 0.035  # TEM referencial de microcrédito
    cuota = monto * tasa_mensual / (1 - (1 + tasa_mensual) ** (-plazo))
    ratio = cuota / ingreso
    score_buro = np.clip(rng.normal(610, 85, size=n) + 30 * cliente_recurrente, 300, 850).round()
    n_ent = rng.poisson(1.3, size=n)
    atraso = np.where(rng.random(n) < 0.25, rng.gamma(1.5, 12, size=n), 0).round()

    # Señales digitales (solo tienen sentido en canales digitales)
    digital = canal != "asesor_campo"
    disp_24h = np.where(digital, rng.poisson(0.25, size=n) + 1, 1)
    antig_linea = np.clip(rng.gamma(2.0, 400, size=n), 1, 4000).round()
    distancia = np.clip(rng.gamma(1.4, 6, size=n), 0.1, 900).round(1)
    biometria = np.where(digital, np.clip(rng.beta(12, 1.6, size=n), 0, 1), 1.0)
    peso_hora = np.array([1, .6, .4, .3, .3, .5, 1.5, 3, 5, 7, 8, 8, 7, 7, 7, 7, 6, 6, 5, 4, 3, 2.5, 2, 1.5])
    hora = rng.choice(np.arange(24), size=n, p=peso_hora / peso_hora.sum())
    cambios = rng.poisson(0.2, size=n)

    # ---- Fraude: patrón NO lineal (interacciones) que las reglas estáticas no capturan
    nocturna = (hora <= 4) | (hora >= 23)
    z_f = (-4.7
           + 1.1 * np.log1p(disp_24h - 1) * 2.2
           + 1.6 * (antig_linea < 30)
           + 2.4 * (biometria < 0.80)
           + 0.9 * nocturna
           + 1.2 * (distancia > 80)
           + 0.8 * (cambios >= 2)
           + 1.4 * ((antig_linea < 30) & (biometria < 0.85))       # interacción
           + 1.0 * ((monto > 15000) & (~cliente_recurrente))
           - 1.5 * cliente_recurrente
           - 2.5 * (~digital))
    fraude = (rng.random(n) < _sigmoide(z_f)).astype(int)

    # Huella digital típica de fraude organizado (aparece en ~60 % de los casos, no en todos)
    f = fraude == 1
    m1 = f & digital & (rng.random(n) < 0.6)
    disp_24h = np.where(m1, disp_24h + rng.poisson(2.0, size=n), disp_24h)
    m2 = f & (rng.random(n) < 0.55)
    antig_linea = np.where(m2, rng.integers(1, 25, size=n), antig_linea)
    m3 = f & digital & (rng.random(n) < 0.5)
    biometria = np.where(m3, np.clip(rng.beta(6, 2.5, size=n), 0, 1), biometria)
    m4 = f & (rng.random(n) < 0.3)
    hora = np.where(m4, rng.choice([0, 1, 2, 3, 23], size=n), hora)
    cambios = np.where(f & (rng.random(n) < 0.4), cambios + rng.integers(2, 5, size=n), cambios)

    # Los defraudadores "se ven bien" en variables declaradas (por eso fallan las reglas)
    ingreso = np.where(fraude == 1, ingreso * rng.uniform(1.0, 1.6, size=n), ingreso).round(-1)
    ratio = cuota / ingreso

    # ---- Mora > 90 días (riesgo de crédito) solo para legítimos
    z_m = (-3.1 + 3.2 * (ratio - 0.25) - 0.006 * (score_buro - 600)
           + 0.035 * atraso + 0.25 * n_ent - 0.006 * antig_negocio
           + 0.35 * (rubro == "agro") - 0.4 * cliente_recurrente)
    mora = np.where(fraude == 1, 0, (rng.random(n) < _sigmoide(z_m)).astype(int))

    return pd.DataFrame({
        "id_solicitud": [f"SOL-{i:06d}" for i in range(1, n + 1)],
        "canal": canal, "rubro": rubro, "edad": edad,
        "cliente_recurrente": cliente_recurrente.astype(int),
        "antiguedad_negocio_meses": antig_negocio,
        "ingreso_mensual_declarado": ingreso,
        "monto_solicitado": monto, "plazo_meses": plazo,
        "cuota_estimada": cuota.round(2), "ratio_cuota_ingreso": ratio.round(4),
        "score_buro": score_buro, "n_entidades_sbs": n_ent,
        "max_dias_atraso_12m": atraso,
        "solicitudes_dispositivo_24h": disp_24h,
        "antiguedad_linea_dias": antig_linea,
        "distancia_agencia_km": distancia,
        "coincidencia_biometrica": biometria.round(3),
        "hora_solicitud": hora, "cambios_datos_7d": cambios,
        "fraude": fraude, "mora_90d": mora,
    })


VARIABLES_NUMERICAS = [
    "edad", "cliente_recurrente", "antiguedad_negocio_meses", "ingreso_mensual_declarado",
    "monto_solicitado", "plazo_meses", "cuota_estimada", "ratio_cuota_ingreso",
    "score_buro", "n_entidades_sbs", "max_dias_atraso_12m", "solicitudes_dispositivo_24h",
    "antiguedad_linea_dias", "distancia_agencia_km", "coincidencia_biometrica",
    "hora_solicitud", "cambios_datos_7d",
]
VARIABLES_CATEGORICAS = ["canal", "rubro"]


if __name__ == "__main__":
    df = generar_solicitudes()
    print(df.head())
    print("Tasa de fraude:", df.fraude.mean().round(4), "| Tasa de mora:", df.mora_90d.mean().round(4))
