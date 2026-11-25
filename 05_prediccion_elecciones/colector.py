"""
Colectores de publicaciones de redes sociales.

- ColectorX: modo EN VIVO. Usa la API v2 de X (búsqueda reciente) con un token Bearer
  (variable de entorno X_BEARER_TOKEN). Consulta cada N minutos los tuits en español que
  mencionan a los candidatos y devuelve solo texto + fecha (no guarda usuario: minimización
  de datos, Ley N.° 29733).
- ColectorSimulado: modo DEMO. Genera un flujo SIMULADO de publicaciones a partir de plantillas,
  con un apoyo latente que cambia en el tiempo y un sesgo de plataforma (redes sociales
  sobrerrepresentan a usuarios urbanos/limeños). Sirve para probar el pipeline de punta a punta
  sin credenciales. Sus textos NO son publicaciones reales.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator

import numpy as np

CONSULTA_X = ('(keiko OR fujimori OR "fuerza popular" OR "roberto sanchez" OR "roberto sánchez" '
              'OR "juntos por el peru" OR #SegundaVuelta2026) lang:es -is:retweet')


class ColectorX:
    URL = "https://api.x.com/2/tweets/search/recent"

    def __init__(self, token: str | None = None, consulta: str = CONSULTA_X):
        self.token = token or os.environ.get("X_BEARER_TOKEN")
        if not self.token:
            raise RuntimeError("Define X_BEARER_TOKEN para usar el modo en vivo.")
        self.consulta = consulta
        self.desde_id = None

    def lote(self, max_resultados: int = 100) -> list[dict]:
        import requests
        params = {"query": self.consulta, "max_results": max_resultados, "tweet.fields": "created_at,lang"}
        if self.desde_id:
            params["since_id"] = self.desde_id
        r = requests.get(self.URL, params=params, headers={"Authorization": f"Bearer {self.token}"}, timeout=30)
        r.raise_for_status()
        datos = r.json().get("data", [])
        if datos:
            self.desde_id = datos[0]["id"]
        return [{"fecha": datetime.fromisoformat(d["created_at"].replace("Z", "+00:00")), "texto": d["text"]} for d in datos]

    def flujo(self, cada_seg: int = 300) -> Iterator[list[dict]]:
        while True:
            yield self.lote()
            time.sleep(cada_seg)


# ------------------------------------------------------------------------------- simulado
PLANT_POS = ["{c} es la mejor opción para el Perú 💪", "Mi voto es por {c}, tiene experiencia y propuestas",
             "Vamos {c}! Firme hasta el final", "Buen debate de {c}, me convenció", "Confío en {c}, trabajo y orden",
             "{c} presidente/a, esperanza para las regiones", "Excelente cierre de campaña de {c} 🙌",
             "Con {c} hay cambio de verdad ✅"]
PLANT_NEG = ["{c} es un peligro para el país", "Jamás votaré por {c} 👎", "{c} miente, pésimo debate",
             "Qué vergüenza las propuestas de {c}", "{c} = crisis y fracaso", "No confío en {c}, basta ya",
             "Antivoto total contra {c} ❌", "{c} improvisado, peor opción"]
PLANT_NEU = ["Hoy debate entre {c} y {d} a las 8 pm", "Mañana se publica la última encuesta sobre {c} y {d}",
             "ONPE recuerda el horario de votación del 7 de junio", "¿Dónde voto? Consulta tu local en ONPE"]
ALIAS = {"fujimori": ["Keiko", "Keiko Fujimori", "#KeikoPresidenta", "Fuerza Popular", "@KeikoFujimori"],
         "sanchez": ["Roberto Sánchez", "Sánchez", "#RobertoSanchez", "Juntos por el Perú", "el profe Sánchez"]}


class ColectorSimulado:
    """Flujo horario de publicaciones simuladas entre `inicio` y `fin`.

    apoyo_latente(t): fracción de apoyo a Fujimori en el electorado (0-1) en el instante t.
    sesgo_plataforma: puntos que la conversación en redes sobrerrepresenta a Fujimori.
    """

    def __init__(self, inicio: datetime, fin: datetime, apoyo_latente, sesgo_plataforma: float = 0.03,
                 posts_por_hora: int = 220, semilla: int = 2026):
        self.inicio, self.fin, self.f = inicio, fin, apoyo_latente
        self.sesgo, self.pph = sesgo_plataforma, posts_por_hora
        self.rng = np.random.default_rng(semilla)

    def _post(self, t: datetime, p_k: float) -> dict:
        r = self.rng
        if r.random() < 0.15:
            txt = r.choice(PLANT_NEU).format(c=r.choice(ALIAS["fujimori"]), d=r.choice(ALIAS["sanchez"]))
            return {"fecha": t, "texto": txt}
        partidario_k = r.random() < p_k
        positivo = r.random() < 0.55          # apoyar a su candidato vs. atacar al rival
        if partidario_k:
            c, plant = ("fujimori", PLANT_POS) if positivo else ("sanchez", PLANT_NEG)
        else:
            c, plant = ("sanchez", PLANT_POS) if positivo else ("fujimori", PLANT_NEG)
        return {"fecha": t, "texto": r.choice(plant).format(c=r.choice(ALIAS[c]))}

    def flujo(self) -> Iterator[list[dict]]:
        t = self.inicio
        while t < self.fin:
            p_k = float(np.clip(self.f(t) + self.sesgo + self.rng.normal(0, 0.01), 0.01, 0.99))
            n = int(self.rng.poisson(self.pph * (1.6 if 19 <= t.hour <= 23 else 0.4 if t.hour < 7 else 1.0)))
            yield [self._post(t + timedelta(minutes=float(self.rng.uniform(0, 60))), p_k) for _ in range(n)]
            t += timedelta(hours=1)
