"""
Análisis de sentimiento y postura (stance) hacia cada candidato en publicaciones de redes sociales.

Algoritmo recomendado (backend "transformer"):
    pysentimiento – modelo RoBERTuito, un transformer tipo RoBERTa preentrenado con ~500 millones
    de tuits en español y ajustado para sentimiento (POS/NEU/NEG). Es el algoritmo adecuado porque
    entiende jerga, abreviaturas, emojis e ironía básica de redes, cosa que los léxicos no logran.
    Requiere:  pip install pysentimiento  (descarga el modelo de Hugging Face la primera vez).

Respaldo (backend "lexico"):
    Léxico político en español con manejo de negación e intensificadores. Se usa cuando no hay
    acceso a Hugging Face (por ejemplo, en el entorno donde se generaron los resultados de demo).

En ambos casos, la publicación se divide en segmentos ("pero", "y", ",", ";") y el sentimiento de
cada segmento se atribuye al candidato que menciona, para manejar textos que comparan a ambos.
"""
from __future__ import annotations

import re
import unicodedata

CANDIDATOS = {
    "fujimori": ["keiko", "fujimori", "keikofujimori", "fuerza popular", "fp", "la señora k", "keikista"],
    "sanchez": ["roberto sanchez", "sanchez", "robertosanchez", "juntos por el peru", "jp", "el profe sanchez"],
}

POSITIVAS = {
    "apoyo", "voto", "votare", "votaremos", "gana", "ganara", "ganador", "presidenta", "presidente", "mejor",
    "confio", "confianza", "honesto", "honesta", "capaz", "experiencia", "orden", "seguridad", "esperanza",
    "cambio", "trabajo", "propuesta", "propuestas", "buen", "buena", "bien", "excelente", "fuerza", "firme",
    "vamos", "si", "lidera", "convence", "convencio", "debate", "gano", "aplausos", "respaldo", "unidad",
}
NEGATIVAS = {
    "corrupto", "corrupta", "corrupcion", "mentira", "mentiroso", "mentirosa", "miedo", "peligro", "nunca",
    "fraude", "robo", "ladron", "ladrona", "comunista", "dictadura", "terror", "crisis", "pesimo", "pesima",
    "mal", "mala", "peor", "verguenza", "rechazo", "antivoto", "no", "jamas", "cae", "pierde", "perdio",
    "improvisado", "incapaz", "investigada", "investigado", "carcel", "odio", "basta", "fracaso", "desastre",
}
NEGADORES = {"no", "nunca", "jamas", "ni"}
INTENSIFICADORES = {"muy": 1.5, "super": 1.5, "totalmente": 1.5, "demasiado": 1.4, "mucho": 1.3}
EMOJIS_POS = "👍💪🙌❤🧡💚✅🔥😀😊"
EMOJIS_NEG = "👎🤮😡💩❌🤡😠"


def normalizar(t: str) -> str:
    t = t.lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return re.sub(r"https?://\S+|@", " ", t).replace("#", " ")


def candidatos_en(segmento: str) -> list[str]:
    s = f" {segmento} "
    return [c for c, alias in CANDIDATOS.items() if any(re.search(rf"\b{re.escape(a)}\b", s) for a in alias)]


class AnalizadorSentimiento:
    def __init__(self, backend: str = "auto"):
        self.backend = "lexico"
        if backend in ("auto", "transformer"):
            try:
                from pysentimiento import create_analyzer  # type: ignore
                self._tr = create_analyzer(task="sentiment", lang="es")
                self.backend = "transformer"
            except Exception as e:  # sin internet / sin paquete -> léxico
                if backend == "transformer":
                    raise RuntimeError("No se pudo cargar pysentimiento/RoBERTuito") from e

    # --- polaridad de un segmento en [-1, 1]
    def polaridad(self, texto_original: str, texto_norm: str) -> float:
        if self.backend == "transformer":
            p = self._tr.predict(texto_original).probas
            return float(p["POS"] - p["NEG"])
        toks = re.findall(r"[a-zñ]+", texto_norm)
        punt, neg_ventana, mult = 0.0, 0, 1.0
        for tk in toks:
            if tk in NEGADORES and tk != "no":
                neg_ventana = 3; continue
            if tk == "no":
                neg_ventana = 3
            if tk in INTENSIFICADORES:
                mult = INTENSIFICADORES[tk]; continue
            v = 1.0 if tk in POSITIVAS else -1.0 if tk in NEGATIVAS else 0.0
            if v and neg_ventana and tk != "no":
                v = -v
            punt += v * mult; mult = 1.0
            neg_ventana = max(0, neg_ventana - 1)
        punt += sum(texto_original.count(e) for e in EMOJIS_POS) - sum(texto_original.count(e) for e in EMOJIS_NEG)
        return max(-1.0, min(1.0, punt / 2.0))

    def analizar(self, texto: str) -> dict[str, float]:
        """Devuelve {candidato: polaridad} para los candidatos mencionados en la publicación."""
        norm = normalizar(texto)
        segs_n = re.split(r"\bpero\b|\bmientras\b|[;,.!?]|\by\b", norm)
        segs_o = re.split(r"\bpero\b|\bmientras\b|[;,.!?]|\by\b", texto, flags=re.I)
        res: dict[str, list[float]] = {}
        ultimo = candidatos_en(norm)[:1]
        for sn, so in zip(segs_n, segs_o + [""] * (len(segs_n) - len(segs_o))):
            if not sn.strip():
                continue
            cands = candidatos_en(sn) or ultimo
            if not cands:
                continue
            ultimo = cands
            p = self.polaridad(so or sn, sn)
            for c in cands:
                res.setdefault(c, []).append(p)
        return {c: sum(v) / len(v) for c, v in res.items()}


if __name__ == "__main__":
    a = AnalizadorSentimiento()
    print("backend:", a.backend)
    for t in ["Keiko es la mejor opción, votaré por ella 💪", "Sánchez no tiene propuestas, pésimo debate",
              "No confío en Keiko pero Sánchez me da miedo", "#RobertoSanchez gana el debate, buena propuesta"]:
        print(t, "->", a.analizar(t))
