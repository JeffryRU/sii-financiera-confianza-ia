"""
Componente 3 – Asistente virtual 24/7 con PLN (clientes y asesores de campo)
Caso: Financiera Confianza Perú

Arquitectura del prototipo
--------------------------
canal (WhatsApp / app / web / Teams)  ->  /chat (API, ver api.py)
   -> 1. Clasificador de intención (PLN: TF-IDF de n-gramas de caracteres + regresión logística)
   -> 2. Extracción de entidades (DNI, código de crédito, código de ticket) con expresiones regulares
   -> 3. Orquestador: consulta la BASE DE DATOS (réplica de lectura del core / ERP y del ITSM)
         mediante funciones de servicio; nunca SQL generado por el usuario.
   -> 4. Respuesta + registro de la conversación para auditoría.
Si la confianza de la intención es < 0.45 o el usuario lo pide, deriva a un agente humano.

La base de datos es SQLite con datos FICTICIOS que imita las tablas del core:
clientes, creditos, cuotas, solicitudes y tickets (mesa de ayuda ITSM).

Ejecución:
    python 04_asistente_virtual/asistente.py          # entrena, evalúa y corre una demo
    python 04_asistente_virtual/asistente.py --chat   # conversación interactiva por consola
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"; RES.mkdir(exist_ok=True)
BD = RES / "core_ficticio.db"
UMBRAL_CONFIANZA = 0.45

# ------------------------------------------------------------------ corpus de intenciones
FRASES = {
    "estado_cuenta": ["cuánto debo", "cuál es mi saldo", "quiero ver mi estado de cuenta", "saldo de mi crédito",
                      "cuanto me falta pagar", "deuda pendiente", "mi saldo capital", "estado de mi prestamo",
                      "cuánto debo todavía de mi préstamo", "consultar deuda", "ver saldo pendiente",
                      "me dices mi saldo por favor", "quiero saber cuanto debo", "estado de cuenta de mi credito"],
    "proxima_cuota": ["cuándo pago mi cuota", "fecha de mi próxima cuota", "cuánto es mi cuota", "cuándo vence",
                      "monto de la siguiente cuota", "qué día tengo que pagar", "próximo pago", "vencimiento de cuota",
                      "cuanto pago este mes", "cuando vence mi letra", "fecha de pago", "mi cuota de este mes"],
    "estado_solicitud": ["cómo va mi solicitud", "ya aprobaron mi crédito", "estado de mi solicitud",
                         "en qué va mi préstamo nuevo", "mi solicitud sigue en evaluación", "cuando me desembolsan",
                         "ya salió mi crédito", "seguimiento de solicitud", "aprobaron mi expediente",
                         "cuando depositan el prestamo", "revisaron mi solicitud"],
    "reportar_incidente": ["la app no funciona", "no puedo ingresar al sistema", "error al registrar solicitud",
                           "la tablet no sincroniza", "se cayó el sistema", "no carga la evaluación",
                           "tengo un problema con el aplicativo", "mi usuario está bloqueado", "falla en el core",
                           "no imprime el cronograma", "reportar una falla", "el sistema me bota",
                           "no puedo subir las fotos del negocio", "el aplicativo de campo se cuelga"],
    "estado_ticket": ["cómo va mi ticket", "estado del incidente", "ya resolvieron mi ticket", "seguimiento de mi reporte",
                      "mi caso de soporte", "en qué quedó mi incidencia", "revisar ticket", "ya atendieron mi reporte"],
    "reprogramacion": ["quiero reprogramar mi deuda", "no puedo pagar este mes", "necesito refinanciar",
                       "puedo cambiar la fecha de pago", "solicitar reprogramación", "me atrasé con la cuota",
                       "ampliar el plazo", "tengo problemas para pagar", "congelar cuotas"],
    "agencias_horario": ["qué agencias hay en trujillo", "horario de atención", "dónde queda la agencia",
                         "a qué hora abren", "agencia más cercana", "atienden los sábados", "dirección de la oficina",
                         "dónde puedo pagar", "agentes autorizados para pagar"],
    "hablar_humano": ["quiero hablar con una persona", "comunícame con un asesor", "necesito un humano",
                      "pásame con un agente", "atención personalizada", "no me entiendes quiero un asesor",
                      "llamar a un operador"],
    "saludo": ["hola", "buenos días", "buenas tardes", "buenas noches", "hola qué tal", "saludos"],
}


def corpus():
    X, y = [], []
    for intent, frases in FRASES.items():
        for f in frases:
            X.append(f); y.append(intent)
            X.append(f.capitalize() + "?"); y.append(intent)          # variante con mayúscula/puntuación
            X.append(f.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u"))
            y.append(intent)                                           # variante sin tildes (escritura real)
    return X, y


def entrenar_clasificador(semilla=42):
    X, y = corpus()
    # Separamos por FRASE BASE para que la prueba no contenga variantes de frases vistas
    bases = np.arange(len(X)) // 3
    ub = np.unique(bases)
    tr_b, te_b = train_test_split(ub, test_size=0.25, random_state=semilla, stratify=[y[b * 3] for b in ub])
    tr = np.isin(bases, tr_b); te = np.isin(bases, te_b)
    X = np.array(X); y = np.array(y)
    modelo = Pipeline([("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
                       ("clf", LogisticRegression(max_iter=3000, C=8))])
    modelo.fit(X[tr], y[tr])
    pred = modelo.predict(X[te])
    met = {"accuracy": accuracy_score(y[te], pred), "f1_macro": f1_score(y[te], pred, average="macro"),
           "n_entrenamiento": int(tr.sum()), "n_prueba": int(te.sum()),
           "reporte": classification_report(y[te], pred, output_dict=True, zero_division=0)}
    modelo.fit(X, y)  # modelo final con todo el corpus
    return modelo, met


# ------------------------------------------------------------------ base de datos ficticia
def crear_bd(ruta: Path = BD) -> None:
    ruta.unlink(missing_ok=True)
    con = sqlite3.connect(ruta); c = con.cursor()
    c.executescript("""
    CREATE TABLE clientes(dni TEXT PRIMARY KEY, nombre TEXT, agencia TEXT, telefono TEXT);
    CREATE TABLE creditos(cod TEXT PRIMARY KEY, dni TEXT REFERENCES clientes, producto TEXT,
                          monto REAL, saldo_capital REAL, estado TEXT, dias_atraso INTEGER);
    CREATE TABLE cuotas(cod_credito TEXT REFERENCES creditos, n INTEGER, vencimiento TEXT, monto REAL, pagada INTEGER);
    CREATE TABLE solicitudes(cod TEXT PRIMARY KEY, dni TEXT, monto REAL, etapa TEXT, actualizado TEXT);
    CREATE TABLE tickets(cod TEXT PRIMARY KEY, solicitante TEXT, descripcion TEXT, categoria TEXT,
                         prioridad TEXT, estado TEXT, creado TEXT, sla_horas INTEGER);
    CREATE TABLE agencias(nombre TEXT, ciudad TEXT, direccion TEXT, horario TEXT);
    """)
    c.executemany("INSERT INTO clientes VALUES(?,?,?,?)", [
        ("40000001", "Rosa Quispe (ficticio)", "Trujillo Centro", "+51 900 000 001"),
        ("40000002", "Julio Huamán (ficticio)", "La Esperanza", "+51 900 000 002")])
    c.executemany("INSERT INTO creditos VALUES(?,?,?,?,?,?,?)", [
        ("CR-0001", "40000001", "Crédito Emprendedor", 8000, 5230.40, "VIGENTE", 0),
        ("CR-0002", "40000002", "Crédito Mujer Emprendedora", 3500, 1980.10, "VIGENTE", 12)])
    hoy = date.today()
    for cod, base in (("CR-0001", 412.35), ("CR-0002", 236.80)):
        for n in range(1, 13):
            v = hoy.replace(day=1) + timedelta(days=31 * (n - 5))
            c.execute("INSERT INTO cuotas VALUES(?,?,?,?,?)", (cod, n, v.replace(day=15).isoformat(), base, int(n <= 5)))
    c.executemany("INSERT INTO solicitudes VALUES(?,?,?,?,?)", [
        ("SOL-7781", "40000001", 15000, "Verificación de campo", (datetime.now() - timedelta(hours=9)).isoformat(timespec="minutes"))])
    c.executemany("INSERT INTO tickets VALUES(?,?,?,?,?,?,?,?)", [
        ("INC-1042", "asesor_07", "Tablet no sincroniza expedientes", "Aplicativo de campo", "Alta", "En atención",
         (datetime.now() - timedelta(hours=3)).isoformat(timespec="minutes"), 4)])
    c.executemany("INSERT INTO agencias VALUES(?,?,?,?)", [
        ("Trujillo Centro", "Trujillo", "Jr. Pizarro 000 (ficticio)", "L-V 9:00-18:00, S 9:00-13:00"),
        ("La Esperanza", "Trujillo", "Av. Ficticia 123", "L-V 9:00-18:00, S 9:00-13:00")])
    con.commit(); con.close()


# ------------------------------------------------------------------ servicios de consulta (capa API)
class ServiciosCore:
    """Funciones que en producción serían llamadas REST al middleware (API Gateway)."""

    def __init__(self, ruta=BD):
        self.con = sqlite3.connect(ruta); self.con.row_factory = sqlite3.Row

    def credito_de(self, dni):
        return self.con.execute("SELECT * FROM creditos WHERE dni=?", (dni,)).fetchone()

    def proxima_cuota(self, cod):
        return self.con.execute("SELECT * FROM cuotas WHERE cod_credito=? AND pagada=0 ORDER BY n LIMIT 1", (cod,)).fetchone()

    def solicitud_de(self, dni):
        return self.con.execute("SELECT * FROM solicitudes WHERE dni=? ORDER BY actualizado DESC LIMIT 1", (dni,)).fetchone()

    def ticket(self, cod):
        return self.con.execute("SELECT * FROM tickets WHERE cod=?", (cod,)).fetchone()

    def crear_ticket(self, solicitante, descripcion):
        cat, pri, sla = clasificar_ticket(descripcion)
        n = self.con.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] + 1043
        cod = f"INC-{n}"
        self.con.execute("INSERT INTO tickets VALUES(?,?,?,?,?,?,?,?)", (cod, solicitante, descripcion, cat, pri,
                         "Registrado", datetime.now().isoformat(timespec="minutes"), sla))
        self.con.commit()
        return cod, cat, pri, sla

    def agencias(self):
        return self.con.execute("SELECT * FROM agencias").fetchall()


def clasificar_ticket(texto: str):
    """Triaje automático del incidente para el ITSM (categoría, prioridad y SLA en horas)."""
    t = texto.lower()
    if any(k in t for k in ("bloquead", "contraseña", "usuario", "ingresar")):
        return "Accesos", "Media", 8
    if any(k in t for k in ("cayó", "cayo", "core", "bota", "todos")):
        return "Core financiero", "Crítica", 2
    if any(k in t for k in ("tablet", "sincroniza", "aplicativo", "app", "fotos")):
        return "Aplicativo de campo", "Alta", 4
    return "General", "Baja", 24


# ------------------------------------------------------------------ orquestador del diálogo
RE_DNI = re.compile(r"\b(\d{8})\b")
RE_TICKET = re.compile(r"\b(INC-\d{3,6})\b", re.I)


class Asistente:
    def __init__(self, modelo, servicios: ServiciosCore):
        self.m, self.s = modelo, servicios
        self.contexto = {}   # memoria de la sesión (DNI identificado, etc.)
        self.registro = []   # log de conversación para auditoría

    def responder(self, texto: str, usuario: str = "cliente") -> dict:
        probs = self.m.predict_proba([texto])[0]
        i = int(np.argmax(probs)); intent = self.m.classes_[i]; conf = float(probs[i])
        if (d := RE_DNI.search(texto)):
            self.contexto["dni"] = d.group(1)
        if (t := RE_TICKET.search(texto)):
            self.contexto["ticket"] = t.group(1).upper()
        if conf < UMBRAL_CONFIANZA and not RE_TICKET.search(texto):
            intent = "derivar"
        if "pendiente" in self.contexto and not RE_TICKET.search(texto) and intent in ("derivar", "reportar_incidente"):
            intent = self.contexto.pop("pendiente")  # el usuario completó el dato que faltaba
        r = self._ejecutar(intent, texto, usuario)
        self.registro.append({"hora": datetime.now().isoformat(timespec="seconds"), "usuario": usuario,
                              "texto": texto, "intencion": intent, "confianza": round(conf, 3), "respuesta": r})
        return {"intencion": intent, "confianza": round(conf, 3), "respuesta": r}

    def _requiere_dni(self, intent):
        if "dni" not in self.contexto:
            self.contexto["pendiente"] = intent
            return "Para proteger tus datos necesito tu DNI (8 dígitos). Te enviaremos un código de verificación por SMS."
        return None

    def _ejecutar(self, intent, texto, usuario):
        s = self.s
        if intent == "saludo":
            return "¡Hola! Soy Confi, el asistente de Financiera Confianza. Puedo darte tu saldo, tu próxima cuota, el estado de tu solicitud o registrar un incidente."
        if intent in ("estado_cuenta", "proxima_cuota", "estado_solicitud", "reprogramacion"):
            if (msg := self._requiere_dni(intent)):
                return msg
            dni = self.contexto["dni"]
            if intent == "estado_solicitud":
                sol = s.solicitud_de(dni)
                return (f"Tu solicitud {sol['cod']} por S/ {sol['monto']:,.2f} está en la etapa «{sol['etapa']}» "
                        f"(actualizado {sol['actualizado']}). Te avisaremos por este canal cuando cambie.") if sol else \
                       "No encuentro solicitudes en curso con ese DNI."
            cr = s.credito_de(dni)
            if not cr:
                return "No encuentro créditos activos con ese DNI."
            if intent == "estado_cuenta":
                atr = f" Tienes {cr['dias_atraso']} días de atraso." if cr["dias_atraso"] else " Estás al día."
                return f"Tu {cr['producto']} ({cr['cod']}) tiene un saldo de capital de S/ {cr['saldo_capital']:,.2f}.{atr}"
            if intent == "proxima_cuota":
                q = s.proxima_cuota(cr["cod"])
                return f"Tu próxima cuota (n.° {q['n']}) es de S/ {q['monto']:,.2f} y vence el {q['vencimiento']}."
            return ("Puedo iniciar tu solicitud de reprogramación. Un asesor evaluará tu caso en máximo 48 h; "
                    "recuerda que la reprogramación puede modificar tu cronograma y el costo total del crédito.")
        if intent == "reportar_incidente":
            cod, cat, pri, sla = s.crear_ticket(usuario, texto)
            return f"Registré el incidente {cod} (categoría: {cat}, prioridad: {pri}). Tiempo objetivo de atención: {sla} h."
        if intent == "estado_ticket":
            cod = self.contexto.get("ticket")
            if not cod:
                self.contexto["pendiente"] = "estado_ticket"
                return "Indícame el código del ticket (por ejemplo, INC-1042)."
            t = s.ticket(cod)
            return f"El ticket {cod} ({t['descripcion']}) está «{t['estado']}», prioridad {t['prioridad']}." if t \
                else f"No encuentro el ticket {cod}."
        if intent == "agencias_horario":
            return " | ".join(f"{a['nombre']}: {a['direccion']}, {a['horario']}" for a in s.agencias())
        return "Te comunico con un asesor humano. Tiempo estimado de espera: 3 minutos."


def demo(asistente: Asistente):
    dialogo = [("cliente", "Hola"), ("cliente", "cuanto debo de mi prestamo?"), ("cliente", "mi dni es 40000001"),
               ("cliente", "cuando vence mi cuota"), ("cliente", "ya aprobaron mi credito?"),
               ("asesor_12", "la tablet no sincroniza los expedientes"), ("asesor_07", "cómo va mi ticket INC-1042"),
               ("cliente", "quiero hablar con una persona")]
    asistente.contexto.clear()
    out = []
    for u, t in dialogo:
        r = asistente.responder(t, u)
        out.append({"usuario": u, "mensaje": t, **r})
        print(f"[{u}] {t}\n  -> ({r['intencion']}, {r['confianza']:.2f}) {r['respuesta']}")
    return out


def main():
    modelo, met = entrenar_clasificador()
    print(f"Clasificador de intenciones: accuracy={met['accuracy']:.3f}, F1-macro={met['f1_macro']:.3f}")
    crear_bd()
    a = Asistente(modelo, ServiciosCore())
    if "--chat" in sys.argv:
        print("Escribe 'salir' para terminar.")
        while (t := input("> ")).lower() != "salir":
            print(a.responder(t)["respuesta"])
        return
    conv = demo(a)
    (RES / "chatbot_metricas.json").write_text(json.dumps({"clasificador": {k: v for k, v in met.items() if k != "reporte"},
                                                           "por_intencion": met["reporte"], "demo": conv},
                                                          indent=2, ensure_ascii=False), encoding="utf-8")
    import joblib; joblib.dump(modelo, RES / "chatbot_intenciones.joblib")


if __name__ == "__main__":
    main()
