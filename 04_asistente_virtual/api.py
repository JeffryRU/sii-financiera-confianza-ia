"""
API REST del asistente virtual (punto de integración con el middleware / API Gateway).

Canales soportados por el mismo endpoint:
  - WhatsApp Business Cloud API (webhook -> este servicio)
  - App móvil y web (widget de chat)
  - Microsoft Teams / mesa de ayuda interna para asesores de campo

Ejecutar:
    pip install fastapi uvicorn
    uvicorn 04_asistente_virtual.api:app --reload   (desde la raíz del repositorio)
Probar:
    curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
         -d '{"sesion":"abc","usuario":"cliente","canal":"web","texto":"cuanto debo"}'
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

_spec = importlib.util.spec_from_file_location("asistente", Path(__file__).with_name("asistente.py"))
asistente = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(asistente)

modelo, _ = asistente.entrenar_clasificador()
asistente.crear_bd()
servicios = asistente.ServiciosCore()
SESIONES: dict[str, "asistente.Asistente"] = {}

app = FastAPI(title="Confi – Asistente virtual Financiera Confianza (prototipo)")


class Mensaje(BaseModel):
    sesion: str
    usuario: str = "cliente"
    canal: str = "web"          # web | app | whatsapp | teams
    texto: str


@app.post("/chat")
def chat(m: Mensaje):
    a = SESIONES.setdefault(m.sesion, asistente.Asistente(modelo, servicios))
    r = a.responder(m.texto, m.usuario)
    return {"canal": m.canal, **r}


@app.get("/salud")
def salud():
    return {"estado": "ok", "intenciones": list(modelo.classes_)}
