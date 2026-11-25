# Sistemas inteligentes integrados – Financiera Confianza Perú

Trabajo individual del curso **Sistemas de Información Integrados** (UPAO, 2026-20).
Autor: Romero Uriol, Jeffry Anderson · Docente: Ing. Carlos Alfredo Mendoza Corpus.

Contiene el código documentado de la propuesta **"Transformación operativa y prevención de fraudes en Financiera Confianza Perú"** y de las tareas de investigación (Machine Learning / Deep Learning y predicción electoral 2026).

> Todos los datos de clientes, créditos y expedientes son **sintéticos** (generados por código). No hay datos de personas reales (Ley N.° 29733). Las encuestas y el resultado electoral sí son reales y su fuente está en cada CSV.

## Estructura

| Carpeta | Qué resuelve | Script principal |
|---|---|---|
| `00_arquitectura/` | Esquema lógico de integración ERP/CRM/ITSM + middleware + motores de IA (Graphviz) | `arquitectura.dot` |
| `01_mineria_procesos/` | **Componente 1** – minería de procesos: descubrimiento, conformidad (token-based replay contra el modelo normativo) y mejora del proceso de aprobación de créditos | `mineria_procesos.py` |
| `02_ml_scoring_fraude/` | **Componente 2** – ML en el ERP: detección de fraude (regresión logística, Random Forest, Gradient Boosting, Isolation Forest) y scoring crediticio dinámico 300–850 → decisión APROBAR / REVISIÓN / RECHAZAR. **Ejemplo de Machine Learning** | `ml_scoring_fraude.py` |
| `03_dl_autoencoder/` | **Ejemplo de Deep Learning** – autoencoder (PyTorch) para detectar fraude nuevo sin etiquetas | `dl_autoencoder.py` |
| `04_asistente_virtual/` | **Componente 3** – asistente virtual 24/7 (PLN): intenciones + entidades + consultas a la BD del core e ITSM; API REST para WhatsApp/app/web/Teams | `asistente.py`, `api.py` |
| `05_prediccion_elecciones/` | Predicción en tiempo real de la segunda vuelta Perú 2026 con encuestas + análisis de sentimiento en redes (RoBERTuito / léxico) | `prediccion_tiempo_real.py` |
| `comun/` | Generador de datos sintéticos de solicitudes de microcrédito | `datos_sinteticos.py` |
| `figures/`, `resultados/` | Gráficos y métricas generadas por los scripts | — |

## Cómo ejecutar

```bash
pip install -r requirements.txt
python 01_mineria_procesos/mineria_procesos.py
python 02_ml_scoring_fraude/ml_scoring_fraude.py
python 03_dl_autoencoder/dl_autoencoder.py
python 04_asistente_virtual/asistente.py            # demo;  --chat para conversar
python 05_prediccion_elecciones/prediccion_tiempo_real.py   # demo; --vivo con X_BEARER_TOKEN
dot -Tpng 00_arquitectura/arquitectura.dot -o figures/arquitectura_integracion.png
```

Para el análisis de sentimiento con el transformer RoBERTuito: `pip install pysentimiento` (descarga el modelo de Hugging Face). Sin conexión, el código usa automáticamente el léxico de respaldo.

## Resultados principales (semilla fija, reproducibles)

| Componente | Resultado |
|---|---|
| Minería de procesos | 2 000 expedientes, 17 690 eventos, 9 variantes; **35,0 % no conformes** (fitness del log 0,945). Retroceso a asesor: mediana 64,3 h vs 18,9 h de los conformes. Solo 38,9 % de los no conformes cumple la meta de 24 h (vs 72,2 % de los conformes). |
| Fraude (ML) | Random Forest: ROC-AUC 0,987, PR-AUC 0,894, recall 88,9 % con 1 % de falsas alarmas. Regla estática actual: recall 45,2 % con 21,6 % de alertas. |
| Scoring dinámico | Gradient Boosting de mora > 90 días: AUC 0,816 (Gini 0,632). Ejemplo: un cliente pasa de 591 a 684 puntos tras 6 meses de buen comportamiento. |
| Autoencoder (DL) | Sin etiquetas: ROC-AUC 0,879; con umbral p99 detecta 20,6 % del fraude con 1,3 % de falsas alarmas (complemento para fraude nuevo). |
| Asistente virtual | Clasificador de 9 intenciones: accuracy 0,768 en paráfrasis no vistas (corpus pequeño; se amplía con conversaciones reales). |
| Elecciones 2026 | Nowcast final (demo con flujo simulado de redes): Fujimori 51,18 % [49,2–53,3], P(gana) = 77 %. Real ONPE/JNE: 50,135 %. Error 1,04 pts vs 5,28 pts usando solo redes. |
