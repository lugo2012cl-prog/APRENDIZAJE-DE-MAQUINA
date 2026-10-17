import base64
import html as html_lib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent          # carpeta TP FINAL
NOTEBOOKS = [RAIZ / f"{n}.ipynb" for n in
             ["01_EDA_y_seleccion", "02_Planteo_y_basal", "03_Modelos_y_ajuste", "04_Evaluacion_y_conclusiones"]]
ARTEFACTOS = RAIZ / "artefactos_ml"

st.set_page_config(
    page_title="Diabetes 130-US Hospitals: TP Aprendizaje de Máquina",
    page_icon="🤖",
    layout="wide",
)

VERDE = "#0F6E5F"
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
    h1 { margin-bottom: 0.1rem; font-size: 2.1rem; }
    [data-testid="stCaptionContainer"] { margin-bottom: 0.4rem; }
    div[role="radiogroup"] { gap: 0.3rem; flex-wrap: wrap; }
    div[role="radiogroup"] label {
        background-color: rgba(151, 166, 195, 0.10);
        border-radius: 6px; padding: 0.35rem 0.8rem; margin-right: 0;
    }
    /* tablas */
    .tb-wrap { overflow-x: auto; margin: 0.4rem 0 1rem 0; border-radius: 10px;
               border: 1px solid rgba(128,128,128,0.25); }
    table.tb { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
    table.tb thead th { background: #0F6E5F; color: #fff; padding: 9px 12px; text-align: center;
                        font-weight: 600; white-space: nowrap; border: none; }
    table.tb thead th:first-child { text-align: left; }
    table.tb tbody td, table.tb tbody th { padding: 7px 12px; border-bottom: 1px solid rgba(128,128,128,0.18);
                        text-align: center; font-variant-numeric: tabular-nums; }
    table.tb tbody th { text-align: left; font-weight: 600; }
    table.tb tbody tr:nth-child(even) { background: rgba(128,128,128,0.07); }
    table.tb tbody tr:hover { background: rgba(15,110,95,0.14); }
    table.tb tbody tr:last-child td, table.tb tbody tr:last-child th { border-bottom: none; }
    td.txt { text-align: left !important; }
    /* tarjetas y flujo */
    .kpis { display: flex; gap: 12px; flex-wrap: wrap; margin: 0.3rem 0 1rem 0; }
    .kpi { flex: 1 1 150px; border-radius: 12px; padding: 14px 16px; color: #fff;
           background: linear-gradient(135deg, #0F6E5F, #3E6B96); }
    .kpi .v { font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
    .kpi .l { font-size: 0.82rem; opacity: 0.9; margin-top: 3px; }
    .flujo { display: flex; align-items: stretch; flex-wrap: wrap; gap: 6px; margin: 0.4rem 0 1rem 0; }
    .paso { flex: 1 1 130px; border-radius: 10px; padding: 10px 12px; font-size: 0.85rem;
            background: rgba(15,110,95,0.13); border: 1px solid rgba(15,110,95,0.45); }
    .paso b { display: block; margin-bottom: 2px; }
    .paso.sellado { background: rgba(196,78,82,0.14); border-color: rgba(196,78,82,0.55); }
    .flecha { align-self: center; font-size: 1.3rem; opacity: 0.6; }
    .cv-fila { display: flex; gap: 4px; margin-bottom: 4px; align-items: center; }
    .cv-nombre { width: 60px; font-size: 0.8rem; opacity: 0.8; }
    .cv-blq { flex: 1; height: 26px; border-radius: 5px; background: rgba(62,107,150,0.35); }
    .cv-blq.val { background: #D98E4A; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 0.3rem 0 1rem 0; }
    .chip { border-radius: 999px; padding: 4px 12px; font-size: 0.82rem;
            background: rgba(151,166,195,0.18); border: 1px solid rgba(151,166,195,0.4); }
    .chip b { color: #0F6E5F; }
    </style>
    """,
    unsafe_allow_html=True,
)

PALETA = {"Basal": "#8C8C8C", "Regresión logística": "#55A868", "Random Forest": "#DD8452",
          "XGBoost": "#4C72B0", "CatBoost": "#C44E52"}
MECANISMOS = {"Grilla": "#8C8C8C", "Aleatoria": "#DD8452", "Optuna (TPE)": "#4C72B0"}

# Botones: etiqueta -> texto del encabezado del notebook donde empieza cada sección
SECCIONES = [
    ("EDA", "## 7.5 Balance de clases"),
    ("Enganche", "# 8."),
    ("Basal", "## 8.1 M1"),
    ("Definición", "## 8.2 M2"),
    ("Validación", "## 8.3 M3"),
    ("Algoritmos", "## 8.4 M4"),
    ("Hiperparámetros", "## 8.5 M5"),
    ("Umbral", "## 8.6 M6"),
    ("Comparación", "## 8.7 M7"),
    ("Interpretabilidad", "## 8.8 M8"),
    ("Test final", "## 8.9 M9"),
    ("Cierre", "## 8.10 M10"),
]
NOMBRE_ETAPA = {0: "enganche", 1: "basal", 2: "definición", 3: "validación", 4: "algoritmos",
                5: "hiperparámetros", 6: "umbral", 7: "comparación", 8: "interpretabilidad",
                9: "test final", 10: "cierre"}


# ----------------------------------------------------------------------------
# Lectura del notebook
# ----------------------------------------------------------------------------
def es_de_enlace(celda):
    """Celdas que solo unen un notebook con el siguiente (portada, cierre y guardado o carga de la sesión)."""
    fuente = "".join(celda["source"])
    if celda["cell_type"] == "code":
        return "dill." in fuente
    primera = fuente.lstrip("-\n ").split("\n")[0]
    return bool(re.match(r"(# Notebook \d de 4|## Cierre del notebook|### Estado de la sesión)", primera))


@st.cache_data(show_spinner="Cargando…")
def cargar_notebook():
    celdas = []
    for ruta in NOTEBOOKS:  # los cuatro notebooks, en orden, se leen como uno solo
        with open(ruta, encoding="utf-8") as f:
            celdas += json.load(f)["cells"]
    celdas = [c for c in celdas if not es_de_enlace(c)]
    fuentes = ["".join(c["source"]) for c in celdas]
    inicios = []
    for _, enc in SECCIONES:
        inicios.append(next(i for i, s in enumerate(fuentes)
                            if celdas[i]["cell_type"] == "markdown" and any(l.startswith(enc) for l in s.split("\n"))))
    rangos = {}
    for k, (etq, _) in enumerate(SECCIONES):
        fin = inicios[k + 1] if k + 1 < len(SECCIONES) else len(celdas)
        rangos[etq] = celdas[inicios[k]:fin]
    eda = [c for c in celdas[:inicios[0]]]
    return rangos, eda


@st.cache_data
def csv(nombre):
    return pd.read_csv(ARTEFACTOS / nombre)


def limpiar(t):
    """Quita las referencias M0..M10 y la numeración de secciones del modelado."""
    t = re.sub(r"^(#+)\s+8\.\d+(\.\d+)*\s+(M\d+(?::|\s+—)\s+)?", r"\1 ", t, flags=re.M)
    t = re.sub(r"^#\s+8\.\s+", "# ", t, flags=re.M)
    t = re.sub(r"\b(?:de|en|a|con|la|el|sección)\s+(?:la\s+)?sección 8\.3\.1", "la etapa de validación", t)
    t = t.replace("la sección 8.3.1", "la etapa de validación").replace("en 8.9", "en la etapa de test final")
    t = re.sub(r"\bM(\d+)\b", lambda m: f"la etapa de {NOMBRE_ETAPA.get(int(m.group(1)), '')}", t)
    t = t.replace("la la etapa", "la etapa").replace("de la etapa de comparación y desde la etapa", "de la etapa")
    return t


def num_es(s):
    s = re.sub(r"(?<=\d)\.(?=\d)", ",", s)
    return s


# ----------------------------------------------------------------------------
# Tablas estéticas
# ----------------------------------------------------------------------------
class _ParserTabla(HTMLParser):
    def __init__(self):
        super().__init__()
        self.filas, self.fila, self.celda, self.en_thead = [], None, None, False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "thead":
            self.en_thead = True
        elif tag == "tbody":
            self.en_thead = False
        elif tag == "tr":
            self.fila = []
        elif tag in ("th", "td"):
            self.celda = {"tag": tag, "texto": "", "colspan": int(a.get("colspan", 1)),
                          "rowspan": int(a.get("rowspan", 1)), "thead": self.en_thead}

    def handle_data(self, data):
        if self.celda is not None:
            self.celda["texto"] += data

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.celda is not None:
            self.fila.append(self.celda)
            self.celda = None
        elif tag == "tr" and self.fila is not None:
            self.filas.append(self.fila)
            self.fila = None


def parsear_tabla(h):
    p = _ParserTabla()
    p.feed(h)
    cab = [[c for c in f for _ in range(c["colspan"])] for f in p.filas if f and f[0]["thead"]]
    cuerpo = [f for f in p.filas if f and not f[0]["thead"]]
    # matriz del cuerpo (con rowspan)
    matriz, activos = [], {}
    for f in cuerpo:
        fila, j, it = [], 0, iter(f)
        while True:
            if j in activos:
                txt, rest = activos[j]
                fila.append(txt)
                activos[j] = (txt, rest - 1) if rest > 1 else None
                if activos[j] is None:
                    del activos[j]
                j += 1
                continue
            c = next(it, None)
            if c is None:
                break
            for _ in range(c["colspan"]):
                fila.append(c["texto"].strip())
                if c["rowspan"] > 1:
                    activos[j] = (c["texto"].strip(), c["rowspan"] - 1)
                j += 1
        matriz.append(fila)
    n_idx = max((sum(1 for c in f if c["tag"] == "th") for f in cuerpo), default=1)
    nombres_idx = [""] * n_idx
    filas_cab = []
    for f in cab:
        textos = [c["texto"].strip() for c in f]
        if textos and any(textos[:n_idx]) and not any(textos[n_idx:]):
            nombres_idx = (textos[:n_idx] + [""] * n_idx)[:n_idx]
        else:
            filas_cab.append(textos)
    ncol = len(matriz[0]) if matriz else 0
    columnas = []
    for j in range(n_idx, ncol):
        partes = []
        for f in filas_cab:
            if j < len(f) and f[j] and f[j] not in partes:
                partes.append(f[j])
        columnas.append(" · ".join(partes))
    return nombres_idx, columnas, matriz, n_idx


_RE_NUM = re.compile(r"-?\d+(?:[.,]\d+)?(?:e-?\d+)?", re.I)
_RE_METRICA = re.compile(r"ROC-AUC|AUC-PR|recall|precisión|F1|exactitud|mejora", re.I)
_RE_NO_METRICA = re.compile(r"desvío|optimismo|segundos|umbral|^p |valor p|tasa de|n=|folds", re.I)


TRADUCCIONES = {
    "Discharged to home": "Alta a domicilio",
    "Discharged/transferred to SNF": "Alta a centro de enfermería especializada",
    "Discharged/transferred to another short te...": "Alta a otro hospital de corta estadía",
    "Discharged/transferred to home with home h...": "Alta a domicilio con atención domiciliaria",
    "Female": "mujeres", "Male": "varones", "3 o mas": "3 o más",
}


def fmt_celda(s):
    s = TRADUCCIONES.get(s.strip(), limpiar(s))
    if re.fullmatch(r"-?\d+\.\d+e-?\d+", s):
        return f"{float(s):.1e}".replace(".", ",")
    return num_es(s)


def tabla_bonita(nombres_idx, columnas, matriz, n_idx=1, calor=True):
    cab = list(nombres_idx) + list(columnas)
    cab = [limpiar(c) for c in cab]
    ncol = len(cab)
    calor = calor and "métrica" not in cab          # filas de métricas distintas no son comparables entre sí
    # mapas de color por columna
    colores = {}
    if calor:
        for j in range(n_idx, ncol):
            nombre = cab[j]
            if not _RE_METRICA.search(nombre) or _RE_NO_METRICA.search(nombre):
                continue
            vals = []
            for f in matriz:
                m = _RE_NUM.search(f[j].replace(",", ".")) if j < len(f) else None
                vals.append(float(m.group()) if m else None)
            ok = [v for v in vals if v is not None]
            if len(ok) >= 2 and max(ok) > min(ok):
                colores[j] = (vals, min(ok), max(ok))
    cuerpo = []
    for i, f in enumerate(matriz):
        celdas = []
        for j, v in enumerate(f[:ncol]):
            txt = html_lib.escape(fmt_celda(v))
            if j < n_idx:
                celdas.append(f"<th>{txt}</th>")
                continue
            estilo, extra = "", ""
            if j in colores:
                vals, lo, hi = colores[j]
                if vals[i] is not None:
                    a = 0.10 + 0.45 * (vals[i] - lo) / (hi - lo)
                    estilo = f' style="background: rgba(15,110,95,{a:.2f});"'
                    if vals[i] == hi:
                        txt = f"<b>{txt}</b>"
            if not re.search(r"\d", v) or len(v) > 40:
                extra = ' class="txt"'
            celdas.append(f"<td{extra}{estilo}>{txt}</td>")
        cuerpo.append("<tr>" + "".join(celdas) + "</tr>")
    thead = "<tr>" + "".join(f"<th>{html_lib.escape(c)}</th>" for c in cab) + "</tr>"
    st.html(f'<div class="tb-wrap"><table class="tb"><thead>{thead}</thead><tbody>{"".join(cuerpo)}</tbody></table></div>')


def tabla_desde_df(df, indice=True, calor=True):
    d = df.reset_index() if indice else df
    matriz = [[("" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, float) else str(v))) for v in fila]
              for fila in d.itertuples(index=False)]
    n_idx = 1 if indice else 0
    tabla_bonita([str(c) for c in d.columns[:n_idx]], [str(c) for c in d.columns[n_idx:]], matriz, n_idx, calor)


def tabla_markdown(lineas):
    filas = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lineas if not re.match(r"^\s*\|[\s\-:|]+\|\s*$", l)]
    cab, cuerpo = filas[0], filas[1:]
    # quita el formato en línea (negritas, código) de las celdas
    cuerpo = [[re.sub(r"[`*]", "", c) for c in f] for f in cuerpo]
    cab = [re.sub(r"[`*]", "", c) for c in cab]
    tabla_bonita([cab[0]], cab[1:], [f for f in cuerpo], 1, calor=False)


def markdown_con_tablas(texto):
    texto = limpiar(texto)
    buffer, tabla = [], []
    for linea in texto.split("\n"):
        if linea.lstrip().startswith("|"):
            if buffer:
                st.markdown("\n".join(buffer))
                buffer = []
            tabla.append(linea)
        else:
            if tabla:
                tabla_markdown(tabla)
                tabla = []
            buffer.append(linea)
    if tabla:
        tabla_markdown(tabla)
    if "\n".join(buffer).strip():
        st.markdown("\n".join(buffer))


# ----------------------------------------------------------------------------
# Celdas del notebook
# ----------------------------------------------------------------------------
GANADORES = None


def mostrar_celda(celda, seccion):
    fuente = "".join(celda["source"])
    if celda["cell_type"] == "markdown":
        if seccion == "Cierre":
            fuente = re.split(r"\n### Reproducibilidad", fuente)[0]
            fuente = fuente.replace("Limitaciones, reflexión y reproducibilidad", "Limitaciones y reflexión")
        markdown_con_tablas(fuente)
        return
    imagenes = 0
    for o in celda.get("outputs", []):
        datos = o.get("data", {})
        if o.get("output_type") == "stream":
            txt = "".join(o["text"]).strip()
            if not txt or txt.startswith("{") or any(k in txt for k in ("Guardado", "Tiempo total", "Verdaderos negativos")):
                continue
            st.caption(limpiar(txt))
        elif "image/png" in datos:
            imagenes += 1
            if seccion == "Test final" and imagenes == 2:       # la matriz de confusión se muestra interactiva abajo
                continue
            st.image(base64.b64decode(datos["image/png"]))
        elif "text/html" in datos:
            nombres_idx, columnas, matriz, n_idx = parsear_tabla("".join(datos["text/html"]))
            columnas, matriz = arreglar_tabla(columnas, matriz, n_idx)
            tabla_bonita(nombres_idx, columnas, matriz, n_idx)


def arreglar_tabla(columnas, matriz, n_idx):
    """Quita columnas truncadas o redundantes y completa los hiperparámetros ganadores."""
    quitar = [j for j, c in enumerate(columnas)
              if (c == "hiperparámetros" and any(len(f[n_idx + j]) > 100 or f[n_idx + j].endswith("...") for f in matriz))
              or c == "supera al basal"]
    if "hiperparámetros" in columnas and "ganador elegido" in columnas:
        j = columnas.index("hiperparámetros")
        gan = json.load(open(ARTEFACTOS / "configuracion_m5.json"))["ganadores"]
        for f in matriz:
            hp = gan.get(f[0], {}).get("hiperparametros", {})
            f[n_idx + j] = "; ".join(f"{k} = {v:.3g}" if isinstance(v, float) else f"{k} = {v}" for k, v in hp.items())
        quitar = [q for q in quitar if q != j]
    if quitar:
        columnas = [c for j, c in enumerate(columnas) if j not in quitar]
        matriz = [f[:n_idx] + [v for j, v in enumerate(f[n_idx:]) if j not in quitar] for f in matriz]
    return columnas, matriz


# ----------------------------------------------------------------------------
# EDA (resumen)
# ----------------------------------------------------------------------------
def kpis(items):
    st.html('<div class="kpis">' + "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>'
                                           for v, l in items) + "</div>")


def seccion_eda(eda):
    st.subheader("Resumen del análisis exploratorio")
    kpis([("71.583", "internaciones de entrenamiento (50.060 pacientes)"),
          ("30.180", "internaciones de prueba (21.455 pacientes)"),
          ("18 de 32", "variables seleccionadas"),
          ("11,3 %", "reingresa antes de 30 días")])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Balance de clases (entrenamiento)**")
        tasa = 0.1127
        df = pd.DataFrame({"clase": ["No reingresa / después de 30 d", "Reingresa antes de 30 d"],
                           "%": [(1 - tasa) * 100, tasa * 100]})
        fig = px.bar(df, x="clase", y="%", color="clase", color_discrete_sequence=["#4C72B0", "#C44E52"], text_auto=".1f")
        fig.update_layout(height=360, showlegend=False, xaxis_title="", yaxis_title="% de las internaciones")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**De 32 variables candidatas a 18**")
        fig = go.Figure(go.Funnel(
            y=["Candidatas", "Sin las de baja relevancia (−10)", "Sin las redundantes (−2)", "Sin las discretizadas (−2)"],
            x=[32, 22, 20, 18], textinfo="value", marker=dict(color=["#3E6B96", "#4C72B0", "#55A868", "#0F6E5F"])))
        fig.update_layout(height=360, margin=dict(l=10, r=10))
        st.plotly_chart(fig, width="stretch")

    # V de Cramér leída de la salida del EDA
    filas = []
    for c in eda:
        for o in c.get("outputs", []):
            t = "".join(o.get("text", [])) if o.get("output_type") == "stream" else "".join(o.get("data", {}).get("text/plain", []))
            if "V_de_Cramer" in t:
                for m in re.finditer(r"^\s*\d+\s+(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", t, flags=re.M):
                    filas.append((m.group(1), float(m.group(5))))
    if filas:
        v = pd.DataFrame(filas, columns=["variable", "V de Cramér"]).drop_duplicates("variable").head(12)
        st.markdown("**Asociación de las variables categóricas con el objetivo (V de Cramér)**")
        fig = px.bar(v[::-1], x="V de Cramér", y="variable", orientation="h", color="V de Cramér",
                     color_continuous_scale="Teal")
        fig.update_layout(height=430, coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, width="stretch")
    st.info("Tres variables concentran la señal: las internaciones previas, la cantidad de encuentros previos del "
            "paciente y el tipo de alta. El desbalance es moderado (89 % / 11 %): en el modelado no se aplica ningún "
            "mecanismo de balanceo (ni pesos de clase ni *undersampling*); la métrica primaria es el ROC-AUC, que no "
            "depende de la prevalencia, y el umbral se elige con predicciones fuera de fold.")


# ----------------------------------------------------------------------------
# Enganche (resumen gráfico)
# ----------------------------------------------------------------------------
def seccion_enganche(celdas):
    st.header("Modelado")
    fuente = "".join(celdas[0]["source"])
    cuerpo = re.split(r"\n## 8\.0", fuente)[0]
    cuerpo = re.sub(r"^# 8\..*\n+", "", cuerpo)
    markdown_con_tablas(cuerpo)

    reg = csv("registro_experimentos.csv")
    kpis([("71.583 / 30.180", "filas de entrenamiento / prueba"), ("0", "pacientes compartidos entre ambos"),
          ("14", "núcleos de cómputo"), (f"{len(reg):,}".replace(",", "."), "experimentos registrados")])

    st.markdown("**Flujo de trabajo**")
    pasos = [("Datos", "Partición 70/30 agrupada por paciente"), ("Test sellado 🔒", "Se abre una sola vez, al final"),
             ("Folds", "5 folds agrupados por paciente × 4 repeticiones"),
             ("Preprocesamiento", "Winsorización, target encoding y escalado, dentro de cada fold"),
             ("Modelo", "4 algoritmos contra el basal"), ("Métricas", "ROC-AUC (primaria) y secundarias")]
    partes = []
    for i, (t, d) in enumerate(pasos):
        clase = "paso sellado" if "Test" in t else "paso"
        partes.append(f'<div class="{clase}"><b>{t}</b>{d}</div>')
        if i < len(pasos) - 1:
            partes.append('<div class="flecha">→</div>')
    st.html('<div class="flujo">' + "".join(partes) + "</div>")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Validación cruzada agrupada (5 folds)**")
        filas = []
        for k in range(5):
            bloques = "".join(f'<div class="cv-blq{" val" if j == k else ""}"></div>' for j in range(5))
            filas.append(f'<div class="cv-fila"><div class="cv-nombre">Fold {k + 1}</div>{bloques}</div>')
        st.html("".join(filas) + '<div class="chips"><span class="chip">🟦 entrenamiento</span>'
                '<span class="chip">🟧 validación (≈14.317 filas)</span><span class="chip">× 4 repeticiones (semillas 42 a 45)</span></div>')
    with c2:
        st.markdown("**Costo de un ajuste (segundos, un fold)**")
        t = pd.DataFrame({"algoritmo": ["Regresión logística", "Random Forest", "XGBoost", "CatBoost"],
                          "segundos": [0.0, 1.8, 1.1, 2.8]})
        fig = px.bar(t, x="algoritmo", y="segundos", color="algoritmo", color_discrete_map=PALETA, text_auto=".1f")
        fig.update_layout(height=290, showlegend=False, xaxis_title="", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Entorno**")
    v = csv("versiones.csv")
    st.html('<div class="chips">' + "".join(
        f'<span class="chip"><b>{html_lib.escape(str(r.iloc[0]))}</b> {html_lib.escape(str(r.iloc[1]))}</span>'
        for _, r in v.iterrows()) + "</div>")


# ----------------------------------------------------------------------------
# Complementos interactivos
# ----------------------------------------------------------------------------
def extra_validacion():
    st.markdown("---")
    st.subheader("Estabilidad de la selección de variables")
    c1, c2 = st.columns([3, 2])
    d = pd.DataFrame({"variable": ["num_lab_procedures", "num_procedures", "number_outpatient",
                                   "admission_type_id", "max_glu_serum"],
                      "% de los 20 folds en que se conserva": [90, 50, 40, 15, 5],
                      "decisión del EDA": ["conservada", "descartada", "descartada", "descartada", "descartada"]})
    fig = px.bar(d[::-1], x="% de los 20 folds en que se conserva", y="variable", orientation="h",
                 color="decisión del EDA", color_discrete_map={"conservada": "#0F6E5F", "descartada": "#C44E52"},
                 text="% de los 20 folds en que se conserva")
    fig.update_layout(height=330, yaxis_title="", xaxis_range=[0, 100], legend=dict(orientation="h", y=-0.25))
    c1.plotly_chart(fig, width="stretch")
    fig2 = go.Figure(go.Pie(labels=["Siempre conservadas", "Siempre descartadas", "Cambian según el fold"],
                            values=[17, 10, 5], hole=0.55, marker=dict(colors=["#0F6E5F", "#8C8C8C", "#D98E4A"])))
    fig2.update_layout(height=330, margin=dict(t=10), legend=dict(orientation="h", y=-0.2))
    c2.plotly_chart(fig2, width="stretch")
    c2.caption("Variables que cambian de decisión según el fold; 27 de las 32 toman siempre la misma decisión.")


def extra_m5():
    st.markdown("---")
    st.subheader("🔎 Convergencia interactiva de la carrera")
    ens = csv("m5_ensayos.csv")
    col1, col2 = st.columns([1, 3])
    algoritmo = col1.selectbox("Algoritmo", ens["algoritmo"].unique(), key="m5_alg")
    semillas = sorted(ens["semilla"].unique())
    semilla = col1.selectbox("Semilla de la carrera", semillas, key="m5_sem")
    sub = ens[(ens["algoritmo"] == algoritmo) & (ens["semilla"] == semilla)]
    fig = go.Figure()
    for mec, color in MECANISMOS.items():
        s = sub[sub["mecanismo"] == mec].sort_values("evaluacion")
        if len(s):
            fig.add_trace(go.Scatter(x=s["evaluacion"], y=s["auc_media"].cummax(), mode="lines", name=mec,
                                     line=dict(color=color, width=3)))
            fig.add_trace(go.Scatter(x=s["evaluacion"], y=s["auc_media"], mode="markers", name=f"{mec} (cada evaluación)",
                                     marker=dict(color=color, size=5, opacity=0.35), visible="legendonly"))
    piso = sub["auc_media"][sub["auc_media"] > 0.6].min()
    fig.update_layout(height=420, xaxis_title="Evaluaciones", yaxis_title="Mejor ROC-AUC hasta el momento",
                      yaxis_range=[max(piso - 0.001, 0.6), sub["auc_media"].max() + 0.0015])
    col2.plotly_chart(fig, width="stretch")
    col2.caption("Las líneas son el mejor valor alcanzado hasta cada evaluación; los puntos (activables en la leyenda) son "
                 "cada evaluación individual. Se recortan las evaluaciones degeneradas (ROC-AUC = 0,5).")


def extra_m7():
    st.markdown("---")
    st.subheader("🔎 ROC-AUC por fold (20 folds), interactivo")
    f = csv("m7_folds.csv")
    metrica = st.radio("Métrica", ["ROC-AUC", "AUC-PR"], horizontal=True, key="m7_met")
    fig = px.box(f, x="modelo", y=metrica, color="modelo", points="all", color_discrete_map=PALETA,
                 hover_data=["semilla", "fold"])
    fig.update_layout(height=430, showlegend=False, xaxis_title="")
    st.plotly_chart(fig, width="stretch")
    res = f.groupby("modelo")[metrica].agg(["mean", "std"]).round(4).rename(columns={"mean": "media", "std": "desvío"})
    tabla_desde_df(res.rename_axis("modelo"), calor=False)


def extra_m9():
    st.markdown("---")
    st.subheader("🔎 Métricas en el test con intervalo de confianza (95 %), interactivo")
    ic = csv("m9_resultados_test_ic.csv")
    metrica = st.radio("Métrica", ic["metrica"].unique().tolist(), horizontal=True, key="m9_met")
    s = ic[ic["metrica"] == metrica]
    fig = go.Figure()
    for _, r in s.iterrows():
        fig.add_trace(go.Scatter(
            x=[r["estimacion"]], y=[r["modelo"]], mode="markers",
            marker=dict(color=PALETA.get(r["modelo"], "#555"), size=12), name=r["modelo"],
            error_x=dict(type="data", symmetric=False, array=[r["ic_superior"] - r["estimacion"]],
                         arrayminus=[r["estimacion"] - r["ic_inferior"]], color=PALETA.get(r["modelo"], "#555")),
            hovertemplate=f"{r['modelo']}: {r['estimacion']:.3f} [{r['ic_inferior']:.3f}; {r['ic_superior']:.3f}]<extra></extra>"))
    fig.update_layout(height=330, showlegend=False, xaxis_title=f"{metrica} (conjunto de prueba)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Los intervalos se obtienen por *bootstrap* de pacientes con 1.000 réplicas.")

    cm = json.load(open(ARTEFACTOS / "m9_matriz_confusion.json"))
    mat = np.array(cm["matriz"])
    pct = mat / mat.sum(axis=1, keepdims=True)
    mostrar_pct = st.toggle("Mostrar también el porcentaje (por fila)", value=False, key="m9_cm")
    etiquetas = ["No reingresa", "Reingresa < 30 d"]
    fmt = lambda n: f"{n:,}".replace(",", ".")
    if mostrar_pct:
        textos = [[f"{fmt(mat[i, j])}<br>{str(round(pct[i, j] * 100, 1)).replace('.', ',')} %" for j in range(2)]
                  for i in range(2)]
    else:
        textos = [[fmt(mat[i, j]) for j in range(2)] for i in range(2)]
    fig_cm = px.imshow(pct, x=etiquetas, y=etiquetas, color_continuous_scale="Teal", zmin=0, zmax=1,
                       labels=dict(x="Predicho", y="Real"))
    fig_cm.update_traces(text=textos, texttemplate="%{text}", textfont_size=16)
    fig_cm.update_layout(height=400, width=540, coloraxis_showscale=False,
                         title=f"{cm['modelo']} en el conjunto de prueba, umbral {str(round(cm['umbral'], 2)).replace('.', ',')}")
    st.plotly_chart(fig_cm, width="content")


EXTRAS = {"Validación": extra_validacion, "Hiperparámetros": extra_m5,
          "Comparación": extra_m7, "Test final": extra_m9}

# ----------------------------------------------------------------------------
# Página
# ----------------------------------------------------------------------------
st.title("Diabetes 130-US Hospitals: TP Aprendizaje de Máquina")
st.caption(
    "Predicción del reingreso a 30 días. Continúa el análisis exploratorio del TP de Análisis de Datos con el "
    "modelado de las etapas de aprendizaje de máquina. Cada botón muestra una etapa."
)

etiquetas = [e for e, _ in SECCIONES]
activa = st.radio("Sección", etiquetas, horizontal=True, label_visibility="collapsed", key="seccion_activa")
st.markdown("---")

try:
    secciones, celdas_eda = cargar_notebook()
except (FileNotFoundError, StopIteration):
    st.error(f"No se pudieron leer los cuatro notebooks en `{RAIZ}`. El dashboard debe quedar en la carpeta "
             "`dashboard_ml/` dentro de `TP FINAL`, junto a los notebooks y a `artefactos_ml/`.")
    st.stop()

if activa == "EDA":
    seccion_eda(celdas_eda)
elif activa == "Enganche":
    seccion_enganche(secciones[activa])
else:
    for celda in secciones[activa]:
        mostrar_celda(celda, activa)
    if activa in EXTRAS:
        EXTRAS[activa]()
