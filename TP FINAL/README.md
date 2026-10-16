# TP Final de Aprendizaje de Máquina: reingreso hospitalario en pacientes diabéticos

Aprendizaje de Máquina (CEIA), cohorte 27Co2026.

Integrantes: Florencia Lucca, Mariano Reimondez y Carlos Lugones.

Dataset: Diabetes 130-US Hospitals for Years 1999-2008 (UCI Machine Learning Repository).
Objetivo: predecir, al alta, si un paciente diabético reingresa al hospital dentro de los 30 días.

## Orden de lectura

El trabajo está dividido en cuatro notebooks. Cada uno continúa la sesión del anterior, así que se corren en orden.

| Notebook | Pregunta que responde | Contenido |
|---|---|---|
| `01_EDA_y_seleccion.ipynb` | ¿Qué datos tenemos y qué variables sirven? | Diccionario de variables, EDA estructural, split agrupado por paciente, EDA analítico, faltantes y outliers, variables nuevas, selección de las 18 variables y codificación |
| `02_Planteo_y_basal.ipynb` | ¿Qué problema resolvemos y cuál es la vara a superar? | M0 entorno y split, M1 basal, M2 definición formal, M3 estrategia de validación |
| `03_Modelos_y_ajuste.ipynb` | ¿Qué algoritmos probamos y cómo los ajustamos? | Medición de cómputo, M4 algoritmos candidatos, M5 ajuste de hiperparámetros, M6 umbral y calibración de las probabilidades |
| `04_Evaluacion_y_conclusiones.ipynb` | ¿Cuánto le ganan al basal, por qué y con qué límites? | M7 comparación, M8 interpretabilidad, errores y perfiles de pacientes (clustering), M9 evaluación en el test con impacto práctico y calibración, M10 limitaciones, respuesta a las preguntas de la investigación, aprendizajes y referencias |

## Basal

Regla de una sola variable, revisada a partir de la clase de consulta: si el paciente tuvo al menos una internación en el año previo, se predice que
reingresa. El corte se calcula con el train comparando la tasa de reingreso de cada nivel con la tasa global.
ROC-AUC 0,606 en validación cruzada y 0,603 en el test.

## Cómo correrlo

Los notebooks se probaron con Python 3.13 y las versiones de `requirements.txt`. Recomendamos un entorno nuevo,
solo para este trabajo.

**1. Crear el entorno e instalar las dependencias** (en una terminal, desde la carpeta `TP FINAL`):

```
conda create -n tp_ml python=3.13 -y
conda activate tp_ml
python -m pip install -r requirements.txt
```

En Mac, XGBoost necesita además la librería OpenMP:

```
conda install -c conda-forge llvm-openmp -y
```

Para verificar la instalación: `python -c "import sklearn, xgboost, catboost, optuna, dill"` no debe dar error.

**2. Abrir la carpeta correcta.** Los notebooks leen `diabetic_data.csv` y la carpeta `artefactos_ml/` con rutas
relativas, así que tienen que ejecutarse desde `TP FINAL`. En VS Code: *File → Open Folder* y elegir `TP FINAL`.

**3. Elegir el kernel `tp_ml` en los cuatro notebooks** (arriba a la derecha de cada notebook).

**4. Correr los notebooks en orden: 01, 02, 03 y 04**, cada uno completo (*Restart* y *Run All*). Al final de los
notebooks 1, 2 y 3 se guarda el estado de la sesión en `artefactos_ml/estado_0X.pkl` con la librería `dill`, y el
notebook siguiente lo carga al empezar. Esos archivos no se suben al repositorio por su tamaño: se generan al correr.

**5. Tiempos.** La búsqueda de hiperparámetros de M5 es la parte más costosa. Sus resultados están guardados en
`artefactos_ml/m5_carrera_semilla*.pkl` y se cargan en lugar de recalcularse. Aun así, el notebook 3 y el 4 tardan
unos 15 minutos cada uno.

**Problemas frecuentes**

| Error | Causa | Solución |
|---|---|---|
| `FileNotFoundError: diabetic_data.csv` | El notebook no se está ejecutando desde `TP FINAL` | Abrir la carpeta `TP FINAL` en VS Code (paso 2) |
| `No module named 'xgboost'` u otra librería | El kernel no es `tp_ml` | Cambiar el kernel (paso 3) |
| `XGBoost Library (libxgboost.dylib) could not be loaded` | Falta OpenMP en Mac | `conda install -c conda-forge llvm-openmp -y` |
| `FileNotFoundError: artefactos_ml/estado_0X.pkl` | No se corrió completo el notebook anterior | Correr el notebook anterior hasta el final |
| `No module named 'pyarrow'` al cargar el estado | El estado se generó en otro entorno | Borrar `artefactos_ml/estado_0*.pkl` y correr de nuevo desde el notebook 1 |

## Otros contenidos

- `artefactos_ml/`: split, folds, registro de experimentos, resultados y modelos finales.
- `dashboard_ml/`: tablero para explorar los resultados.
