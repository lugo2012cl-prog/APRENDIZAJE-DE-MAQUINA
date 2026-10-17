# Predicción de reingreso hospitalario temprano en pacientes diabéticos

Trabajo Final de Aprendizaje de Máquina. Carrera de Especialización en Inteligencia Artificial (CEIA), FIUBA,
cohorte 27Co2026.

| Integrantes | Docentes |
|---|---|
| Florencia Lucca | María Carina Roldán |
| Mariano Reimondez | Facundo Lucianna |
| Carlos Lugones | |

## De qué se trata

Queremos saber si, con la información disponible al momento del alta, es posible anticipar qué pacientes
diabéticos van a volver a internarse dentro de los 30 días. Un reingreso temprano suele indicar que algo del
tratamiento o del seguimiento no funcionó, y si se identifica a tiempo a los pacientes de mayor riesgo, se puede
priorizar su seguimiento después del alta.

Trabajamos con el dataset *Diabetes 130-US Hospitals for Years 1999-2008* (unas 100.000 internaciones de 130
hospitales de Estados Unidos) y nos propusimos responder tres preguntas:

1. ¿Qué información del paciente y de la internación se asocia más con el reingreso temprano?
2. ¿Alcanza una regla simple basada en el historial del paciente, o hace falta un modelo que combine varias variables?
3. Si un modelo mejora a la regla simple, ¿cuánto mejora, qué aprendió y dónde sigue equivocándose?

## Cómo leer el trabajo

Todo está en la carpeta [`TP FINAL`](TP%20FINAL/). El trabajo se divide en cuatro notebooks que se leen en orden;
cada uno continúa donde terminó el anterior.

| Notebook | Pregunta que responde | Contenido |
|---|---|---|
| [01_EDA_y_seleccion](TP%20FINAL/01_EDA_y_seleccion.ipynb) | ¿Qué datos tenemos y qué variables sirven? | Análisis exploratorio, split agrupado por paciente, faltantes y outliers, variables nuevas y selección de las 18 variables |
| [02_Planteo_y_basal](TP%20FINAL/02_Planteo_y_basal.ipynb) | ¿Qué problema resolvemos y cuál es la vara a superar? | Basal, definición formal del problema y estrategia de validación |
| [03_Modelos_y_ajuste](TP%20FINAL/03_Modelos_y_ajuste.ipynb) | ¿Qué algoritmos probamos y cómo los ajustamos? | Cuatro algoritmos, búsqueda de hiperparámetros con tres mecanismos, umbral y calibración |
| [04_Evaluacion_y_conclusiones](TP%20FINAL/04_Evaluacion_y_conclusiones.ipynb) | ¿Cuánto le ganan al basal, por qué y con qué límites? | Comparación, interpretabilidad, perfiles de pacientes, evaluación en el test, limitaciones y respuesta a las tres preguntas |

Los notebooks están guardados con sus salidas, así que se pueden leer directamente en GitHub sin correrlos.

## Resultados principales

El basal es una regla de una sola variable: si el paciente tuvo al menos una internación en el año previo, se
predice que reingresa. La comparamos con cuatro algoritmos ajustados con validación cruzada agrupada por paciente.
El test se usó una sola vez, al final.

| Modelo | ROC-AUC en el test (IC 95 %) | Recall | Precisión |
|---|---|---|---|
| Basal (regla de internaciones previas) | 0,603 [0,592; 0,613] | 0,488 | 0,162 |
| Regresión logística | 0,665 [0,654; 0,676] | 0,427 | 0,201 |
| Random Forest (modelo elegido) | 0,677 [0,666; 0,687] | 0,405 | 0,214 |
| XGBoost | 0,679 [0,668; 0,689] | 0,445 | 0,205 |
| CatBoost | 0,680 [0,669; 0,690] | 0,425 | 0,210 |

En pocas palabras:

1. El historial de internaciones del paciente y el tipo de alta son, por lejos, lo que más se asocia con el
   reingreso. Lo muestran el análisis exploratorio, la importancia de variables de tres modelos distintos y un
   análisis de perfiles con K-means.
2. La regla simple captura buena parte de la señal, pero los cuatro modelos la superan en todos los folds y en el
   test, con mejoras de ROC-AUC de 0,063 a 0,077 cuyos intervalos de confianza no incluyen el cero.
3. Con la misma carga de trabajo que la regla (seguir a 328 pacientes de cada 1.000 altas), elegir a quién seguir
   con el modelo detecta 61 reingresos en lugar de 53. Las probabilidades del modelo están bien calibradas, de modo
   que pueden comunicarse como riesgos. La mejora es real pero modesta: los cuatro algoritmos llegan al mismo techo,
   lo que sugiere que el límite lo ponen los datos disponibles y no el algoritmo.

El detalle, las limitaciones y los caminos que dejamos abiertos están al final del notebook 4.

## Cómo reproducirlo

Usamos Python 3.13 y las versiones fijadas en [`requirements.txt`](TP%20FINAL/requirements.txt). En resumen:

```
conda create -n tp_ml python=3.13 -y
conda activate tp_ml
cd "TP FINAL"
python -m pip install -r requirements.txt
```

Después se corren los cuatro notebooks en orden, desde la carpeta `TP FINAL` y con el kernel `tp_ml`. Los pasos
completos, los tiempos y la solución a los errores más comunes (por ejemplo, la librería OpenMP que XGBoost
necesita en Mac) están en el [README de `TP FINAL`](TP%20FINAL/README.md).

También incluimos un tablero en Streamlit que recorre las etapas del modelado con tablas y gráficos interactivos.
Muestra lo guardado en los notebooks y se abre desde [`TP FINAL/dashboard_ml`](TP%20FINAL/dashboard_ml/), donde
está explicado cómo correrlo.

## Estructura del repositorio

```
TP FINAL/
├── 01_EDA_y_seleccion.ipynb
├── 02_Planteo_y_basal.ipynb
├── 03_Modelos_y_ajuste.ipynb
├── 04_Evaluacion_y_conclusiones.ipynb
├── README.md            cómo correr los notebooks y problemas frecuentes
├── requirements.txt     versiones de las librerías
├── diabetic_data.csv    datos originales
├── artefactos_ml/       split, folds, búsquedas de hiperparámetros, modelos finales y resultados del test
└── dashboard_ml/        tablero en Streamlit
```

## Datos

*Diabetes 130-US Hospitals for Years 1999-2008*, UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Strack, B. et al. (2014). Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical
database patient records. *BioMed Research International*, 2014, 781670.
