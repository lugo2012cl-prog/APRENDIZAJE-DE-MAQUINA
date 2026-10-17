# Dashboard — TP Aprendizaje de Máquina (Diabetes 130-US Hospitals)

Dashboard en Streamlit que muestra, desde la sección **7.5 Balance de clases** del notebook en adelante, cada etapa
del modelado (**M0 a M10**) con su botón. Lee directamente los cuatro notebooks de `TP FINAL`, en orden (textos,
tablas y gráficos con sus lecturas), y `../artefactos_ml/` (resultados guardados), por lo que debe quedar en la carpeta
`dashboard_ml/` dentro de `TP FINAL`. Muestra lo que quedó guardado en los notebooks, así que conviene abrirlo después
de haberlos corrido.

## Cómo correrlo

Desde esta carpeta, con el entorno `tp_ml` activado:

```bash
python -m pip install -r requirements.txt
streamlit run app_ml.py
```

## Botones

EDA · Enganche · Basal · Definición · Validación · Algoritmos · Hiperparámetros · Umbral · Comparación ·
Interpretabilidad · Test final · Cierre.

Cada botón muestra esa etapa tal como figura en el notebook, con tablas con formato propio (sin código). Hay resumen
gráfico en EDA y Enganche, y complementos interactivos en Validación (estabilidad de la selección), Hiperparámetros
(convergencia), Comparación (boxplots de los 20 folds) y Test final (métricas con intervalo de confianza y matriz de
confusión con interruptor de porcentaje).

`precomputar_modelado_ml.py`, `data_utils.py` y `model_artifacts_ml.pkl` pertenecen a la versión anterior del dashboard y
ya no los usa `app_ml.py`.
