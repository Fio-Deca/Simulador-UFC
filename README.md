# 🥊 Simulador Predictivo UFC & MMA Analytics

Bienvenido al repositorio del **Simulador Predictivo de la UFC**, una aplicación web interactiva desarrollada en Python orientada al análisis de datos deportivos y la predicción de combates de Artes Marciales Mixtas (MMA) mediante Inteligencia Artificial.

Este proyecto forma parte de mi Trabajo de Fin de Máster (TFM) y busca eliminar el sesgo humano en el análisis deportivo mediante la aplicación rigurosa de la Ciencia de Datos.

## ✨ Características Principales

* 📓 **Investigación y Modelado (Jupyter Notebook):** En la carpeta `/notebooks` se incluye el código fuente original de la investigación. Aquí se detalla todo el proceso de *Data Wrangling*, *Feature Engineering* (creación de variables diferenciales y Ape Index) y el entrenamiento de los algoritmos.
* 🎮 **Simulador de Combates Reactivo:** Motor de predicción impulsado por un modelo de **Regresión Logística**. Durante la fase de investigación, se realizó una comparativa exhaustiva con distintos algoritmos de *Machine Learning*, decantándose finalmente por la Regresión Logística debido a su alta fiabilidad y excelente explicabilidad para calcular probabilidades reales de victoria.
* 📊 **Análisis Exploratorio de Datos (EDA) Dinámico:** Visualización interactiva con Plotly que muestra las tasas de finalización (KO, Sumisión, Decisión) segmentadas por categoría de peso oficial de la UFC.
* 🧬 **Perfilado de Estilos de Pelea (PCA + K-Means):** Aplicación de aprendizaje no supervisado. El algoritmo K-Means, justificado matemáticamente mediante el Coeficiente de Silueta (k=2), separa a los atletas en *Strikers* y *Grapplers* basándose puramente en su volumen de golpes y derribos, reduciendo la dimensionalidad previamente mediante PCA.

## 🛠️ Tecnologías Utilizadas

* **Front-end & Deployment:** Streamlit
* **Manipulación de Datos:** Pandas, NumPy
* **Machine Learning & Clustering:** Scikit-Learn (`LogisticRegression`, `StandardScaler`, `PCA`, `KMeans`, `silhouette_score`)
* **Visualización:** Plotly Express, Matplotlib, Seaborn, Streamlit Native Charts

## 🚀 Cómo ejecutar el proyecto localmente

1. Clona este repositorio:
   ```bash
   git clone [https://github.com/Fio-Deca/Predicci-n-de-peleas-UFC](https://github.com/Fio-Deca/Predicci-n-de-peleas-UFC.git)
