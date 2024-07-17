# Análisis de Clústeres de Tarjetas de Crédito

Este proyecto tiene como objetivo realizar un análisis de clústeres en un conjunto de datos de tarjetas de crédito, permitiendo a los analistas y responsables de la toma de decisiones identificar segmentos de clientes y formular estrategias de marketing personalizadas.

## Descripción del Proyecto

La herramienta interactiva desarrollada con Streamlit facilita la carga de datos, preprocesamiento, aplicación de PCA y K-Means, visualización de resultados y obtención de recomendaciones basadas en los clústeres identificados.

### Funcionalidades

1. **Carga y Preprocesamiento de Datos**
    - Carga de archivos CSV con datos de tarjetas de crédito.
    - Visualización de información general y estructura de los datos.
    - Manejo de valores nulos y atípicos.
    - Normalización de datos y creación de nuevas características.

2. **Aplicación de PCA y K-Means**
    - Aplicación de PCA para reducir la dimensionalidad.
    - Determinación del número óptimo de clústeres utilizando el método del codo y el análisis de silueta.
    - Aplicación de K-Means y asignación de clústeres a los datos.

3. **Visualización de Resultados**
    - Visualización de la varianza explicada por PCA.
    - Visualización de clústeres en gráficos 2D y 3D.
    - Perfilado y caracterización de clústeres.

4. **Guardado y Carga de Modelos**
    - Guardado de modelos PCA y K-Means.
    - Carga de modelos guardados para su uso en análisis futuros.

### Requisitos

- Python 3.7 o superior
- Streamlit
- Pandas
- Numpy
- Scikit-learn
- Plotly
- Joblib

### Instalación

```bash
poetry add streamlit pandas numpy scikit-learn plotly joblib
```

### Ejecución
git clone https://github.com/tu-usuario/analisis-cluster-tarjetas-credito.git
cd src
streamlit run app.py
```

Estructura del Proyecto
app.py: Archivo principal de la aplicación Streamlit.
data/: Directorio que contiene los datos de entrada.
models/: Directorio que contiene los modelos guardados.
requirements.txt: Archivo de requisitos para la instalación de dependencias.
Detalles de la Interfaz
Visión General: Visualización de estadísticas generales y distribución de clústeres.
Recomendador de Estrategias: Recomendaciones de estrategias de marketing basadas en el clúster seleccionado.
Comparador de Clientes: Comparación de características de clientes seleccionados.
Predictor de Clúster: Predicción del clúster para un nuevo cliente basado en sus características.
Análisis Detallado: Análisis detallado de la distribución de variables, matriz de correlación y análisis de componentes principales.
Desarrollo
Preprocesamiento de Datos

Manejo de valores nulos utilizando la mediana.
Normalización de datos numéricos.
Creación de nuevas características.
Aplicación de PCA

Reducción de dimensionalidad a tres componentes principales.
Visualización de los dos primeros componentes principales.
Determinación del Número Óptimo de Clústeres

Método del codo y análisis de silueta para determinar el número óptimo de clústeres.
Aplicación de K-Means

Aplicación de K-Means y asignación de clústeres a los datos.
Visualización de clústeres en gráficos de dispersión en 2D y 3D.
Guardado de Modelos

Guardado de modelos de PCA y K-Means.
Guardado de datos con los clústeres asignados en un archivo CSV.
Contribuciones
Las contribuciones son bienvenidas. Si deseas contribuir, por favor realiza un fork del repositorio y envía un pull request con tus mejoras.

Licencia
Este proyecto está licenciado bajo los términos de la licencia MIT.

