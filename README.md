# covid19DF

Sistema completo de **Machine Learning** con arquitectura **MLOps** para análisis de datos de COVID-19. Implementa pipelines automatizados de **regresión**, **clasificación** y **agrupamiento**, con optimización de hiperparámetros, orquestación de pipelines y versionamiento de datos.

---

## 📋 Tabla de Contenidos

- [Características](#características)
- [Arquitectura del proyecto](#arquitectura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Estructura de carpetas](#estructura-de-carpetas)
- [Pipelines](#pipelines)
- [Uso](#uso)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

---

## 🌟 Características

- Automatización de pipelines de Machine Learning con **Kedro**.
- Pipelines para:
  - **Regresión**: predicción de variables continuas relacionadas con COVID-19.
  - **Clasificación**: detección de patrones y categorización de datos.
  - **Agrupamiento (Clustering)**: análisis de segmentación de datos.
- Optimización de hiperparámetros con técnicas avanzadas.
- Orquestación con Apache Airflow (opcional).
- Versionamiento de datasets y modelos con **DVC**.
- Contenerización con **Docker** para despliegue reproducible.

---

## 🏗 Arquitectura del proyecto

El proyecto sigue una estructura basada en **MLOps y pipelines modulares**:

DVC_Local_Repo #Configuracion del versionado DVC
│
covid19df/

│

  ├─ airflow/

   │ ├─ dags # Ubicacion de ambos dags de airflow

  ├─ data/ # Datasets (raw, processed, etc.)

  ├─ notebooks/ # Notebooks de análisis exploratorio

  ├─ src/ # Código fuente de pipelines y nodos

  │ ├─ pipelines/

  │ ├─ nodes/

  │ └─ utils/
  
  ├─ conf/ # Configuraciones de Kedro y DVC

  ├─ logs/ # Logs de ejecución

  └─ README.md


---

## 🛠 Requisitos

- Python >= 3.10
- Kedro >= 0.19
- Pandas, NumPy, scikit-learn, matplotlib, seaborn
- DVC >= 2.0 (opcional, para versionamiento)
- Docker (opcional, para contenerización)
- Apache Airflow (opcional, para orquestación)

---

## ⚡ Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/HansIgnaci0/covid19DF_Ev02.git
cd covid19DF
````
2.-Crear y activar entorno virtual:
```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
````
3.- Instalar las dependencias
```bash
pip install -r requirements.txt
````
4.- Inicializar DVC
```bash
dvc init
dvc pull
````

Regresión

Predice variables continuas relacionadas con la evolución del COVID-19.

Clasificación

Clasifica registros según criterios definidos en el dataset.

Agrupamiento

Agrupa datos para identificar patrones y clusters relevantes.

Ejecutar un pipeline:

```bash
kedro run --pipeline <clasificacion>
````
🚀 Uso

Ejecuta pipelines completos con:
```bash
kedro run
````

📊 Resultados y Conclusiones

La arquitectura modular permite ejecutar, depurar y escalar cada pipeline de manera independiente.

Los pipelines muestran que Kedro + DVC es muy útil para reproducibilidad y control de versiones de datos y modelos.

Gracias a la separación entre regresión, clasificación y clustering, se facilita la comparación de técnicas y algoritmos sobre el mismo dataset.

Este proyecto sirve como base para proyectos MLOps completos, donde los pipelines pueden integrarse con Airflow para orquestación y Docker para despliegue.

💡 Buenas prácticas

Mantener los datos crudos en data/raw/ y procesados en data/processed/.

Documentar cambios en pipelines y nodos para facilitar colaboraciones.

Usar .gitkeep en carpetas vacías si es necesario mantener la estructura.

Versionar modelos y datasets con DVC para asegurar reproducibilidad.

🤝 Contribuciones

Fork del repositorio.

Crear rama feature: git checkout -b feature/nueva-funcionalidad.

Commit y push:
```bash
git commit -am "Agrego nueva funcionalidad"
git push origin feature/nueva-funcionalidad
````

Desarollador del proyecto: Hans Ignacio Mancilla Sandoval

Contacto: ha.mancilla@duocuc.cl

Asignatura: Machine Learning

Profesor: Giocrisrai Godoy
