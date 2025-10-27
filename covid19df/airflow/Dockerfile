# Dockerfile para Airflow con Kedro integrado
FROM apache/airflow:2.8.3-python3.11

# Cambiar a usuario airflow antes de instalar
USER airflow

# Instalar Kedro y dependencias del proyecto
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir kedro
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
