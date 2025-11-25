# 1️⃣ Imagen base: Python oficial
FROM python:3.11-slim

# 2️⃣ Establecer directorio de trabajo dentro del contenedor
WORKDIR /app

# 3️⃣ Copiar los archivos de tu proyecto al contenedor
COPY . /app

# 4️⃣ Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 5️⃣ Instalar dependencias de Python
# Usa requirements.txt si lo tienes actualizado
RUN pip install --upgrade pip
RUN pip install -r src/requirements.txt || pip install -r requirements.txt

# 6️⃣ Establecer variable de entorno para evitar buffering de logs
ENV PYTHONUNBUFFERED=1

# 7️⃣ Comando por defecto: ejecutar el pipeline completo
CMD ["kedro", "run"]
# askdasmdksa