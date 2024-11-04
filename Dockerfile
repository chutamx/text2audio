# Usar una imagen base de NVIDIA con soporte completo
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Crear un enlace simbólico para python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Actualizar pip
RUN python3 -m pip install --upgrade pip

# Instalar PyTorch con soporte CUDA
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copiar los archivos de requerimientos
COPY requirements.txt .

# Instalar el resto de dependencias en grupos
RUN pip install --no-cache-dir fastapi==0.104.1 uvicorn==0.24.0 python-multipart==0.0.6
RUN pip install --no-cache-dir soundfile==0.12.1 numpy==1.24.3
RUN pip install --no-cache-dir transformers>=4.31.0 accelerate>=0.21.0
RUN pip install --no-cache-dir diffusers>=0.21.4 einops>=0.6.1
RUN pip install --no-cache-dir librosa>=0.10.1 scipy>=1.11.2

# Copiar el código de la aplicación
COPY ./app ./app

# Crear directorio temporal para los archivos de audio
RUN mkdir -p temp

# Exponer el puerto que usa FastAPI
EXPOSE 8000

# Comando para ejecutar la aplicación con hot-reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 