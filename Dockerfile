FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip
RUN pip install --upgrade pip

# Instalar PyTorch con soporte CUDA específico para V100
RUN pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

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

# Usar python3 directamente de la imagen base
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]