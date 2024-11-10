from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import soundfile as sf
import numpy as np
import os
from typing import Optional
import torch
from diffusers import AudioLDMPipeline

app = FastAPI(
    title="Audio Generator API",
    description="API para generar audio a partir de texto usando AudioLDM",
    version="1.0.0"
)

# Inicializar el modelo
try:
    pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-l-full")
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")
    pipe = None

class TextPrompt(BaseModel):
    text: str
    steps: Optional[int] = 100

@app.post("/generate-audio/")
async def generate_audio(prompt: TextPrompt):
    """
    Genera un archivo de audio basado en el texto proporcionado
    """
    if pipe is None:
        raise HTTPException(status_code=500, detail="Modelo no inicializado correctamente")
    
    try:
        # Crear directorio temporal si no existe
        os.makedirs("temp", exist_ok=True)
        
        # Generar audio
        audio = pipe(
            prompt.text, 
            num_inference_steps=prompt.steps,
            audio_length_in_s=10.0
        ).audios[0]
        
        # Guardar el audio temporalmente
        output_path = f"temp/{hash(prompt.text)}.wav"
        sf.write(output_path, audio, samplerate=16000)
        
        # Devolver el archivo
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"{prompt.text[:30]}.wav"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar audio: {str(e)}")

@app.get("/")
async def root():
    """
    Endpoint de bienvenida
    """
    return {
        "message": "Bienvenido a la API de Audio Generator",
        "usage": "Realiza una petición POST a /generate-audio/ con un JSON que contenga el campo 'text'"
    } 