from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from kokoro import KPipeline
import soundfile as sf
import torch
import io
import base64
import tempfile
import numpy as np
from datetime import datetime
from typing import Optional, List
import uvicorn

app = FastAPI(title="Kokoro TTS Service", version="1.0.0")

# Initialize Kokoro pipeline
pipeline = KPipeline(lang_code='a')

# Pydantic models for request/response
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "am_adam"
    return_format: Optional[str] = "base64"  # "base64" or "file"

class TTSResponse(BaseModel):
    audio_base64: Optional[str] = None
    sample_rate: int = 24000
    format: str = "wav"
    voice_used: str

class VoicesResponse(BaseModel):
    voices: List[str]

class HealthResponse(BaseModel):
    status: str
    service: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", service="Kokoro TTS")

@app.get("/voices", response_model=VoicesResponse)
async def get_available_voices():
    """Get list of available voices"""
    voices = [
        "af_alloy", "af_aoede", "af_bella", "af_heart", 
        "af_jessica", "af_kore", "af_nicole", "af_nova", 
        "af_river", "af_sarah", "af_sky", "am_adam", 
        "am_echo", "am_eric", "am_fenrir", "am_liam", 
        "am_michael", "am_onyx", "am_puck", "am_santa"
    ]
    return VoicesResponse(voices=voices)

@app.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text
    """
    try:
        # Generate audio using Kokoro
        generator = pipeline(request.text, voice=request.voice)
        
        # Collect all audio chunks
        audio_chunks = []
        for i, (gs, ps, audio) in enumerate(generator):
            audio_chunks.append(audio)
        
        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        # Combine audio chunks
        combined_audio = np.concatenate(audio_chunks)
        
        if request.return_format == 'base64':
            # Convert to base64
            buffer = io.BytesIO()
            sf.write(buffer, combined_audio, 24000, format='WAV')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            return TTSResponse(
                audio_base64=audio_base64,
                sample_rate=24000,
                format="wav",
                voice_used=request.voice
            )
        
        elif request.return_format == 'file':
            # Save to temporary file and return file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, combined_audio, 24000)
            temp_file.close()
            
            return FileResponse(
                temp_file.name,
                media_type='audio/wav',
                filename=f'tts_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'
            )
        
        else:
            raise HTTPException(status_code=400, detail="Invalid return_format. Use 'base64' or 'file'")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize_file")
async def synthesize_speech_file(request: TTSRequest):
    """
    Synthesize speech and return as file download
    """
    try:
        # Generate audio using Kokoro
        generator = pipeline(request.text, voice=request.voice)
        
        # Collect all audio chunks
        audio_chunks = []
        for i, (gs, ps, audio) in enumerate(generator):
            audio_chunks.append(audio)
        
        if not audio_chunks:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        # Combine audio chunks
        combined_audio = np.concatenate(audio_chunks)
        
        # Save to temporary file and return file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, combined_audio, 24000)
        temp_file.close()
        
        return FileResponse(
            temp_file.name,
            media_type='audio/wav',
            filename=f'tts_{request.voice}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize_streaming")
async def synthesize_speech_streaming(request: TTSRequest):
    """
    Synthesize speech with streaming response
    """
    try:
        def generate_audio_stream():
            generator = pipeline(request.text, voice=request.voice)
            for i, (gs, ps, audio) in enumerate(generator):
                # Convert chunk to base64
                buffer = io.BytesIO()
                sf.write(buffer, audio, 24000, format='WAV')
                buffer.seek(0)
                chunk_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                
                yield f"data: {chunk_base64}\n\n"
        
        return StreamingResponse(
            generate_audio_stream(),
            media_type="text/plain"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8001)