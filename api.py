from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
from PIL import Image
import torch
from transformers import pipeline
import time

app = FastAPI(
    title="OCR API",
    version="1.0",
    description="API для распознавания англоязычного текста с изображений"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr_pipeline = None
model_loaded = False

def load_model():
    """Загрузка OCR модели"""
    global ocr_pipeline, model_loaded
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Загрузка модели на устройство: {device}")
        
        try:
            ocr_pipeline = pipeline(
                "image-to-text",
                model="microsoft/trocr-base-printed",
                device=device
            )
            print("Основная модель загружена")
        except:
            print("Загружаем резервную модель...")
            ocr_pipeline = pipeline(
                "image-to-text",
                model="microsoft/trocr-base-stage1",
                device=device
            )
            print("Резервная модель загружена")
        
        model_loaded = True
        print(f"Модель загружена на: {device}")
        
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        model_loaded = False

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {
        "app": "OCR API",
        "version": "1.0",
        "description": "API для распознавания текста с изображений",
        "endpoints": {
            "health": "/health",
            "ocr": "/ocr",
            "docs": "/docs"
        },
        "model_loaded": model_loaded
    }

@app.post("/ocr")
async def recognize_text(
    file: UploadFile = File(..., description="Изображение для распознавания (JPG, PNG)")
):
    allowed_types = ["jpg", "jpeg", "png"]
    file_extension = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    
    if file_extension not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат. Разрешены: {', '.join(allowed_types)}"
        )
    
    if not model_loaded or ocr_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Попробуйте позже."
        )
    
    try:
        contents = await file.read()
        
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Файл слишком большой. Максимум: 10MB"
            )
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        start_time = time.time()
        result = ocr_pipeline(image)
        processing_time = time.time() - start_time
        
        text = result[0]['generated_text'] if result else ""
        
        return {
            "status": "success",
            "text": text,
            "processing_time": round(processing_time, 2),
            "filename": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки: {str(e)}"
        )

@app.post("/ocr/url")
async def recognize_from_url(image_url: str):
    import requests
    
    if not model_loaded or ocr_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Попробуйте позже."
        )
    
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        start_time = time.time()
        result = ocr_pipeline(image)
        processing_time = time.time() - start_time
        
        text = result[0]['generated_text'] if result else ""
        
        return {
            "status": "success",
            "text": text,
            "processing_time": round(processing_time, 2),
            "image_url": image_url
        }
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка загрузки изображения: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки: {str(e)}"
        )

if __name__ == "__main__":
    # Запуск сервера
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )