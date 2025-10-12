from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path

app = FastAPI(
    title="Manifold Backend Server",
    description="Backend für XAI SNN Manifold Learning",
    version="1.0.0"
)

# CORS-Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion einschränken!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload-Verzeichnis erstellen
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Willkommen zur Manifold API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Datei hochladen und speichern"""
    try:
        # Datei speichern
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = file_path.stat().st_size
        
        return {
            "filename": file.filename,
            "status": "success",
            "size": file_size,
            "content_type": file.content_type,
            "path": str(file_path)
        }
    except Exception as e:
        return {
            "filename": file.filename,
            "status": "error",
            "error": str(e)
        }
    finally:
        await file.close()