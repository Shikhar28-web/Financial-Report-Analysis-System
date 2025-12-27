import os
import uuid
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from fastapi.openapi.docs import get_swagger_ui_html

from retrieval.ingestion.pdf_loader import ingest_user_file
from retrieval.ingestion.answer_user_query import answer_user_query


# -------------------------------------------
# TAGS
# -------------------------------------------
tags_metadata = [
    {"name": "📄 File Upload", "description": "Upload PDFs/CSVs."},
    {"name": "🤖 AI Query", "description": "Ask questions about financial reports."},
    {"name": "🩺 System", "description": "System health monitoring."},
]


# -------------------------------------------
# FASTAPI APP
# -------------------------------------------
app = FastAPI(
    title="Financial Report Analysis System",
    version="1.0.0",
    
    openapi_tags=tags_metadata,
    docs_url=None,
    redoc_url=None
)


# -------------------------------------------
# LIGHT THEME + BOLD + CLEAN UI
# -------------------------------------------
@app.get("/docs", include_in_schema=False)
async def custom_docs():
    html = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Financial Report Analysis System",
    ).body.decode()

    css = """
    <style>
        body { 
            background:#fafafa !important;
            margin:0;
            font-family:'Segoe UI', sans-serif;
            animation:fadeIn .4s ease-in;
        }

        @keyframes fadeIn {
            from {opacity:0; transform:translateY(8px);}
            to {opacity:1; transform:translateY(0);}
        }

        /* Header */
        .topbar { 
            background:#ffffff !important; 
            border-bottom:1px solid #e2e2e2;
        }
        .topbar-wrapper .link span { 
            color:#000 !important; 
            font-size:22px;
            font-weight:700; 
        }

        /* Tags */
        .opblock-tag {
            background:#ffffff !important;
            border-radius:10px !important;
            border:1px solid #ddd !important;
            transition:0.2s ease;
            font-weight:700 !important;
        }
        .opblock-tag:hover {
            background:#f0f0f0 !important;
            transform:scale(1.01);
        }

        /* Endpoint Blocks */
        .opblock {
            background:#ffffff !important;
            border-radius:12px !important;
            border:1px solid #e0e0e0 !important;
            transition:0.25s ease;
        }
        .opblock:hover {
            background:#f7f7f7 !important;
            transform:scale(1.005);
        }

        /* Method Colors */
        .opblock-summary-method { 
            border-radius:6px !important; 
            font-weight:700 !important;
        }
        .opblock.opblock-post .opblock-summary-method {
            background:#007bff !important;
        }
        .opblock.opblock-get .opblock-summary-method {
            background:#28a745 !important;
        }

        /* Input boxes */
        input, textarea {
            background:#ffffff !important;
            color:#000 !important;
            border:1px solid #ccc !important;
            border-radius:6px !important;
        }

        /* Buttons */
        .btn {
            border-radius:6px !important;
            transition:0.2s ease !important;
            font-weight:600 !important;
        }
        .btn:hover {
            transform:scale(1.05);
            background:#e6e6e6 !important;
        }

        /* Title section text */
        h2, h3, h4, h5, h6, .markdown p, .markdown strong {
            color:#000 !important;
            font-weight:700 !important;
        }

        /* Schemas box */
        .model-box { 
            background:#ffffff !important; 
            border:1px solid #ddd !important; 
        }
    </style>
    """

    return HTMLResponse(html + css)


# -------------------------------------------
# CORS
# -------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------
# FILE STORAGE
# -------------------------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -------------------------------------------
# 1️⃣ Upload
# -------------------------------------------
@app.post("/upload", tags=["📄 File Upload"])
async def upload_file(file: UploadFile = File(...)):
    user_id = str(uuid.uuid4())
    file_path = f"{UPLOAD_DIR}/{user_id}_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = ingest_user_file(file_path, user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "uploaded",
        "user_id": user_id,
        "chunks": result["chunks"],
        "collection": result["collection"]
    }


# -------------------------------------------
# 2️⃣ Ask Question
# -------------------------------------------
class QueryRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = 8


@app.post("/ask", tags=["🤖 AI Query"])
async def ask_question(req: QueryRequest):
    try:
        return {"answer": answer_user_query(req.query, req.user_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------
# 3️⃣ Health Check
# -------------------------------------------
@app.get("/health", tags=["🩺 System"])
async def health():
    return {"status": "ok"}
