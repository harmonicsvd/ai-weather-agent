from __future__ import annotations
from contextlib import asynccontextmanager, contextmanager
import logging

"""Sham internal API: weather intelligence, document ingestion, and LangGraph workflows."""

import hmac
import time
import os
from datetime import datetime, timedelta, timezone, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI, Header, Query, UploadFile, File, Form, Request, logger
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from uuid import uuid4
from io import BytesIO
from pypdf import PdfReader
import psycopg

from apps.rag.vector_store import init_vector_store, save_chunk_vectors

from apps.graph.workflows import build_meeting_preview_graph

from apps.rag.ingestion import chunk_uploaded_markdown

from apps.rag.embeddings import GeminiEmbeddingProvider

from apps.skills import get_skill, list_skills
from apps.google_clients import get_calendar_service

# Import skill modules to register them
import apps.skills.calendar_skills
import apps.skills.meetings_skills

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)

WEATHER_INTERNAL_API_KEY = os.getenv("WEATHER_INTERNAL_API_KEY", "")
MEETING_PREVIEW_APP = build_meeting_preview_graph(checkpointer=None)

logger = logging.getLogger("uvicorn.error")
VECTOR_STORE_AVAILABLE = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Sham vector storage."""
    global VECTOR_STORE_AVAILABLE
    try:
        init_vector_store()
        VECTOR_STORE_AVAILABLE = True
        logger.info("vector_store_init_done")
    except Exception as exc:
        VECTOR_STORE_AVAILABLE = False
        logger.warning("vector_store_init_failed_continuing_with_file_fallback error=%r", exc)

    yield

app = FastAPI(title="Sham Weather Intelligence Internal API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def require_internal_api_key(x_internal_api_key: str | None):
    """
    Validate backend-to-backend auth header for internal weather endpoints.

    Returns:
    - JSONResponse error object when key is missing/invalid
    - None when key is valid
    """
    # Internal endpoint guard: backend-to-backend only.
    if not WEATHER_INTERNAL_API_KEY:
        return JSONResponse({"error": "WEATHER_INTERNAL_API_KEY is not configured"}, status_code=500)
    if not x_internal_api_key or not hmac.compare_digest(x_internal_api_key, WEATHER_INTERNAL_API_KEY):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return None


@app.get("/health")
def health():
    """Minimal liveness probe for orchestration platforms and local checks."""
    return {"ok": True}


@app.post("/internal/knowledge/upload")
async def internal_knowledge_upload(
    user_sub: str = Form(...),
    file: UploadFile = File(...),
    x_internal_api_key: str | None = Header(default=None),
):
    """Receive an authenticated Ram user PDF for Sham RAG ingestion."""
    err = require_internal_api_key(x_internal_api_key)
    if err:
        return err

    if file.content_type != "application/pdf":
        return JSONResponse(
            {"error": "Only PDF files are supported"},
            status_code=400,
        )

    file_bytes = await file.read()
    
    if not file_bytes:
        return JSONResponse(
            {"error": "Uploaded file is empty"},
            status_code=400,
        )
        
    original_name = Path(file.filename or "upload.pdf").name
    document_id = uuid4().hex
    stored_name = f"{document_id}-{original_name}"

    upload_dir = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "knowledge_uploads"
        / user_sub
        / "originals"
    )
    upload_dir.mkdir(parents=True, exist_ok=True)

    stored_path = upload_dir / stored_name
    stored_path.write_bytes(file_bytes)

    logger.info(
        "weather_knowledge_pdf_saved document_id=%s user_sub=%s stored_path=%s size_bytes=%s",
        document_id,
        user_sub,
        stored_path,
        len(file_bytes),
    )

    markdown_text = extract_pdf_to_markdown(file_bytes, original_name)

    markdown_dir = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "knowledge_uploads"
        / user_sub
        / "markdown"
    )
    markdown_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = markdown_dir / f"{document_id}.md"
    markdown_path.write_text(markdown_text, encoding="utf-8")
    
    chunks = chunk_uploaded_markdown(
        user_sub=user_sub,
        markdown_path=markdown_path,
        original_filename=original_name,
    )

    logger.info(
    "weather_knowledge_chunks_created document_id=%s chunks=%s",
    document_id,
    len(chunks),
)   
    embedding_provider = GeminiEmbeddingProvider()
    vectors = embedding_provider.embed_documents(
        [chunk.text for chunk in chunks]
    )
    
    if len(vectors) != len(chunks):
        return JSONResponse(
            {"error": "Embedding count does not match chunk count"},
            status_code=500,
        )
        
    saved_vector_count = 0
    vector_store_available = VECTOR_STORE_AVAILABLE
    try:
        saved_vector_count = save_chunk_vectors(
            document_id=document_id,
            user_sub=user_sub,
            chunks=chunks,
            vectors=vectors,
        )
        vector_store_available = True
    except Exception as exc:
        vector_store_available = False
        logger.warning(
            "weather_knowledge_vector_save_failed_continuing_with_file_fallback document_id=%s error=%r",
            document_id,
            exc,
        )

    logger.info(
            "weather_knowledge_markdown_saved document_id=%s chars=%s markdown_path=%s",
            document_id,
            len(markdown_text),
            markdown_path,
        )

    logger.info(
        "weather_knowledge_pdf_received user_sub=%s filename=%s size_bytes=%s",
        user_sub,
        file.filename,
        len(file_bytes),
    )

    return {
        "ok": True,
        "document_id": document_id,
        "user_sub": user_sub,
        "filename": original_name,
        "stored_name": stored_name,
        "size_bytes": len(file_bytes),
        "markdown_path": str(markdown_path),
        "markdown_chars": len(markdown_text),
        "vector_count": len(vectors),
        "saved_vector_count": saved_vector_count,
        "vector_store_available": vector_store_available,
    }


def extract_pdf_to_markdown(file_bytes: bytes, original_name: str) -> str:
    """Extract readable PDF text into Markdown sections."""
    reader = PdfReader(BytesIO(file_bytes))
    sections = [f"# {original_name}"]

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(f"## Page {page_number}\n\n{text}")

    return "\n\n".join(sections).strip()


@app.post("/internal/skills/{skill_name}")
async def execute_skill(
    skill_name: str,
    request: Request,
    x_internal_api_key: str | None = Header(default=None),
):
    """
    Generic skill execution endpoint.
    Executes any registered skill by name with provided parameters.
    Now checks if user has the skill installed.
    Accepts JSON body with 'parameters' and 'user_sub' fields.
    """
    err = require_internal_api_key(x_internal_api_key)
    if err:
        return err

    try:
        # Parse JSON body
        body = await request.json()
        parameters = body.get("parameters", {})
        user_sub = body.get("user_sub", "")
        
        if not user_sub:
            return JSONResponse({"error": "user_sub is required"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Invalid JSON body: {str(e)}"}, status_code=400)

    # Check if user has this skill installed

    INTERNAL_SKILLS = []  # No internal skills - all require installation
    if skill_name not in INTERNAL_SKILLS:
        from apps.skills import get_user_skills
        user_skills = get_user_skills(user_sub)
        if skill_name not in user_skills:
            return JSONResponse(
                {"error": f"Skill '{skill_name}' not installed for user"}, 
                status_code=403
            )

    skill = get_skill(skill_name)
    if not skill:
        return JSONResponse({"error": f"Skill '{skill_name}' not found"}, status_code=404)

    try:
        # Add user_sub to parameters for skills that need it
        parameters["user_sub"] = user_sub
        result = await skill.ainvoke(parameters)
        return {"result": result, "success": True}
    except Exception as exc:
        logger.error(f"Skill execution error for {skill_name}: {exc}", exc_info=True)
        return JSONResponse({"error": str(exc), "success": False}, status_code=500)


@app.get("/internal/skills")
def list_available_skills(
    user_sub: str = Query(...),
    x_internal_api_key: str | None = Header(default=None)
):
    """List skills available to a specific user."""
    err = require_internal_api_key(x_internal_api_key)
    if err:
        return err

    from apps.skills import get_user_skills
    skills = get_user_skills(user_sub)
    return {"skills": skills}


@app.post("/internal/skills/cache/clear")
def clear_skills_cache(
    user_sub: str = Form(...),
    x_internal_api_key: str | None = Header(default=None)
):
    """Clear skills cache for a user (call after install/uninstall)."""
    err = require_internal_api_key(x_internal_api_key)
    if err:
        return err
    
    from apps.skills import clear_user_skills_cache
    clear_user_skills_cache(user_sub)
    return {"ok": True}






DATABASE_URL = os.getenv("DATABASE_URL", "")

@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = psycopg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()

def db_execute(conn, query, params=None):
    """Execute a SQL query and return cursor."""
    cursor = conn.cursor(row_factory=psycopg.rows.dict_row)
    cursor.execute(query, params or ())
    return cursor


