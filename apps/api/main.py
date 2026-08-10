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

from apps.tools import get_tool, list_tools
from apps.google_clients import get_calendar_service

# Import tool modules to register them
import apps.tools.calendar_tools
import apps.tools.meetings_tools

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


@app.post("/internal/tools/{tool_name}")
async def execute_tool(
    tool_name: str,
    parameters: dict,
    x_internal_api_key: str | None = Header(default=None),
):
    """
    Generic tool execution endpoint.
    Executes any registered tool by name with provided parameters.
    """
    err = require_internal_api_key(x_internal_api_key)
    if err:
        return err

    tool = get_tool(tool_name)
    if not tool:
        return JSONResponse({"error": f"Tool '{tool_name}' not found"}, status_code=404)

    try:
        result = await tool.ainvoke(parameters)
        return {"result": result, "success": True}
    except Exception as exc:
        logger.error(f"Tool execution error for {tool_name}: {exc}", exc_info=True)
        return JSONResponse({"error": str(exc), "success": False}, status_code=500)


@app.get("/internal/tools")
def list_available_tools(x_internal_api_key: str | None = Header(default=None)):
    """List all available tools."""
    err = require_internal_api_key(x_internal_api_key)
    if err:
        return err

    tools = list_tools()
    return {"tools": tools}






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


