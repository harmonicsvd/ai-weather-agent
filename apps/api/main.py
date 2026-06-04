from __future__ import annotations
from contextlib import asynccontextmanager
import logging

"""Sham internal API: weather intelligence, document ingestion, and LangGraph workflows."""

import hmac
import time
import os
from datetime import datetime, timedelta, timezone, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI, Header, Query, UploadFile, File, Form, logger
from fastapi.responses import JSONResponse

from uuid import uuid4
from io import BytesIO
from pypdf import PdfReader

from apps.rag.vector_store import init_vector_store, save_chunk_vectors

from apps.graph.workflows import build_meeting_preview_graph

from apps.rag.ingestion import chunk_uploaded_markdown

from apps.rag.embeddings import GeminiEmbeddingProvider



load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)

WEATHER_INTERNAL_API_KEY = os.getenv("WEATHER_INTERNAL_API_KEY", "")
MEETING_PREVIEW_APP = build_meeting_preview_graph(checkpointer=None)

logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Sham vector storage."""
    try:
        init_vector_store()
        logger.info("vector_store_init_done")
    except Exception as exc:
        logger.error("vector_store_init_failed error=%r", exc)
        raise

    yield

app = FastAPI(title="Sham Weather Intelligence Internal API", lifespan=lifespan)



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


def _to_utc_iso_z(dt: datetime) -> str:
    """Format timezone-aware datetime as canonical UTC `...Z` string."""
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _day_window_utc(target_date: str | None, timezone_name: str) -> tuple[str, str, str, str]:
    """
    Resolve a local day into a UTC `[from_iso, to_iso)` query window.

    This keeps "today" semantics correct for the user's timezone while querying
    calendar APIs that expect UTC timestamps.
    """
    try:
        tz = ZoneInfo(timezone_name)
        resolved_tz = timezone_name
    except Exception:
        tz = ZoneInfo("UTC")
        resolved_tz = "UTC"

    if target_date:
        try:
            local_day = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("date must be in YYYY-MM-DD format") from exc
    else:
        local_day = datetime.now(tz).date()

    start_local = datetime.combine(local_day, dt_time.min, tz)
    end_local = start_local + timedelta(days=1)

    from_iso = _to_utc_iso_z(start_local.astimezone(timezone.utc))
    to_iso = _to_utc_iso_z(end_local.astimezone(timezone.utc))
    return from_iso, to_iso, local_day.isoformat(), resolved_tz


def _is_in_person_event(event: dict) -> bool:
    """
    Normalize event mode classification with backward-compatible heuristics.

    Priority:
    1) explicit `meeting_mode`
    2) fallback to inverse of `is_virtual` for legacy rows
    """
    mode = (event.get("meeting_mode") or "unknown").strip().lower()
    if mode == "in_person":
        return True
    if mode == "online":
        return False
    return not bool(event.get("is_virtual"))


def _format_event_time(start_value: str | None, timezone_name: str) -> str:
    """Render an ISO datetime as `HH:MM` in the requested timezone."""
    if not start_value:
        return "unknown time"
    try:
        dt = datetime.fromisoformat(start_value.replace("Z", "+00:00"))
        try:
            dt = dt.astimezone(ZoneInfo(timezone_name))
        except Exception:
            pass
        return dt.strftime("%H:%M")
    except Exception:
        return "unknown time"


def _build_summary_payload(result: dict, user_sub: str, resolved_date: str, resolved_tz: str) -> dict:
    """
    Convert raw graph state output into stable API response contract.

    The voice backend depends on this shape (`counts`, `summary_text`, etc.),
    so this helper keeps that mapping centralized.
    """
    events = result.get("events") or []
    in_person_events = result.get("in_person_events") or []
    online_events = [event for event in events if not _is_in_person_event(event)]
    risk_summary = result.get("risk_summary") or []
    recommendations = result.get("recommendations") or []
    retrieved_context = result.get("retrieved_context") or []
    if not events:
        summary_text = f"You have no meetings on {resolved_date}."
    else:
        lines = [
            f"On {resolved_date}, you have {len(events)} meetings: "
            f"{len(in_person_events)} in-person and {len(online_events)} online."
        ]

        if online_events:
            online_labels = [
                f"{event.get('title', 'Untitled')} at {_format_event_time(event.get('time'), resolved_tz)}"
                for event in online_events
            ]
            lines.append("Online meetings: " + "; ".join(online_labels) + ".")

        if recommendations:
            lines.append("Weather guidance: " + " ".join(recommendations))

        summary_text = " ".join(lines)

    return {
        "user_sub": user_sub,
        "date": resolved_date,
        "timezone": resolved_tz,
        "counts": {
            "total": len(events),
            "in_person": len(in_person_events),
            "online": len(online_events),
        },
        "events": events,
        "risk_summary": risk_summary,
        "recommendations": recommendations,
        "retrieved_context": retrieved_context,
        "summary_text": summary_text,
        
    }


@app.get("/health")
def health():
    """Minimal liveness probe for orchestration platforms and local checks."""
    return {"ok": True}


@app.get("/internal/meeting-weather-summary")
def internal_meeting_weather_summary(
    user_sub: str = Query(..., description="Google user sub"),
    date: str | None = Query(default=None, description="YYYY-MM-DD"),
    tz: str = Query(default="Europe/Berlin", description="IANA timezone"),
    x_internal_api_key: str | None = Header(default=None),
):
    """
    Backend endpoint consumed by voice-scheduling-agent.
    It runs the LangGraph pipeline and returns summary + structured details.
    """
    err = require_internal_api_key(x_internal_api_key)
    if err:
        return err
    total_started = time.perf_counter()
    try:
        from_iso, to_iso, resolved_date, resolved_tz = _day_window_utc(date, tz)
        graph_started = time.perf_counter()
        request_started_at_monotonic = time.monotonic()
        llm_budget_seconds = 6.0
        llm_min_time_remaining_seconds = 1.5
        result = MEETING_PREVIEW_APP.invoke(
            {
                "user_query": f"What are my meetings on {resolved_date}?",
                "user_id": "api",
                "user_sub": user_sub,
                "from_iso": from_iso,
                "to_iso": to_iso,
                "timezone": resolved_tz,
                "target_date": resolved_date,
                "request_started_at_monotonic": request_started_at_monotonic,
                "llm_deadline_monotonic": request_started_at_monotonic + llm_budget_seconds,
                "llm_min_time_remaining_seconds": llm_min_time_remaining_seconds,
            }
        )
        graph_ms = (time.perf_counter() - graph_started) * 1000
        build_started = time.perf_counter()

        payload = _build_summary_payload(result, user_sub, resolved_date, resolved_tz)
        build_ms = (time.perf_counter() - build_started) * 1000
        if result.get("error"):
            payload["error"] = result["error"]
        return payload
    
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    
    finally:
        total_ms = (time.perf_counter() - total_started) * 1000
        logger.info(
            "internal_meeting_weather_summary_done user_sub=%s date=%s tz=%s total_ms=%.1f graph_ms=%.1f payload_ms=%.1f",
            user_sub,
            date,
            tz,
            total_ms,
            locals().get("graph_ms", -1.0),
            locals().get("build_ms", -1.0),
        )


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
        
    saved_vector_count = save_chunk_vectors(
            document_id=document_id,
            user_sub=user_sub,
            chunks=chunks,
            vectors=vectors,
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
