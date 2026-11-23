"""Knowledge-base endpoints: upload, list, search, delete."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from ..config import settings
from ..rag.service import TEXT_SUFFIXES, PDF_SUFFIXES, rag, safe_filename
from ..schemas import (
    DocumentInfo,
    DocumentListResponse,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    TextDocumentRequest,
)
from ..security import guard

router = APIRouter(prefix="/api/documents", tags=["documents"], dependencies=[Depends(guard)])

ALLOWED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES


async def _list() -> DocumentListResponse:
    await rag.ensure_seeded()
    counts = rag.sources()
    documents = [DocumentInfo(source=source, chunks=chunks) for source, chunks in sorted(counts.items())]
    return DocumentListResponse(
        documents=documents,
        total_chunks=rag.total_chunks,
        embedding_backend=settings.embedding_backend,
    )


@router.get("", response_model=DocumentListResponse, summary="List indexed documents")
async def list_documents() -> DocumentListResponse:
    return await _list()


@router.post("", response_model=IngestResponse, status_code=status.HTTP_201_CREATED, summary="Upload a document")
async def upload_document(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "A file name is required.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type '{suffix or 'none'}'. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    data = await file.read()
    limit = settings.max_upload_mb * 1024 * 1024
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")
    if len(data) > limit:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File is larger than {settings.max_upload_mb} MB (Vercel's body limit is ~4.5 MB).",
        )

    try:
        chunks = await rag.add_file(file.filename, data)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Could not index file: {exc}") from exc

    return IngestResponse(
        source=safe_filename(file.filename), chunks=chunks, total_chunks=rag.total_chunks
    )


@router.post("/text", response_model=IngestResponse, status_code=status.HTTP_201_CREATED, summary="Add raw text")
async def add_text(request: TextDocumentRequest) -> IngestResponse:
    try:
        chunks = await rag.add_text(request.text, request.source)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return IngestResponse(source=request.source, chunks=chunks, total_chunks=rag.total_chunks)


@router.post("/search", response_model=SearchResponse, summary="Search the knowledge base")
async def search(request: SearchRequest) -> SearchResponse:
    hits = await rag.search(request.query, k=request.k)
    return SearchResponse(query=request.query, hits=hits)


@router.delete("", response_model=IngestResponse, summary="Delete a document")
async def delete_document(source: str = Query(min_length=1, max_length=200)) -> IngestResponse:
    removed = rag.remove(source)
    if removed == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No indexed document named '{source}'.")
    return IngestResponse(source=source, chunks=removed, total_chunks=rag.total_chunks)
