"""Document management API."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import shutil
from pathlib import Path
import re

from src.config import SUPPORTED_EXTENSIONS, DATA_PATH  # Use shared configuration

router = APIRouter(prefix="/api/documents", tags=["documents"])

# ============ Helpers ============
def get_data_path() -> Path:
    """
    Return the data directory path and ensure it exists.

    Returns:
        Path object for the data directory.
    """
    path = Path(DATA_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to avoid path traversal.

    Args:
        filename: Raw filename.

    Returns:
        Safe filename.

    Raises:
        ValueError: filename is unsafe.
    """
    # Strip path separators
    filename = filename.replace("/", "").replace("\\", "")

    # Keep only alphanumerics, underscore, hyphen, dot
    filename = re.sub(r'[^\w\-.]', '_', filename)

    # Detect traversal patterns
    if ".." in filename or filename.startswith("."):
        raise ValueError("Filename contains illegal characters")

    # Length limits
    if len(filename) > 255:
        raise ValueError("Filename too long")

    if not filename:
        raise ValueError("Filename is empty")

    return filename

def validate_file_extension(filename: str) -> bool:
    """
    Validate whether the extension is supported.

    Args:
        filename: Filename.

    Returns:
        True if supported.
    """
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS

def get_safe_file_path(filename: str) -> Path:
    """
    Build a safe file path inside the data directory.

    Args:
        filename: Filename.

    Returns:
        Fully resolved safe path.

    Raises:
        ValueError: path is unsafe.
    """
    safe_filename = sanitize_filename(filename)

    data_path = get_data_path()

    file_path = data_path / safe_filename

    try:
        file_path = file_path.resolve()

        if not str(file_path).startswith(str(data_path)):
            raise ValueError("Path traversal detected")
    except Exception:
        raise ValueError("File path is unsafe")

    return file_path

# ============ API endpoints ============

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document with safety checks.

    Safety:
    - Filename sanitization (prevent traversal)
    - Extension allowlist
    - Resolved path validation
    - File size limit

    After upload, rerun ingest.py to rebuild the vector store.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is empty")

        try:
            safe_filename = sanitize_filename(file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Unsafe filename: {str(e)}")

        if not validate_file_extension(safe_filename):
            supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: {supported}"
            )

        try:
            file_path = get_safe_file_path(safe_filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Unsafe path: {str(e)}")

        if file_path.exists():
            raise HTTPException(
                status_code=409,
                detail=f"File already exists: {safe_filename}"
            )

        MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
        file_size = 0

        with open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):  # 8KB chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    buffer.close()
                    file_path.unlink()
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large; max {MAX_FILE_SIZE / (1024**2):.0f}MB"
                    )
                buffer.write(chunk)

        return {
            "message": "File uploaded successfully",
            "filename": safe_filename,
            "original_filename": file.filename,
            "size_bytes": file_size,
            "size_mb": f"{file_size / (1024**2):.2f} MB",
            "data_path": str(get_data_path()),
            "note": "Run 'python ingest.py' to rebuild the vector store"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/list")
async def list_documents():
    """List all documents (with safety checks)."""
    try:
        files = []
        data_path = get_data_path()

        for ext in SUPPORTED_EXTENSIONS.keys():
            for file_path in data_path.rglob(f"*{ext}"):
                if file_path.is_file():
                    try:
                        file_path.resolve().relative_to(data_path)

                        files.append({
                            "filename": file_path.name,
                            "path": str(file_path.relative_to(data_path)),
                            "size": file_path.stat().st_size,
                            "size_mb": f"{file_path.stat().st_size / (1024**2):.2f} MB",
                            "extension": file_path.suffix
                        })
                    except ValueError:
                        continue

        return {
            "documents": files,
            "count": len(files),
            "data_path": str(data_path),
            "supported_formats": list(SUPPORTED_EXTENSIONS.keys())
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{filename}")
async def delete_document(filename: str):
    """Delete a document with safety validation."""
    try:
        try:
            safe_filename = sanitize_filename(filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Unsafe filename: {str(e)}")

        try:
            file_path = get_safe_file_path(safe_filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Unsafe path: {str(e)}")

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        file_path.unlink()

        return {
            "message": "File deleted successfully",
            "filename": safe_filename,
            "data_path": str(get_data_path()),
            "note": "Run 'python ingest.py' to rebuild the vector store"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/supported-formats")
async def get_supported_formats():
    """Return the supported file formats."""
    return {
        "formats": [
            {
                "extension": ext,
                "type": file_type,
                "description": {
                    "text": "Plain text file",
                    "pdf": "PDF document",
                    "docx": "Word document",
                    "csv": "CSV table",
                    "html": "HTML page"
                }.get(file_type, "Unknown type")
            }
            for ext, file_type in SUPPORTED_EXTENSIONS.items()
        ]
    }

@router.get("/config")
async def get_documents_config():
    """
    Return document management configuration (debug helper).
    """
    data_path = get_data_path()

    return {
        "data_path": str(data_path),
        "data_path_exists": data_path.exists(),
        "data_path_absolute": str(data_path.resolve()),
        "supported_extensions": list(SUPPORTED_EXTENSIONS.keys()),
        "max_file_size_mb": 100,
        "file_count": len(list(data_path.rglob("*.*"))),
    }
