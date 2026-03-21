#!/usr/bin/env python3
"""
FastAPI server for Glotta - Language Learning with Constrained LLM

Provides REST API endpoints for the mobile app to:
- User auth (register / login / refresh)
- User profile + settings
- Per-user vocabulary management with SRS
- Learning sessions + progress
- Constrained LLM text generation (simple + agentic)
"""

import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from typing import List, Optional, Dict
import uvicorn

from database import init_db
from japanese_generator import JapaneseGenerator
from user_vocabulary import UserVocabulary
from agentic_graph import AgenticGlotta
from auth import get_current_user
from models import User
from limiter import limiter
from routers import auth, users, vocabulary, sessions, friends, challenges

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ============================================================================
# Lifespan — DB init on startup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="Glotta API",
    description="Language learning with constrained LLM generation",
    version="0.2.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    ms = int((time.monotonic() - start) * 1000)
    log.info("%s %s → %d (%dms)", request.method, request.url.path, response.status_code, ms)
    return response

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(vocabulary.router)
app.include_router(sessions.router)
app.include_router(friends.router)
app.include_router(challenges.router)

# CORS middleware — set CORS_ORIGINS env var as a comma-separated list.
# Defaults to localhost only. Mobile native clients bypass CORS entirely,
# so this mainly matters if a web client is ever added.
_cors_origins = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global state (TODO: Replace with proper database and session management)
# ============================================================================

# Initialize generator (lazy loading)
_generator: Optional[JapaneseGenerator] = None
_agentic_glotta: Optional[AgenticGlotta] = None

def get_generator() -> JapaneseGenerator:
    """Get or initialize the generator."""
    global _generator
    if _generator is None:
        print("🔧 Initializing Japanese Generator...")
        _generator = JapaneseGenerator(model_name="gpt2-small")
        print("✓ Generator ready!")
    return _generator

def get_agentic_glotta() -> AgenticGlotta:
    """Get or initialize the agentic system."""
    global _agentic_glotta
    if _agentic_glotta is None:
        print("🔧 Initializing Agentic Glotta system...")
        generator = get_generator()
        _agentic_glotta = AgenticGlotta(generator)
        print("✓ Agentic system ready!")
    return _agentic_glotta

# ============================================================================
# Request/Response Models
# ============================================================================

_ALLOWED_SYSTEM_PROMPT = re.compile(r"^[\w\s\u3000-\u9fff\u30a0-\u30ff\u3040-\u309f.,;:!?()\-]{1,200}$")

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 50
    temperature: float = 0.8
    use_constraints: bool = True
    constraint_mode: str = "hard"
    num_sequences: int = 1
    system_prompt: Optional[str] = None  # grammar theme hint (sanitized server-side)

class GenerateResponse(BaseModel):
    texts: List[str]
    prompt: str
    constraint_mode: Optional[str]

class Word(BaseModel):
    word: str
    reading: Optional[str] = None
    meaning: Optional[str] = None
    jlpt_level: Optional[str] = None

class AddWordsRequest(BaseModel):
    words: List[str]

class VocabularyResponse(BaseModel):
    total_words: int
    total_expressions: int
    total_token_ids: int
    sample_words: List[str]

class StatsResponse(BaseModel):
    vocabulary_size: int
    constraint_mode: str
    model_name: str

class AgenticGenerateRequest(BaseModel):
    user_input: str
    user_level: str = "N5"
    use_constraints: bool = True
    constraint_mode: str = "hard"
    max_iterations: int = 3

class AgenticGenerateResponse(BaseModel):
    response: str
    intent: Optional[str]
    corrected_input: Optional[str]
    validation: Optional[Dict]
    iterations: int

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Glotta API",
        "version": "0.1.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check with model status."""
    try:
        generator = get_generator()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_info": generator.get_model_info()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e)
        }

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate Japanese text with vocabulary constraints.

    Args:
        request: Generation parameters

    Returns:
        Generated text(s)
    """
    try:
        generator = get_generator()

        # Set constraint mode if different
        if request.use_constraints and generator.constraint_mode != request.constraint_mode:
            generator.set_constraint_mode(request.constraint_mode)

        # Prepend system_prompt — only if it matches the allowed pattern
        effective_prompt = request.prompt
        if request.system_prompt:
            if not _ALLOWED_SYSTEM_PROMPT.match(request.system_prompt):
                raise HTTPException(status_code=400, detail="Invalid system_prompt content")
            effective_prompt = f"{request.system_prompt}\n{request.prompt}"

        # Generate
        texts = generator.generate(
            prompt=effective_prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            use_constraints=request.use_constraints,
            num_return_sequences=request.num_sequences
        )

        return GenerateResponse(
            texts=texts,
            prompt=request.prompt,
            constraint_mode=request.constraint_mode if request.use_constraints else None
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Generation failed")
        raise HTTPException(status_code=500, detail="Generation failed")

@app.get("/api/vocabulary", response_model=VocabularyResponse)
async def get_vocabulary():
    """Get current vocabulary statistics."""
    try:
        generator = get_generator()
        stats = generator.user_vocabulary.get_stats()

        return VocabularyResponse(**stats)

    except Exception:
        log.exception("Failed to get vocabulary stats")
        raise HTTPException(status_code=500, detail="Failed to get vocabulary")

@app.post("/api/vocabulary/words")
async def add_words(request: AddWordsRequest):
    """Add words to user's vocabulary."""
    try:
        generator = get_generator()
        generator.user_vocabulary.add_words(request.words)

        return {
            "status": "success",
            "words_added": len(request.words),
            "total_words": len(generator.user_vocabulary)
        }

    except Exception:
        log.exception("Failed to add words")
        raise HTTPException(status_code=500, detail="Failed to add words")

@app.delete("/api/vocabulary/words/{word}")
async def remove_word(word: str):
    """Remove a word from user's vocabulary."""
    try:
        generator = get_generator()
        generator.user_vocabulary.remove_word(word)

        return {
            "status": "success",
            "word_removed": word,
            "total_words": len(generator.user_vocabulary)
        }

    except Exception:
        log.exception("Failed to remove word")
        raise HTTPException(status_code=500, detail="Failed to remove word")

_VOCAB_FILENAME_RE = re.compile(r'^[\w\-]+\.json$')
_VOCAB_DIR = Path("vocabularies")


def _safe_vocab_path(filename: str) -> Path:
    """Resolve filename inside the vocabularies/ directory, rejecting path traversal."""
    name = Path(filename).name  # strip any directory component
    if name != filename or not _VOCAB_FILENAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid filename")
    _VOCAB_DIR.mkdir(exist_ok=True)
    return _VOCAB_DIR / name


@app.post("/api/vocabulary/save")
async def save_vocabulary(filename: str = "user_vocabulary.json"):
    """Save vocabulary to file."""
    try:
        safe_path = _safe_vocab_path(filename)
        generator = get_generator()
        generator.user_vocabulary.save_to_file(str(safe_path))
        return {"status": "success", "filename": safe_path.name}
    except HTTPException:
        raise
    except Exception:
        log.exception("Failed to save vocabulary")
        raise HTTPException(status_code=500, detail="Failed to save vocabulary")


@app.post("/api/vocabulary/load")
async def load_vocabulary(filename: str = "user_vocabulary.json"):
    """Load vocabulary from file."""
    try:
        safe_path = _safe_vocab_path(filename)
        if not safe_path.exists():
            raise HTTPException(status_code=404, detail="Vocabulary file not found")
        generator = get_generator()
        generator.user_vocabulary.load_from_file(str(safe_path))
        return {"status": "success", "filename": safe_path.name, "total_words": len(generator.user_vocabulary)}
    except HTTPException:
        raise
    except Exception:
        log.exception("Failed to load vocabulary")
        raise HTTPException(status_code=500, detail="Failed to load vocabulary")

@app.get("/api/settings/constraint-mode")
async def get_constraint_mode():
    """Get current constraint mode."""
    try:
        generator = get_generator()
        return {
            "constraint_mode": generator.constraint_mode
        }
    except Exception:
        log.exception("Failed to get constraint mode")
        raise HTTPException(status_code=500, detail="Failed to get constraint mode")

@app.post("/api/settings/constraint-mode")
async def set_constraint_mode(mode: str):
    """Set constraint mode (hard/soft/adaptive)."""
    try:
        if mode not in ["hard", "soft", "adaptive"]:
            raise HTTPException(status_code=400, detail="Invalid mode. Must be: hard, soft, or adaptive")

        generator = get_generator()
        generator.set_constraint_mode(mode)

        return {
            "status": "success",
            "constraint_mode": mode
        }

    except HTTPException:
        raise
    except Exception:
        log.exception("Failed to set constraint mode")
        raise HTTPException(status_code=500, detail="Failed to set constraint mode")

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get general statistics."""
    try:
        generator = get_generator()

        return StatsResponse(
            vocabulary_size=len(generator.user_vocabulary),
            constraint_mode=generator.constraint_mode,
            model_name=generator.get_model_info()["model_name"]
        )

    except Exception:
        log.exception("Failed to get stats")
        raise HTTPException(status_code=500, detail="Failed to get stats")

@app.post("/api/agentic/generate", response_model=AgenticGenerateResponse)
async def agentic_generate(request: AgenticGenerateRequest):
    """
    Generate Japanese text using the agentic multi-LLM system.

    This endpoint uses LangGraph to orchestrate multiple LLMs:
    - Intent detection
    - Error correction
    - Dynamic system prompt building
    - Constrained generation
    - Response validation

    Args:
        request: Agentic generation parameters

    Returns:
        Generated response with metadata
    """
    try:
        agentic = get_agentic_glotta()

        result = agentic.process(
            user_input=request.user_input,
            user_level=request.user_level,
            use_constraints=request.use_constraints,
            constraint_mode=request.constraint_mode,
            max_iterations=request.max_iterations
        )

        return AgenticGenerateResponse(
            response=result["response"],
            intent=result.get("intent"),
            corrected_input=result.get("corrected_input"),
            validation=result.get("validation"),
            iterations=result.get("iterations", 0)
        )

    except Exception:
        log.exception("Agentic generation failed")
        raise HTTPException(status_code=500, detail="Agentic generation failed")

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                       GLOTTA API SERVER                        ║
    ║        Language Learning with Constrained LLM Generation       ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
