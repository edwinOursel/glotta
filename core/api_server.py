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

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn

from database import init_db
from japanese_generator import JapaneseGenerator
from user_vocabulary import UserVocabulary
from agentic_graph import AgenticGlotta
from auth import get_current_user
from models import User
from routers import auth, users, vocabulary, sessions

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

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(vocabulary.router)
app.include_router(sessions.router)

# CORS middleware for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
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

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 50
    temperature: float = 0.8
    use_constraints: bool = True
    constraint_mode: str = "hard"
    num_sequences: int = 1
    system_prompt: Optional[str] = None  # grammar theme / word focus hint

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

        # Prepend system_prompt to prime the model toward the desired context
        effective_prompt = request.prompt
        if request.system_prompt:
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/api/vocabulary", response_model=VocabularyResponse)
async def get_vocabulary():
    """Get current vocabulary statistics."""
    try:
        generator = get_generator()
        stats = generator.user_vocabulary.get_stats()

        return VocabularyResponse(**stats)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get vocabulary: {str(e)}")

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add words: {str(e)}")

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove word: {str(e)}")

@app.post("/api/vocabulary/save")
async def save_vocabulary(filename: str = "user_vocabulary.json"):
    """Save vocabulary to file."""
    try:
        generator = get_generator()
        generator.user_vocabulary.save_to_file(filename)

        return {
            "status": "success",
            "filename": filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save vocabulary: {str(e)}")

@app.post("/api/vocabulary/load")
async def load_vocabulary(filename: str = "user_vocabulary.json"):
    """Load vocabulary from file."""
    try:
        generator = get_generator()
        generator.user_vocabulary.load_from_file(filename)

        return {
            "status": "success",
            "filename": filename,
            "total_words": len(generator.user_vocabulary)
        }

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to load vocabulary: {str(e)}")

@app.get("/api/settings/constraint-mode")
async def get_constraint_mode():
    """Get current constraint mode."""
    try:
        generator = get_generator()
        return {
            "constraint_mode": generator.constraint_mode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agentic generation failed: {str(e)}")

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
