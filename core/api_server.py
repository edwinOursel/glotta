#!/usr/bin/env python3
"""
FastAPI server for Glotta - Language Learning with Constrained LLM

Provides REST API endpoints for the mobile app to:
- Generate text with vocabulary constraints
- Manage user vocabulary
- Get JLPT word lists
- Track learning progress
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn

from japanese_generator import JapaneseGenerator
from user_vocabulary import UserVocabulary

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="Glotta API",
    description="Language learning with constrained LLM generation",
    version="0.1.0"
)

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

def get_generator() -> JapaneseGenerator:
    """Get or initialize the generator."""
    global _generator
    if _generator is None:
        print("🔧 Initializing Japanese Generator...")
        _generator = JapaneseGenerator(model_name="gpt2-small")
        print("✓ Generator ready!")
    return _generator

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

        # Generate
        texts = generator.generate(
            prompt=request.prompt,
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
