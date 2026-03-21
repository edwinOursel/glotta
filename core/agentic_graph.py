#!/usr/bin/env python3
"""
Agentic LangGraph system for Glotta.

This module implements a multi-agent system using LangGraph to:
- Detect user intent
- Correct errors in user input
- Dynamically build system prompts
- Generate constrained Japanese text
- Validate response quality
"""

import logging
import os
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from japanese_generator import JapaneseGenerator

log = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class GraphState(TypedDict):
    """State passed through the graph."""
    # Input
    user_input: str
    user_level: str  # JLPT level (N5, N4, N3, N2, N1)
    use_constraints: bool
    constraint_mode: str

    # Intermediate
    intent: Optional[str]  # practice, correction, conversation, quiz, etc.
    corrected_input: Optional[str]
    system_prompt: Optional[str]
    generation_params: Optional[dict]

    # Output
    generated_text: Optional[str]
    validation_result: Optional[dict]
    feedback: Optional[str]
    final_response: Optional[str]

    # Metadata
    iterations: int
    max_iterations: int


# ============================================================================
# Agent Nodes
# ============================================================================

class IntentDetector:
    """Detects user intent from input."""

    INTENTS = {
        "practice": "User wants to practice writing/conversation",
        "correction": "User wants their Japanese corrected",
        "conversation": "User wants to have a conversation",
        "quiz": "User wants to be quizzed on vocabulary",
        "translation": "User wants translation help",
        "explanation": "User wants grammar/vocabulary explanation"
    }

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def detect(self, state: GraphState) -> GraphState:
        """Detect intent from user input."""
        log.info("[IntentDetector] Analyzing user intent...")

        prompt = f"""Analyze this user input and determine their intent.
User input: "{state['user_input']}"
User JLPT level: {state['user_level']}

Available intents:
{chr(10).join(f"- {k}: {v}" for k, v in self.INTENTS.items())}

Respond with ONLY the intent name (e.g., "practice", "correction", etc.).
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        intent = response.content.strip().lower()

        if intent not in self.INTENTS:
            intent = "conversation"  # Default fallback

        log.info("[IntentDetector] Detected intent: %s", intent)
        state["intent"] = intent
        return state


class ErrorCorrector:
    """Corrects errors in user's Japanese input if needed."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def correct(self, state: GraphState) -> GraphState:
        """Correct errors in user input if intent is correction."""
        log.info("[ErrorCorrector] Checking for errors...")

        if state["intent"] != "correction":
            state["corrected_input"] = state["user_input"]
            log.info("[ErrorCorrector] No correction needed for this intent")
            return state

        prompt = f"""You are a Japanese language teacher. Analyze this text and correct any errors.

Text: "{state['user_input']}"
User level: {state['user_level']}

If there are errors, provide:
1. The corrected version
2. Brief explanation of errors (in English)

If no errors, respond with "NO_ERRORS".

Format:
CORRECTED: <corrected text>
EXPLANATION: <explanation>
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        if "NO_ERRORS" in content:
            state["corrected_input"] = state["user_input"]
            state["feedback"] = "✓ Your Japanese is correct!"
        else:
            lines = content.split("\n")
            corrected = state["user_input"]
            explanation = ""

            for line in lines:
                if line.startswith("CORRECTED:"):
                    corrected = line.replace("CORRECTED:", "").strip()
                elif line.startswith("EXPLANATION:"):
                    explanation = line.replace("EXPLANATION:", "").strip()

            state["corrected_input"] = corrected
            state["feedback"] = f"Corrections:\n{explanation}"

        log.info("[ErrorCorrector] Result: %s", state["corrected_input"])
        return state


class SystemPromptBuilder:
    """Dynamically builds system prompts based on intent and user level."""

    PROMPTS = {
        "practice": """You are a Japanese language practice partner.
Help the user practice at JLPT {level} level.
Keep your responses simple and appropriate for their level.
Use vocabulary they know when possible.""",

        "conversation": """You are a friendly Japanese conversation partner.
Adapt to JLPT {level} level.
Be natural and engaging, but keep vocabulary appropriate for their level.""",

        "quiz": """You are a Japanese vocabulary quiz master.
Create quiz questions appropriate for JLPT {level}.
Test their knowledge with engaging questions.""",

        "translation": """You are a Japanese-English translation helper.
Help with translations at JLPT {level} level.
Explain nuances and provide context.""",

        "explanation": """You are a Japanese language teacher.
Explain concepts clearly for JLPT {level} students.
Use simple examples and break down complex ideas."""
    }

    def build(self, state: GraphState) -> GraphState:
        """Build system prompt based on intent and level."""
        log.info("[SystemPromptBuilder] Building system prompt...")

        intent = state.get("intent", "conversation")
        level = state.get("user_level", "N5")

        template = self.PROMPTS.get(intent, self.PROMPTS["conversation"])
        state["system_prompt"] = template.format(level=level)

        log.info("[SystemPromptBuilder] Built prompt for intent=%s level=%s", intent, level)
        return state


class ConstrainedGenerator:
    """Generates Japanese text with vocabulary constraints."""

    def __init__(self, generator: JapaneseGenerator):
        self.generator = generator

    def generate(self, state: GraphState) -> GraphState:
        """Generate constrained Japanese text."""
        log.info("[ConstrainedGenerator] Generating response...")

        prompt = state.get("corrected_input") or state["user_input"]

        if state.get("system_prompt"):
            prompt = f"{state['system_prompt']}\n\nUser: {prompt}\nAssistant:"

        try:
            texts = self.generator.generate(
                prompt=prompt,
                max_length=100,
                temperature=0.8,
                use_constraints=state.get("use_constraints", True),
                num_return_sequences=1
            )

            generated = texts[0] if texts else ""
            state["generated_text"] = generated
            log.info("[ConstrainedGenerator] Generated %d chars", len(generated))

        except Exception:
            log.exception("[ConstrainedGenerator] Generation failed")
            state["generated_text"] = "申し訳ございません。エラーが発生しました。"

        return state


class ResponseValidator:
    """Validates generated response quality and appropriateness."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def validate(self, state: GraphState) -> GraphState:
        """Validate response quality."""
        log.info("[ResponseValidator] Validating response...")

        if not state.get("use_constraints") or state["iterations"] >= state["max_iterations"]:
            state["validation_result"] = {"valid": True, "score": 1.0}
            state["final_response"] = state["generated_text"]
            return state

        prompt = f"""Evaluate this Japanese response for quality and appropriateness.

User level: {state['user_level']}
Intent: {state['intent']}
User input: {state['user_input']}
Generated response: {state['generated_text']}

Evaluate:
1. Grammar correctness (0-1)
2. Appropriate for user level (0-1)
3. Relevance to input (0-1)
4. Natural Japanese (0-1)

Respond in this format:
SCORE: <average score 0-1>
VALID: <YES/NO>
FEEDBACK: <brief feedback>
"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            score = 0.7
            valid = True
            feedback = ""

            for line in content.split("\n"):
                if line.startswith("SCORE:"):
                    try:
                        score = float(line.replace("SCORE:", "").strip())
                    except ValueError:
                        pass
                elif line.startswith("VALID:"):
                    valid = "YES" in line.upper()
                elif line.startswith("FEEDBACK:"):
                    feedback = line.replace("FEEDBACK:", "").strip()

            state["validation_result"] = {"valid": valid, "score": score, "feedback": feedback}
            log.info("[ResponseValidator] valid=%s score=%.2f", valid, score)

        except Exception:
            log.exception("[ResponseValidator] Validation failed, assuming valid")
            state["validation_result"] = {"valid": True, "score": 0.5}

        return state


def should_retry(state: GraphState) -> str:
    """Decide if we should retry generation."""
    validation = state.get("validation_result", {})

    if validation.get("valid", True) or state["iterations"] >= state["max_iterations"]:
        return "finalize"

    if validation.get("score", 1.0) < 0.6:
        state["iterations"] += 1
        return "retry"

    return "finalize"


def finalize_response(state: GraphState) -> GraphState:
    """Finalize the response with feedback."""
    log.info("[Finalizer] Preparing final response...")

    final_response = state["generated_text"]

    if state.get("feedback"):
        final_response = f"{state['feedback']}\n\n{final_response}"

    validation = state.get("validation_result", {})
    if validation.get("score", 1.0) < 0.8 and validation.get("feedback"):
        final_response += f"\n\nNote: {validation['feedback']}"

    state["final_response"] = final_response
    log.info("[Finalizer] Final response ready")
    return state


# ============================================================================
# Graph Construction
# ============================================================================

def create_agentic_graph(generator: JapaneseGenerator) -> StateGraph:
    """Create the agentic LangGraph system."""

    intent_detector = IntentDetector()
    error_corrector = ErrorCorrector()
    prompt_builder = SystemPromptBuilder()
    constrained_gen = ConstrainedGenerator(generator)
    validator = ResponseValidator()

    workflow = StateGraph(GraphState)

    workflow.add_node("detect_intent", intent_detector.detect)
    workflow.add_node("correct_errors", error_corrector.correct)
    workflow.add_node("build_prompt", prompt_builder.build)
    workflow.add_node("generate", constrained_gen.generate)
    workflow.add_node("validate", validator.validate)
    workflow.add_node("finalize", finalize_response)

    workflow.set_entry_point("detect_intent")
    workflow.add_edge("detect_intent", "correct_errors")
    workflow.add_edge("correct_errors", "build_prompt")
    workflow.add_edge("build_prompt", "generate")
    workflow.add_edge("generate", "validate")

    workflow.add_conditional_edges(
        "validate",
        should_retry,
        {"retry": "generate", "finalize": "finalize"}
    )

    workflow.add_edge("finalize", END)

    return workflow.compile()


# ============================================================================
# Main Interface
# ============================================================================

class AgenticGlotta:
    """Main interface for the agentic Glotta system."""

    def __init__(self, generator: JapaneseGenerator):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "The agentic system requires a valid OpenAI API key."
            )
        self.generator = generator
        self.graph = create_agentic_graph(generator)

    def process(
        self,
        user_input: str,
        user_level: str = "N5",
        use_constraints: bool = True,
        constraint_mode: str = "hard",
        max_iterations: int = 3
    ) -> dict:
        """Process user input through the agentic graph."""

        initial_state: GraphState = {
            "user_input": user_input,
            "user_level": user_level,
            "use_constraints": use_constraints,
            "constraint_mode": constraint_mode,
            "intent": None,
            "corrected_input": None,
            "system_prompt": None,
            "generation_params": None,
            "generated_text": None,
            "validation_result": None,
            "feedback": None,
            "final_response": None,
            "iterations": 0,
            "max_iterations": max_iterations
        }

        log.info("Starting agentic processing for input: %.50s", user_input)
        final_state = self.graph.invoke(initial_state)
        log.info("Agentic processing complete after %d iteration(s)", final_state.get("iterations", 0))

        return {
            "response": final_state.get("final_response", ""),
            "intent": final_state.get("intent"),
            "corrected_input": final_state.get("corrected_input"),
            "validation": final_state.get("validation_result"),
            "iterations": final_state.get("iterations", 0)
        }
