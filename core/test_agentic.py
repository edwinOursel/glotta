#!/usr/bin/env python3
"""
Test script for the agentic Glotta system.

This script demonstrates the multi-agent architecture with different user intents.
"""

import os
from agentic_graph import AgenticGlotta
from japanese_generator import JapaneseGenerator


def print_separator():
    print("\n" + "="*70 + "\n")


def test_conversation():
    """Test conversational intent."""
    print_separator()
    print("TEST 1: Conversation Intent")
    print_separator()

    agentic = AgenticGlotta(generator)

    result = agentic.process(
        user_input="こんにちは！今日はいい天気ですね。",
        user_level="N5",
        use_constraints=True,
        constraint_mode="soft"
    )

    print("📥 Input:", "こんにちは！今日はいい天気ですね。")
    print("🎯 Intent:", result["intent"])
    print("📤 Response:", result["response"])
    print("📊 Validation:", result["validation"])
    print("🔄 Iterations:", result["iterations"])


def test_correction():
    """Test correction intent."""
    print_separator()
    print("TEST 2: Correction Intent")
    print_separator()

    agentic = AgenticGlotta(generator)

    result = agentic.process(
        user_input="私は学校を行きます。",  # Incorrect: を instead of に
        user_level="N4",
        use_constraints=True,
        constraint_mode="hard"
    )

    print("📥 Input:", "私は学校を行きます。")
    print("🎯 Intent:", result["intent"])
    print("✏️  Corrected:", result["corrected_input"])
    print("📤 Response:", result["response"])


def test_practice():
    """Test practice intent."""
    print_separator()
    print("TEST 3: Practice Intent")
    print_separator()

    agentic = AgenticGlotta(generator)

    result = agentic.process(
        user_input="Let me practice writing about my day",
        user_level="N5",
        use_constraints=True,
        constraint_mode="adaptive"
    )

    print("📥 Input:", "Let me practice writing about my day")
    print("🎯 Intent:", result["intent"])
    print("📤 Response:", result["response"])


def test_no_constraints():
    """Test without vocabulary constraints."""
    print_separator()
    print("TEST 4: No Constraints (Advanced Mode)")
    print_separator()

    agentic = AgenticGlotta(generator)

    result = agentic.process(
        user_input="日本の文化について教えてください。",
        user_level="N1",
        use_constraints=False,
        max_iterations=1
    )

    print("📥 Input:", "日本の文化について教えてください。")
    print("🎯 Intent:", result["intent"])
    print("📤 Response:", result["response"])


def main():
    """Run all tests."""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not set!")
        print("   The agentic system requires OpenAI API for intent detection,")
        print("   error correction, and validation agents.")
        print()
        print("   Set it with: export OPENAI_API_KEY='sk-...'")
        print()
        response = input("Continue anyway? (tests may fail) [y/N]: ")
        if response.lower() != 'y':
            print("Exiting...")
            return

    print("""
╔════════════════════════════════════════════════════════════════════╗
║              Glotta Agentic System - Test Suite                    ║
║                                                                      ║
║  This test suite demonstrates the multi-agent architecture:         ║
║  - Intent Detection                                                  ║
║  - Error Correction                                                  ║
║  - Dynamic System Prompts                                            ║
║  - Constrained Generation                                            ║
║  - Response Validation                                               ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    global generator
    print("🔧 Initializing Japanese Generator...")
    generator = JapaneseGenerator(model_name="gpt2-small")
    print("✅ Generator ready!\n")

    # Run tests
    try:
        test_conversation()
        test_correction()
        test_practice()
        test_no_constraints()

        print_separator()
        print("✨ All tests completed!")
        print_separator()

    except KeyboardInterrupt:
        print("\n\n👋 Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
