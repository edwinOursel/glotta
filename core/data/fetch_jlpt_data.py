#!/usr/bin/env python3
"""
Fetch comprehensive JLPT vocabulary data from GitHub repositories.

This script downloads vocabulary data from elzup/jlpt-word-list and
converts it to our JSON format for seeding the database.
"""

import csv
import json
import urllib.request
from pathlib import Path
from typing import Dict, List

# URLs for JLPT vocabulary CSV files
JLPT_URLS = {
    'N5': 'https://raw.githubusercontent.com/elzup/jlpt-word-list/master/src/n5.csv',
    'N4': 'https://raw.githubusercontent.com/elzup/jlpt-word-list/master/src/n4.csv',
    'N3': 'https://raw.githubusercontent.com/elzup/jlpt-word-list/master/src/n3.csv',
    'N2': 'https://raw.githubusercontent.com/elzup/jlpt-word-list/master/src/n2.csv',
    'N1': 'https://raw.githubusercontent.com/elzup/jlpt-word-list/master/src/n1.csv',
}


def infer_part_of_speech(word: str, reading: str, meaning: str) -> str:
    """Infer part of speech from word characteristics."""
    meaning_lower = meaning.lower()

    # Verb patterns
    if any(verb_marker in meaning_lower for verb_marker in ['to ', 'will ']):
        if word.endswith('する'):
            return 'suru-verb'
        elif reading.endswith('る'):
            return 'verb'
        else:
            return 'verb'

    # Adjective patterns
    if word.endswith('い') and not word.endswith('ない'):
        return 'i-adjective'

    # Na-adjective patterns (common endings)
    if any(word.endswith(na) for na in ['的', '的な', 'な']):
        return 'na-adjective'

    # Particles
    if len(word) <= 2 and any(p in word for p in ['は', 'が', 'を', 'に', 'で', 'の', 'と', 'も', 'から', 'まで']):
        return 'particle'

    # Expressions/greetings
    if any(expr in meaning_lower for expr in ['hello', 'goodbye', 'thank', 'excuse me', 'sorry']):
        return 'expression'

    # Default to noun
    return 'noun'


def fetch_vocabulary(url: str, level: str) -> List[Dict]:
    """Fetch vocabulary from URL and parse CSV."""
    print(f"📥 Fetching {level} vocabulary from GitHub...")

    try:
        with urllib.request.urlopen(url) as response:
            content = response.read().decode('utf-8')

        # Parse CSV
        reader = csv.DictReader(content.splitlines())
        words = []

        for row in reader:
            expression = row.get('expression', '').strip()
            reading = row.get('reading', '').strip()
            meaning = row.get('meaning', '').strip()

            if not expression or not meaning:
                continue

            # Infer part of speech
            part_of_speech = infer_part_of_speech(expression, reading, meaning)

            words.append({
                'word': expression,
                'reading': reading if reading else expression,
                'meaning': meaning,
                'part_of_speech': part_of_speech
            })

        print(f"   ✅ Fetched {len(words)} words for {level}")
        return words

    except Exception as e:
        print(f"   ❌ Error fetching {level}: {e}")
        return []


def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║           JLPT Vocabulary Data Fetcher                         ║
╚════════════════════════════════════════════════════════════════╝
    """)

    all_vocabulary = {}
    total_words = 0

    # Fetch all levels
    for level, url in JLPT_URLS.items():
        words = fetch_vocabulary(url, level)
        if words:
            all_vocabulary[level] = words
            total_words += len(words)

    if not all_vocabulary:
        print("\n❌ No vocabulary data fetched. Exiting.")
        return

    # Save to JSON
    output_path = Path(__file__).parent / 'jlpt_vocabulary.json'

    print(f"\n💾 Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_vocabulary, f, ensure_ascii=False, indent=2)

    print(f"✅ Successfully saved {total_words} words!")

    # Show statistics
    print("\n📊 Vocabulary statistics:")
    for level in ['N5', 'N4', 'N3', 'N2', 'N1']:
        if level in all_vocabulary:
            count = len(all_vocabulary[level])
            print(f"   {level}: {count:,} words")

    print(f"\n   Total: {total_words:,} words")

    print("""
✨ Next steps:
   1. Review the generated jlpt_vocabulary.json file
   2. Run: python seed_vocabulary.py --reset
   3. The database will now contain thousands of JLPT words!
    """)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
