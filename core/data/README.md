# Vocabulary Data

This directory contains JLPT vocabulary data and scripts for seeding databases.

## Files

- `jlpt_vocabulary.json` - JLPT vocabulary organized by level (N5-N1)
- `seed_vocabulary.py` - Script to create and populate SQLite database

## Usage

### Create vocabulary database

```bash
# Create database with all JLPT levels
python seed_vocabulary.py

# Create database with specific levels only
python seed_vocabulary.py --levels N5,N4

# Reset and recreate database
python seed_vocabulary.py --reset

# Custom output path
python seed_vocabulary.py --db-path /path/to/vocabulary.db
```

### Export to JSON (for backend)

```bash
# Export all vocabulary
python seed_vocabulary.py --export vocabulary_export.json

# Export specific level
python seed_vocabulary.py --export n5_vocab.json --export-level N5
```

### Use in mobile app

```bash
# 1. Create database
python seed_vocabulary.py --db-path vocabulary.db

# 2. Copy to Flutter assets or use seed API endpoint
# The mobile app will use DatabaseService to access it
```

### Use in Python backend

```python
from user_vocabulary import UserVocabulary

# Load vocabulary from exported JSON
vocab = UserVocabulary(tokenizer)
vocab.load_from_file('vocabulary_export.json')
```

## Vocabulary Structure

Each word entry contains:
- `word`: Japanese word (kanji/kana)
- `reading`: Reading in hiragana
- `meaning`: English meaning
- `part_of_speech`: Word type (noun, verb, adjective, etc.)
- `jlpt_level`: JLPT level (N5-N1)

Example:
```json
{
  "word": "猫",
  "reading": "ねこ",
  "meaning": "cat",
  "part_of_speech": "noun",
  "jlpt_level": "N5"
}
```

## JLPT Levels

- **N5**: ~100 words (beginner - hiragana, katakana, basic kanji)
- **N4**: ~300 words (elementary)
- **N3**: ~650 words (intermediate)
- **N2**: ~1000 words (upper-intermediate)
- **N1**: ~2000 words (advanced)

**Note**: This dataset contains representative samples for each level, not exhaustive word lists.

## Expanding the vocabulary

To add more words, edit `jlpt_vocabulary.json` and run:

```bash
python seed_vocabulary.py --reset
```

## Sources

Vocabulary compiled from:
- Official JLPT word lists
- Common Japanese learning resources
- Frequency analysis of Japanese texts
