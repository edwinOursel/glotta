#!/usr/bin/env python3
"""
Seed vocabulary database with JLPT words.

This script creates and populates a SQLite database with Japanese vocabulary
organized by JLPT levels (N5-N1).

Usage:
    python seed_vocabulary.py [--db-path PATH] [--levels N5,N4,N3]

The database can be used by:
1. Flutter mobile app (copy to app's database location)
2. Python backend (to initialize user vocabulary)
"""

import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime


def create_database(db_path: str):
    """Create the database schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create vocabulary table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vocabulary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL UNIQUE,
            reading TEXT,
            meaning TEXT,
            part_of_speech TEXT,
            jlpt_level TEXT,
            date_added TEXT NOT NULL,
            times_seen INTEGER DEFAULT 0,
            times_correct INTEGER DEFAULT 0,
            mastery_level INTEGER DEFAULT 0,
            notes TEXT
        )
    ''')

    # Create index on JLPT level
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_jlpt_level ON vocabulary(jlpt_level)
    ''')

    conn.commit()
    return conn


def load_vocabulary_data(json_path: str) -> dict:
    """Load vocabulary data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def insert_vocabulary(conn, vocabulary_data: dict, levels: list = None):
    """Insert vocabulary into database."""
    cursor = conn.cursor()
    date_added = datetime.now().isoformat()

    if levels is None:
        levels = list(vocabulary_data.keys())

    total_inserted = 0

    for level in levels:
        if level not in vocabulary_data:
            print(f"⚠️  Level {level} not found in vocabulary data")
            continue

        words = vocabulary_data[level]
        print(f"📚 Inserting {len(words)} words for JLPT {level}...")

        for word_data in words:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO vocabulary
                    (word, reading, meaning, part_of_speech, jlpt_level, date_added)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    word_data['word'],
                    word_data.get('reading'),
                    word_data.get('meaning'),
                    word_data.get('part_of_speech'),
                    level,
                    date_added
                ))
                total_inserted += 1
            except sqlite3.IntegrityError as e:
                print(f"   ⚠️  Skipping duplicate: {word_data['word']}")

    conn.commit()
    return total_inserted


def get_stats(conn):
    """Get database statistics."""
    cursor = conn.cursor()

    # Total count
    cursor.execute('SELECT COUNT(*) FROM vocabulary')
    total = cursor.fetchone()[0]

    # Count by level
    cursor.execute('''
        SELECT jlpt_level, COUNT(*)
        FROM vocabulary
        GROUP BY jlpt_level
        ORDER BY jlpt_level DESC
    ''')
    level_counts = cursor.fetchall()

    return {
        'total': total,
        'by_level': dict(level_counts)
    }


def export_to_json(conn, output_path: str, level: str = None):
    """Export vocabulary to JSON format for backend."""
    cursor = conn.cursor()

    if level:
        cursor.execute('''
            SELECT word, reading, meaning, part_of_speech, jlpt_level
            FROM vocabulary
            WHERE jlpt_level = ?
            ORDER BY word
        ''', (level,))
    else:
        cursor.execute('''
            SELECT word, reading, meaning, part_of_speech, jlpt_level
            FROM vocabulary
            ORDER BY jlpt_level DESC, word
        ''')

    rows = cursor.fetchall()

    words_data = []
    for row in rows:
        words_data.append({
            'word': row[0],
            'reading': row[1],
            'meaning': row[2],
            'part_of_speech': row[3],
            'jlpt_level': row[4]
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'words': words_data}, f, ensure_ascii=False, indent=2)

    return len(words_data)


def main():
    parser = argparse.ArgumentParser(
        description='Seed vocabulary database with JLPT words'
    )
    parser.add_argument(
        '--db-path',
        default='vocabulary.db',
        help='Path to SQLite database file (default: vocabulary.db)'
    )
    parser.add_argument(
        '--data-path',
        default='data/jlpt_vocabulary.json',
        help='Path to vocabulary JSON file (default: data/jlpt_vocabulary.json)'
    )
    parser.add_argument(
        '--levels',
        help='Comma-separated list of levels to import (e.g., N5,N4,N3)'
    )
    parser.add_argument(
        '--export',
        help='Export vocabulary to JSON file (for backend)'
    )
    parser.add_argument(
        '--export-level',
        help='Export only specific JLPT level'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset database (delete existing data)'
    )

    args = parser.parse_args()

    print("""
╔════════════════════════════════════════════════════════════════╗
║           Glotta - Vocabulary Database Seeder                  ║
╚════════════════════════════════════════════════════════════════╝
    """)

    db_path = Path(args.db_path)

    # Reset if requested
    if args.reset and db_path.exists():
        print(f"🗑️  Deleting existing database: {db_path}")
        db_path.unlink()

    # Create database
    print(f"📦 Creating/opening database: {db_path}")
    conn = create_database(str(db_path))

    # Load vocabulary data
    data_path = Path(__file__).parent / args.data_path
    if not data_path.exists():
        print(f"❌ Vocabulary data not found: {data_path}")
        return

    print(f"📖 Loading vocabulary from: {data_path}")
    vocabulary_data = load_vocabulary_data(str(data_path))

    # Parse levels
    levels = None
    if args.levels:
        levels = [l.strip() for l in args.levels.split(',')]
        print(f"📝 Importing levels: {', '.join(levels)}")
    else:
        print(f"📝 Importing all levels: {', '.join(vocabulary_data.keys())}")

    # Insert vocabulary
    total_inserted = insert_vocabulary(conn, vocabulary_data, levels)
    print(f"✅ Inserted {total_inserted} words")

    # Show stats
    print("\n📊 Database statistics:")
    stats = get_stats(conn)
    print(f"   Total words: {stats['total']}")
    for level, count in sorted(stats['by_level'].items(), reverse=True):
        print(f"   {level}: {count} words")

    # Export if requested
    if args.export:
        export_path = Path(args.export)
        print(f"\n📤 Exporting to: {export_path}")
        count = export_to_json(conn, str(export_path), args.export_level)
        print(f"✅ Exported {count} words")

    conn.close()
    print(f"\n✨ Done! Database saved to: {db_path}")

    print("""
📱 To use with Flutter mobile app:
   1. Copy vocabulary.db to mobile app's database location
   2. Or import words via API

🐍 To use with Python backend:
   1. Export to JSON: python seed_vocabulary.py --export vocabulary.json
   2. Load in backend: UserVocabulary.load_from_file('vocabulary.json')
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
