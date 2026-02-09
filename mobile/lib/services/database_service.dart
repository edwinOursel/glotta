import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import '../models/word.dart';
import '../models/generated_text.dart';
import '../models/learning_session.dart';

/// SQLite database service for local data persistence
class DatabaseService {
  static final DatabaseService _instance = DatabaseService._internal();
  static Database? _database;

  factory DatabaseService() => _instance;

  DatabaseService._internal();

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    final databasesPath = await getDatabasesPath();
    final path = join(databasesPath, 'glotta.db');

    return await openDatabase(
      path,
      version: 1,
      onCreate: _onCreate,
      onUpgrade: _onUpgrade,
    );
  }

  Future<void> _onCreate(Database db, int version) async {
    // Vocabulary table
    await db.execute('''
      CREATE TABLE vocabulary (
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
    ''');

    // Create index on JLPT level for faster queries
    await db.execute('''
      CREATE INDEX idx_jlpt_level ON vocabulary(jlpt_level)
    ''');

    // Generated texts table
    await db.execute('''
      CREATE TABLE generated_texts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        prompt TEXT NOT NULL,
        constraint_mode TEXT,
        timestamp TEXT NOT NULL
      )
    ''');

    // Learning sessions table
    await db.execute('''
      CREATE TABLE learning_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        duration INTEGER NOT NULL,
        words_practiced INTEGER DEFAULT 0,
        constraint_mode TEXT,
        notes TEXT
      )
    ''');

    // User settings table
    await db.execute('''
      CREATE TABLE user_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      )
    ''');
  }

  Future<void> _onUpgrade(Database db, int oldVersion, int newVersion) async {
    // Handle database migrations here
    if (oldVersion < 2) {
      // Example migration
      // await db.execute('ALTER TABLE vocabulary ADD COLUMN new_field TEXT');
    }
  }

  // =========================================================================
  // Vocabulary CRUD operations
  // =========================================================================

  Future<int> insertWord(Word word) async {
    final db = await database;
    return await db.insert(
      'vocabulary',
      word.toMap(),
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  Future<int> insertWords(List<Word> words) async {
    final db = await database;
    final batch = db.batch();

    for (final word in words) {
      batch.insert(
        'vocabulary',
        word.toMap(),
        conflictAlgorithm: ConflictAlgorithm.replace,
      );
    }

    final results = await batch.commit();
    return results.length;
  }

  Future<List<Word>> getVocabulary() async {
    final db = await database;
    final List<Map<String, dynamic>> maps = await db.query('vocabulary');
    return List.generate(maps.length, (i) => Word.fromMap(maps[i]));
  }

  Future<List<Word>> getVocabularyByLevel(String jlptLevel) async {
    final db = await database;
    final List<Map<String, dynamic>> maps = await db.query(
      'vocabulary',
      where: 'jlpt_level = ?',
      whereArgs: [jlptLevel],
    );
    return List.generate(maps.length, (i) => Word.fromMap(maps[i]));
  }

  Future<Word?> getWord(String word) async {
    final db = await database;
    final List<Map<String, dynamic>> maps = await db.query(
      'vocabulary',
      where: 'word = ?',
      whereArgs: [word],
      limit: 1,
    );

    if (maps.isEmpty) return null;
    return Word.fromMap(maps.first);
  }

  Future<int> updateWord(Word word) async {
    final db = await database;
    return await db.update(
      'vocabulary',
      word.toMap(),
      where: 'id = ?',
      whereArgs: [word.id],
    );
  }

  Future<int> deleteWord(int id) async {
    final db = await database;
    return await db.delete(
      'vocabulary',
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<int> deleteWordByText(String word) async {
    final db = await database;
    return await db.delete(
      'vocabulary',
      where: 'word = ?',
      whereArgs: [word],
    );
  }

  Future<List<Word>> searchVocabulary(String query) async {
    final db = await database;
    final List<Map<String, dynamic>> maps = await db.query(
      'vocabulary',
      where: 'word LIKE ? OR reading LIKE ? OR meaning LIKE ?',
      whereArgs: ['%$query%', '%$query%', '%$query%'],
    );
    return List.generate(maps.length, (i) => Word.fromMap(maps[i]));
  }

  Future<int> getVocabularyCount() async {
    final db = await database;
    final result = await db.rawQuery('SELECT COUNT(*) FROM vocabulary');
    return Sqflite.firstIntValue(result) ?? 0;
  }

  Future<int> getVocabularyCountByLevel(String jlptLevel) async {
    final db = await database;
    final result = await db.rawQuery(
      'SELECT COUNT(*) FROM vocabulary WHERE jlpt_level = ?',
      [jlptLevel],
    );
    return Sqflite.firstIntValue(result) ?? 0;
  }

  // =========================================================================
  // Generated texts operations
  // =========================================================================

  Future<int> insertGeneratedText(GeneratedText text) async {
    final db = await database;
    return await db.insert('generated_texts', text.toMap());
  }

  Future<List<GeneratedText>> getGeneratedTexts({int limit = 50}) async {
    final db = await database;
    final List<Map<String, dynamic>> maps = await db.query(
      'generated_texts',
      orderBy: 'timestamp DESC',
      limit: limit,
    );
    return List.generate(maps.length, (i) => GeneratedText.fromMap(maps[i]));
  }

  Future<int> deleteGeneratedText(int id) async {
    final db = await database;
    return await db.delete(
      'generated_texts',
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<int> clearGeneratedTexts() async {
    final db = await database;
    return await db.delete('generated_texts');
  }

  // =========================================================================
  // Learning sessions operations
  // =========================================================================

  Future<int> insertSession(LearningSession session) async {
    final db = await database;
    return await db.insert('learning_sessions', session.toMap());
  }

  Future<List<LearningSession>> getSessions({int limit = 30}) async {
    final db = await database;
    final List<Map<String, dynamic>> maps = await db.query(
      'learning_sessions',
      orderBy: 'date DESC',
      limit: limit,
    );
    return List.generate(maps.length, (i) => LearningSession.fromMap(maps[i]));
  }

  Future<LearningSession?> getTodaySession() async {
    final db = await database;
    final today = DateTime.now().toIso8601String().split('T')[0];
    final List<Map<String, dynamic>> maps = await db.query(
      'learning_sessions',
      where: 'date LIKE ?',
      whereArgs: ['$today%'],
      limit: 1,
    );

    if (maps.isEmpty) return null;
    return LearningSession.fromMap(maps.first);
  }

  // =========================================================================
  // Settings operations
  // =========================================================================

  Future<void> setSetting(String key, String value) async {
    final db = await database;
    await db.insert(
      'user_settings',
      {'key': key, 'value': value},
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  Future<String?> getSetting(String key) async {
    final db = await database;
    final List<Map<String, dynamic>> maps = await db.query(
      'user_settings',
      where: 'key = ?',
      whereArgs: [key],
      limit: 1,
    );

    if (maps.isEmpty) return null;
    return maps.first['value'] as String;
  }

  // =========================================================================
  // Utility operations
  // =========================================================================

  Future<void> clearAllData() async {
    final db = await database;
    await db.delete('vocabulary');
    await db.delete('generated_texts');
    await db.delete('learning_sessions');
    await db.delete('user_settings');
  }

  Future<void> close() async {
    final db = await database;
    await db.close();
  }
}
