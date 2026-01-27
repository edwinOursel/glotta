# 📱 Glotta Mobile - Flutter App

Cross-platform mobile application for learning Japanese with constrained LLM generation.

## 🚀 Getting Started

### Prerequisites

- Flutter SDK 3.0.0 or higher
- Dart 3.0.0 or higher
- Android Studio / Xcode (for mobile development)

### Installation

```bash
# Navigate to mobile directory
cd mobile

# Get dependencies
flutter pub get

# Run on connected device/emulator
flutter run
```

### Development

```bash
# Run with hot reload
flutter run

# Run tests
flutter test

# Build for production
flutter build apk        # Android
flutter build ipa        # iOS
flutter build web        # Web
```

## 🏗️ Architecture

```
mobile/
├── lib/
│   ├── main.dart                    # Entry point
│   ├── models/                      # Data models
│   │   ├── user_vocabulary.dart
│   │   ├── learning_session.dart
│   │   └── generated_text.dart
│   ├── providers/                   # Riverpod providers (state management)
│   │   ├── vocabulary_provider.dart
│   │   ├── api_provider.dart
│   │   └── settings_provider.dart
│   ├── services/                    # Business logic
│   │   ├── api_service.dart         # Backend communication
│   │   ├── database_service.dart    # Local storage
│   │   └── auth_service.dart        # Authentication
│   ├── screens/                     # UI screens
│   │   ├── learn/
│   │   ├── vocabulary/
│   │   ├── progress/
│   │   └── settings/
│   ├── widgets/                     # Reusable widgets
│   │   ├── word_card.dart
│   │   ├── generated_text_view.dart
│   │   └── progress_chart.dart
│   └── utils/                       # Utilities
│       ├── constants.dart
│       └── helpers.dart
├── test/                            # Tests
├── assets/                          # Images, fonts, etc.
└── pubspec.yaml                     # Dependencies
```

## 🎨 Screens

### 1. Learn (学習)
- Main learning interface
- Generate Japanese text based on user's vocabulary
- Interactive reading with translations
- Practice mode

### 2. Vocabulary (単語帳)
- Manage known words
- Add/remove words
- Browse by JLPT level
- Search and filter
- Import/export vocabulary lists

### 3. Progress (進捗)
- Learning statistics
- Vocabulary growth chart
- Time spent learning
- Streak tracking

### 4. Settings (設定)
- Target language selection
- JLPT level preference
- Constraint mode (hard/soft/adaptive)
- Theme settings
- Backend API configuration

## 🔧 State Management

Using **Riverpod** for state management:

```dart
// Example: Vocabulary Provider
final vocabularyProvider = StateNotifierProvider<VocabularyNotifier, VocabularyState>((ref) {
  return VocabularyNotifier(ref.read(databaseServiceProvider));
});

// Usage in widgets
class VocabularyPage extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final vocabulary = ref.watch(vocabularyProvider);
    // ...
  }
}
```

## 🌐 API Integration

The app communicates with the Python backend via REST API:

```dart
// Example: Generate text
final apiService = ApiService(baseUrl: 'http://localhost:8000');

final result = await apiService.generateText(
  prompt: '今日は',
  maxLength: 50,
  useConstraints: true,
);
```

### API Endpoints (to be implemented in backend)

- `POST /api/generate` - Generate text with vocabulary constraints
- `GET /api/vocabulary` - Get user's vocabulary
- `POST /api/vocabulary/words` - Add words to vocabulary
- `DELETE /api/vocabulary/words/{word}` - Remove word
- `GET /api/jlpt/{level}` - Get JLPT vocabulary list
- `GET /api/user/stats` - Get user statistics

## 💾 Local Storage

Using **SQLite** for local data persistence:

```sql
-- vocabulary table
CREATE TABLE vocabulary (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  word TEXT NOT NULL UNIQUE,
  reading TEXT,
  meaning TEXT,
  jlpt_level TEXT,
  date_added DATETIME DEFAULT CURRENT_TIMESTAMP,
  times_seen INTEGER DEFAULT 0,
  mastery_level INTEGER DEFAULT 0
);

-- learning_sessions table
CREATE TABLE learning_sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date DATETIME DEFAULT CURRENT_TIMESTAMP,
  duration INTEGER,  -- seconds
  words_practiced INTEGER,
  constraint_mode TEXT
);
```

## 🎯 Features Roadmap

- [x] Basic app scaffold with navigation
- [ ] Vocabulary management UI
  - [ ] Add/edit/delete words
  - [ ] Search and filter
  - [ ] JLPT level categorization
  - [ ] Import from CSV/JSON
- [ ] Text generation UI
  - [ ] Prompt input
  - [ ] Generated text display
  - [ ] Word highlighting
  - [ ] Translation on tap
- [ ] API integration
  - [ ] Connect to Python backend
  - [ ] Error handling
  - [ ] Loading states
  - [ ] Offline mode
- [ ] Local database
  - [ ] SQLite setup
  - [ ] Vocabulary CRUD operations
  - [ ] Session tracking
- [ ] Progress tracking
  - [ ] Statistics dashboard
  - [ ] Charts and graphs
  - [ ] Streak counter
- [ ] User authentication
  - [ ] Login/signup
  - [ ] Profile management
  - [ ] Cloud sync
- [ ] Advanced features
  - [ ] Grammar-focused learning
  - [ ] Conversation mode
  - [ ] Speech recognition
  - [ ] Text-to-speech
  - [ ] Flashcard mode
  - [ ] Spaced repetition

## 🧪 Testing

```bash
# Run all tests
flutter test

# Run with coverage
flutter test --coverage

# View coverage report
genhtml coverage/lcov.info -o coverage/html
open coverage/html/index.html
```

## 📦 Dependencies

### Core
- `flutter_riverpod`: State management
- `provider`: Alternative state management

### Networking
- `http`: HTTP client
- `dio`: Advanced HTTP client with interceptors

### Storage
- `sqflite`: SQLite database
- `shared_preferences`: Key-value storage
- `path_provider`: File system paths

### UI
- `google_fonts`: Custom fonts
- `flutter_svg`: SVG support

### Utilities
- `intl`: Internationalization
- `logger`: Logging
- `json_annotation` + `json_serializable`: JSON serialization

## 🎨 Design System

Following Material Design 3 with custom theming for Japanese language learning:

**Color Palette:**
- Primary: Indigo (学習の色)
- Secondary: Teal (進歩の色)
- Accent: Amber (成功の色)

**Typography:**
- Headers: System default
- Body: Noto Sans JP (for Japanese text)
- Code: Roboto Mono

## 🌍 Internationalization

Support for multiple UI languages:
- English
- Japanese (日本語)
- French (Français)

```dart
// Example
Text(AppLocalizations.of(context).learnTab)
```

## 🔐 Security

- API keys stored in secure storage (flutter_secure_storage)
- User data encrypted locally
- HTTPS only for API communication
- No sensitive data in logs

## 📱 Platform Support

- ✅ Android 5.0+ (API 21+)
- ✅ iOS 12.0+
- ⚠️ Web (limited functionality, no offline mode)
- ⚠️ Desktop (experimental)

## 🤝 Contributing

When adding new features:
1. Create feature branch
2. Follow Flutter/Dart style guide
3. Add tests
4. Update documentation
5. Submit PR

## 📄 License

MIT License

---

**Note:** This app is in active development. Many features are placeholders and will be implemented progressively.
