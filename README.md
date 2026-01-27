# 🗣️ Glotta - Language Learning with Constrained LLM

Glotta is a language learning system (focus: Japanese) that uses Large Language Models (LLMs) with vocabulary constraints. The concept: **force the model to generate text only using words and grammatical structures you already know**.

## 🏗️ Monorepo Structure

This is a monorepo containing:

```
glotta/
├── core/          # Python backend - LLM with vocabulary constraints
├── mobile/        # Flutter mobile app - User interface
└── README.md      # This file
```

### Core (Python Backend)

The core logic for constraining LLM generation based on user vocabulary.

**Key features:**
- Manipulates logits (output probabilities) before token sampling
- Three constraint modes: hard, soft, adaptive
- User vocabulary management with JSON persistence
- Support for pre-trained Japanese LLM models (rinna/japanese-gpt2)

[📖 See core/README.md for details](./core/README.md)

### Mobile (Flutter App)

Cross-platform application for learning Japanese - runs on Android, iOS, and **web browsers**.

**Key features:**
- Vocabulary management (add/remove words)
- Interactive text generation
- Progress tracking
- JLPT level support
- **Web version** (no Flutter SDK needed!)
- **CI/CD** with GitHub Actions (auto-build & deploy)
- **Termux-friendly** (develop on Android)

[📱 See mobile/README.md for details](./mobile/README.md)
[🤖 Termux Guide](./mobile/TERMUX_GUIDE.md) - Develop on Android!

## 🚀 Quick Start

### Option 1: Termux / Android (No Flutter SDK needed!)

```bash
# Backend
cd core
uv pip install -e .
uv run python api_server.py

# Frontend (auto-built by GitHub Actions)
cd mobile
python download_build.py  # Downloads latest build
python serve_web.py       # Serves on http://localhost:8080
```

📖 Full guide: [mobile/TERMUX_GUIDE.md](./mobile/TERMUX_GUIDE.md)

### Option 2: With Flutter SDK

```bash
# Backend
cd core
uv pip install -e .
python demo.py

# Mobile app
cd mobile
flutter pub get
flutter run  # or: flutter run -d chrome
```

## 💡 How It Works

```
User Profile           Mobile App (Flutter)
(Known vocabulary)            ↓
        ↓              REST API / gRPC
        ↓                     ↓
    Core (Python)      Logits Processor
        ↓              (Vocabulary constraints)
        ↓                     ↓
Japanese LLM Model → Constrained Generation
                            ↓
                    Text adapted to user level
```

## 🎯 Project Goals

1. **Personalized learning**: Generate content adapted to each user's level
2. **Active practice**: Interactive conversations in Japanese
3. **Progressive difficulty**: Automatically increase complexity as user improves
4. **Grammar focus**: Practice specific grammatical patterns
5. **Mobile-first**: Learn anywhere, anytime

## 🛠️ Tech Stack

**Backend:**
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- FastAPI (for mobile API)

**Mobile:**
- Flutter 3.x
- Dart
- Provider/Riverpod (state management)
- SQLite (local vocabulary storage)

## 📖 Documentation

- [Core Backend Documentation](./core/README.md)
- [Mobile App Documentation](./mobile/README.md)
- [API Documentation](./docs/API.md) _(coming soon)_
- [Architecture Decisions](./docs/ARCHITECTURE.md) _(coming soon)_

## 🗺️ Roadmap

**Core (Backend):**
- [x] Logits manipulation for vocabulary constraints
- [x] Three constraint modes (hard/soft/adaptive)
- [x] User vocabulary management
- [x] FastAPI backend with REST endpoints
- [x] uv-based dependency management
- [ ] JLPT vocabulary database integration
- [ ] Grammar-guided generation
- [ ] Conversation mode

**Mobile (Frontend):**
- [x] Basic Flutter app scaffold
- [x] Flutter web support
- [x] PWA (Progressive Web App) capabilities
- [x] CI/CD with GitHub Actions
- [x] Termux compatibility (develop on Android!)
- [ ] Vocabulary management UI
- [ ] Text generation UI
- [ ] User authentication
- [ ] Progress tracking
- [ ] Offline mode with local models

**Integration:**
- [ ] Backend ↔ Mobile API integration
- [ ] Real-time text generation
- [ ] User profiles with cloud sync

**Deployment:**
- [x] GitHub Pages (web app)
- [ ] Backend to cloud (Railway/Fly.io)
- [ ] Mobile apps to stores (Play Store/App Store)

## 🤝 Contributing

This is an experimental project in active development. Contributions are welcome!

## 📝 License

MIT License

---

**頑張ってください！** (Ganbatte kudasai! - Good luck!)
