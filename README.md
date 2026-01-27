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

Cross-platform mobile application for learning Japanese.

**Key features:**
- Vocabulary management (add/remove words)
- Interactive text generation
- Progress tracking
- JLPT level support
- Offline mode (with downloaded models)

[📱 See mobile/README.md for details](./mobile/README.md)

## 🚀 Quick Start

### Backend (Core)

```bash
cd core
pip install -r requirements.txt
python demo.py
```

### Mobile App

```bash
cd mobile
flutter pub get
flutter run
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

- [x] Core: Logits manipulation for vocabulary constraints
- [x] Core: Three constraint modes (hard/soft/adaptive)
- [x] Core: User vocabulary management
- [ ] Core: FastAPI backend with REST endpoints
- [ ] Mobile: Basic Flutter app scaffold
- [ ] Mobile: Vocabulary management UI
- [ ] Mobile: Text generation UI
- [ ] Mobile: User authentication
- [ ] Mobile: Progress tracking
- [ ] Integration: Backend ↔ Mobile API
- [ ] Features: JLPT level integration
- [ ] Features: Grammar-guided generation
- [ ] Features: Conversation mode
- [ ] Deploy: Backend to cloud
- [ ] Deploy: Mobile apps to stores

## 🤝 Contributing

This is an experimental project in active development. Contributions are welcome!

## 📝 License

MIT License

---

**頑張ってください！** (Ganbatte kudasai! - Good luck!)
