# 🚀 Glotta - Quick Start Guide

Get up and running with Glotta in 5 minutes!

## Prerequisites

- **Python 3.8+** with uv (or pip)
- **Flutter 3.0+** (for mobile development)
- **Git**

**Install uv (recommended):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Option 1: Backend Only (Quick Test)

Try the core functionality without the mobile app:

```bash
# 1. Clone the repo
git clone https://github.com/edwinOursel/glotta.git
cd glotta

# 2. Install Python dependencies
cd core
uv pip install -e .
# or with classic pip: pip install -e .

# 3. Run a demo
uv run python demo.py 1
# or: python demo.py 1

# Or try interactive mode
uv run python demo.py 4
```

## Option 2: Backend + API Server

Run the backend with API server for mobile integration:

```bash
# 1. Navigate to core
cd core

# 2. Install dependencies (if not already done)
uv pip install -e .

# 3. Start the API server
uv run python api_server.py
# or: python api_server.py

# Server will run at http://localhost:8000
# API docs available at http://localhost:8000/docs
```

**Test the API:**

```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "私は",
    "max_length": 30,
    "use_constraints": true,
    "constraint_mode": "hard"
  }'
```

## Option 3: Full Stack (Backend + Mobile)

### Step 1: Start the Backend

```bash
# Terminal 1: Start API server
cd core
uv run python api_server.py
```

### Step 2: Run the Mobile App

**If Flutter is not installed yet:**

```bash
# Install Flutter (example for macOS)
brew install flutter

# Or download from: https://flutter.dev/docs/get-started/install
```

**Run the app:**

```bash
# Terminal 2: Start mobile app
cd mobile

# First time only: Get dependencies
flutter pub get

# Run on connected device/emulator
flutter run

# Or run on specific device
flutter devices          # List available devices
flutter run -d chrome    # Run on Chrome
flutter run -d macos     # Run on macOS desktop
```

## What to Expect

### Backend Demo

When you run `python demo.py 1`, you'll see:
- Text generation **without** constraints (full model capabilities)
- Text generation **with** constraints (only your known vocabulary)
- Comparison of the two modes

Example output:
```
📝 Prompt: '私は'
   🔓 Sans contraintes:
      私はこの問題について深く考察し、複雑な哲学的観点から...
   🔒 Avec contraintes (hard):
      私は猫が好きです。犬も好きです。
```

### API Server

When you start the API server:
- FastAPI server runs on `http://localhost:8000`
- Interactive API docs at `http://localhost:8000/docs`
- Model loads automatically on first request (~30 seconds)
- Subsequent requests are fast

### Mobile App

When you run the mobile app:
- Navigation bar with 4 tabs: Learn, Vocabulary, Progress, Settings
- Basic UI structure (most features coming soon)
- Can connect to local API server
- Material Design 3 with Japanese labels

## Troubleshooting

### Python: `ModuleNotFoundError`
```bash
uv pip install -e .
# or: pip install -e .
```

### Python: Model download fails
The first run downloads ~500MB model from Hugging Face. Ensure:
- Stable internet connection
- Sufficient disk space (~2GB free)
- Not behind restrictive firewall

### Flutter: `flutter: command not found`
Install Flutter SDK: https://flutter.dev/docs/get-started/install

### Flutter: `Pub get failed`
```bash
flutter clean
flutter pub get
```

### API: Connection refused from mobile
- Ensure backend is running: `curl http://localhost:8000/health`
- On iOS simulator: use `http://localhost:8000`
- On Android emulator: use `http://10.0.2.2:8000`
- On physical device: use your computer's IP address

## Next Steps

1. **Add your vocabulary**: Edit `core/vocabulary_example.json`
2. **Try different modes**: hard, soft, adaptive
3. **Explore the API**: Open http://localhost:8000/docs
4. **Customize the mobile app**: Edit `mobile/lib/main.dart`

## Need Help?

- Read the full docs: [README.md](./README.md)
- Backend details: [core/README.md](./core/README.md)
- Mobile details: [mobile/README.md](./mobile/README.md)
- Open an issue: https://github.com/edwinOursel/glotta/issues

---

**頑張ってください！** (Good luck with your Japanese learning!)
