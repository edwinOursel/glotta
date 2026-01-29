# 📱 Guide Termux - Utiliser Glotta sur Android

Guide complet pour développer et utiliser Glotta sur Android avec Termux, sans avoir besoin de Flutter SDK.

## 🎯 Workflow complet

```
PC (optionnel)                GitHub Actions              Termux (Android)
     │                              │                            │
     │  git push                    │                            │
     ├─────────────────────────────>│                            │
     │                              │ Build Flutter web          │
     │                              │ Publish to GitHub Pages    │
     │                              │ Create artifact            │
     │                              │                            │
     │                              │    python download_build   │
     │                              │<───────────────────────────│
     │                              │                            │
     │                              │    artifact ZIP            │
     │                              ├───────────────────────────>│
     │                              │                            │
     │                              │                   Extract & serve
     │                              │                   python serve_web.py
     │                              │                   http://localhost:8080
```

## 🚀 Installation initiale (une seule fois)

### 1. Installer Termux et dépendances

```bash
# Dans Termux
pkg update && pkg upgrade
pkg install python git
```

### 2. Installer uv (gestionnaire Python moderne)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 3. Cloner le repo

```bash
cd ~
git clone https://github.com/edwinOursel/glotta.git
cd glotta
```

### 4. Installer les dépendances Python

```bash
# Backend
cd core
uv pip install -e .

# Mobile (pour les scripts Python)
cd ../mobile
uv pip install requests
# Ou si vous voulez tout installer depuis pyproject.toml:
# uv pip install -r pyproject.toml
```

## 📦 Télécharger la dernière build web

### Option A : Script automatique (recommandé)

```bash
cd mobile
python download_build.py
```

**Si erreur "requests module not found" :**
```bash
# Installer juste requests :
uv pip install requests
# ou avec pip classique:
pip install requests
```

Le script va :
1. ✅ Trouver la dernière build réussie sur GitHub Actions
2. ✅ Télécharger l'artifact (fichier ZIP)
3. ✅ Extraire dans `mobile/build/web/`
4. ✅ Afficher les infos de build

**Note pour repos privés :**
```bash
# Créer un token sur: https://github.com/settings/tokens
# Scope nécessaire: 'actions:read' ou 'repo'
python download_build.py --token ghp_votre_token_ici
```

### Option B : Télécharger manuellement

1. Aller sur : https://github.com/edwinOursel/glotta/actions
2. Cliquer sur la dernière build réussie (✓)
3. Scroller en bas → "Artifacts"
4. Télécharger `flutter-web-build`
5. Extraire dans `mobile/build/web/`

### Option C : Utiliser GitHub Pages (le plus simple!)

Si la CI/CD est configurée, l'app est automatiquement déployée sur :
```
https://edwinoursel.github.io/glotta/
```

Pas besoin de télécharger, juste ouvrir dans le navigateur ! 🎉

## 🖥️ Lancer l'application

### Backend (API Python)

Terminal 1 :
```bash
cd ~/glotta/core
uv run python api_server.py

# API disponible sur: http://localhost:8000
# Documentation: http://localhost:8000/docs
```

### Frontend (App web)

Terminal 2 :
```bash
cd ~/glotta/mobile

# Option 1: Servir la build locale
python serve_web.py
# App disponible sur: http://localhost:8080

# Option 2: Utiliser GitHub Pages directement
# Ouvrir: https://edwinoursel.github.io/glotta/
```

### Ouvrir dans le navigateur

Sur votre téléphone Android, ouvrir :
- **Build locale** : `http://localhost:8080`
- **GitHub Pages** : `https://edwinoursel.github.io/glotta/`

## 🔄 Mettre à jour l'application

### Méthode 1 : Pull + Re-télécharger

```bash
cd ~/glotta
git pull origin claude/mobile-app-flutter-dEUYr

# Backend
cd core
uv pip install -e .  # Si nouvelles dépendances

# Frontend
cd ../mobile
python download_build.py  # Télécharge la dernière build
```

### Méthode 2 : Utiliser GitHub Pages

Rien à faire ! L'app sur GitHub Pages est automatiquement mise à jour.

## 💡 Workflow quotidien

### Scénario 1 : Juste utiliser l'app

```bash
# Backend (si besoin de l'API)
cd ~/glotta/core
uv run python api_server.py

# Puis ouvrir dans le navigateur:
# https://edwinoursel.github.io/glotta/
```

### Scénario 2 : Développer le backend

```bash
# Éditer des fichiers Python dans Termux
cd ~/glotta/core
nano api_server.py  # ou vim, micro, etc.

# Tester
uv run python api_server.py

# Commit
git add .
git commit -m "Update backend"
git push
```

### Scénario 3 : Développer le frontend

**Sur PC :**
```bash
cd mobile
# Éditer lib/main.dart
flutter run -d chrome  # Tester
git push
# → GitHub Actions build automatiquement
```

**Sur Termux :**
```bash
# Attendre le build GitHub Actions (2-3 minutes)
cd ~/glotta/mobile
python download_build.py  # Télécharger nouvelle version
python serve_web.py       # Tester
```

## 🔧 Trucs et astuces

### Accéder depuis d'autres appareils

Si vous voulez tester sur un autre téléphone/PC sur le même WiFi :

```bash
# Trouver votre IP locale
ifconfig  # ou ip addr

# Lancer le serveur
python serve_web.py 8080

# Sur autre appareil, ouvrir:
# http://192.168.X.X:8080
```

### Background processes (avec tmux)

```bash
# Installer tmux
pkg install tmux

# Session backend
tmux new -s backend
cd ~/glotta/core
uv run python api_server.py
# Ctrl+B puis D pour détacher

# Session frontend
tmux new -s frontend
cd ~/glotta/mobile
python serve_web.py
# Ctrl+B puis D pour détacher

# Réattacher
tmux attach -t backend
tmux attach -t frontend

# Lister les sessions
tmux ls
```

### Vérifier les builds disponibles

```bash
# Voir les dernières builds sur GitHub
curl -s "https://api.github.com/repos/edwinOursel/glotta/actions/runs?status=success&per_page=5" \
  | grep '"conclusion": "success"' | wc -l
```

### Debug mode

Si l'app ne fonctionne pas :

```bash
# Backend : vérifier les logs
cd ~/glotta/core
uv run python api_server.py
# Regarder les erreurs dans le terminal

# Frontend : ouvrir DevTools dans le navigateur
# Chrome: Menu → Plus d'outils → Outils de développement
# Regarder l'onglet Console pour les erreurs
```

## 🐛 Problèmes courants

### `python download_build.py` échoue

**Problème : "No successful builds found"**
```bash
# Vérifier manuellement sur:
# https://github.com/edwinOursel/glotta/actions

# Si pas de builds, déclencher manuellement:
# 1. Aller sur GitHub Actions
# 2. Sélectionner "Build Flutter Web"
# 3. Cliquer "Run workflow"
```

**Problème : "Authentication failed"**
```bash
# Repo privé ? Créer un token:
# https://github.com/settings/tokens
# Scope: actions:read

python download_build.py --token ghp_xxxxx
```

### Port déjà utilisé

```bash
# Trouver le processus
lsof -i :8080  # ou netstat -tulpn | grep 8080

# Tuer le processus
kill -9 <PID>

# Ou utiliser un autre port
python serve_web.py 8081
```

### API CORS errors

Si le frontend ne peut pas contacter le backend :

```python
# Dans core/api_server.py, vérifier:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autoriser toutes les origines
    # ...
)
```

## 📚 Resources utiles

- **GitHub Actions** : https://github.com/edwinOursel/glotta/actions
- **GitHub Pages** : https://edwinoursel.github.io/glotta/
- **API Docs** : http://localhost:8000/docs (quand backend lancé)
- **Termux Wiki** : https://wiki.termux.com/

## 🎓 Aller plus loin

### Configurer GitHub Token (permanent)

```bash
# Dans Termux
echo 'export GITHUB_TOKEN=ghp_votre_token' >> ~/.bashrc
source ~/.bashrc

# Maintenant le script l'utilise automatiquement
python download_build.py
```

### Auto-update script

```bash
#!/bin/bash
# ~/glotta/update.sh

cd ~/glotta
git pull
cd core && uv pip install -e .
cd ../mobile && python download_build.py
echo "✅ Glotta updated!"
```

```bash
chmod +x ~/glotta/update.sh
~/glotta/update.sh
```

### Raccourcis pratiques

```bash
# Dans ~/.bashrc
alias glotta-backend='cd ~/glotta/core && uv run python api_server.py'
alias glotta-frontend='cd ~/glotta/mobile && python serve_web.py'
alias glotta-update='cd ~/glotta && git pull && cd mobile && python download_build.py'

source ~/.bashrc

# Utilisation
glotta-backend    # Lance le backend
glotta-frontend   # Lance le frontend
glotta-update     # Met à jour
```

---

**Bon apprentissage du japonais depuis votre téléphone Android ! 📱🗾**
