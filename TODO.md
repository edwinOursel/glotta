# Glotta — Audit Todo List

## CRITICAL

- [x] `core/auth.py:25` — SECRET_KEY : crasher au startup si absent (actuellement fallback sur valeur hardcodée)
- [x] `core/api_server.py:268,282` — Path traversal sur `/api/vocabulary/save` et `/load` (filename non validé)
- [x] `core/routers/*.py` — Stack traces exposées dans les réponses d'erreur (detail=str(e))
- [ ] `core/agentic_graph.py` — Clé OpenAI non validée au startup

## HIGH

- [x] `core/routers/auth.py` — Pas de rate limiting (brute force sur login/register)
- [x] `core/routers/` (génération) — Pas de rate limiting sur les appels LLM (coûteux)
- [ ] `core/routers/users.py:37` — ORM object retourné après fermeture de session (LazyLoad crash)
- [x] `core/api_server.py:200` — Prompt injection via `system_prompt` fourni par l'utilisateur
- [ ] `core/database.py` — Pas de migrations Alembic (create_all() en prod = pas de rollback)
- [ ] `mobile/lib/services/api_service.dart:18-20` — URL API hardcodée (`10.0.2.2:8000`, Android emulator only)
- [ ] `mobile/lib/services/api_service.dart` — Pas de timeout HTTP ni de retry
- [ ] `mobile/lib/providers/review_provider.dart:128` — submitRating() fire-and-forget (rating perdu si réseau fail)
- [ ] `core/routers/` — Tous les codes d'erreur mappés sur 500 (pas de 400/403/404)
- [ ] `core/` — print() partout, pas de logging structuré

## MEDIUM — Architecture

- [ ] `core/models.py` — Index manquants sur FK (composite index sur status+requester_id, status+addressee_id)
- [ ] `core/routers/vocabulary.py:74` — LIKE wildcard injection (%, _ non échappés dans search)
- [ ] `core/routers/sessions.py:114` — Calcul du streak en mémoire (fetch toutes les dates)
- [ ] `core/routers/friends.py:35` — Duplication du calcul de streak (même logique que sessions.py)
- [ ] `core/routers/challenges.py:36` — Score recalculé live à chaque vue
- [ ] `core/auth.py` — Pas de blacklist/rotation des tokens (logout ne révoque pas les JWT)
- [ ] `core/auth.py` — Pas de rate limiting sur refresh token
- [ ] `mobile/lib/providers/` — StateNotifier (Riverpod v1) → migrer vers AsyncNotifierProvider (v2)
- [ ] `mobile/lib/providers/vocabulary_provider.dart:105` — Double appel API sur filterByLevel()
- [ ] `mobile/lib/providers/gamification_provider.dart` — Trophées hardcodés côté client
- [ ] `mobile/lib/providers/gamification_provider.dart:31` — Streak/trophées en SharedPreferences non chiffré
- [ ] `mobile/lib/models/` — Pas de == / hashCode sur les modèles Dart
- [ ] `mobile/lib/` — Pas de gestion offline
- [ ] `core/schemas.py` — Pas de max_length sur les champs string (username, word, notes, etc.)
- [ ] `core/database.py:12` — DATABASE_URL avec fallback SQLite hardcodé dans le cwd

## LOW

- [ ] `core/models.py` — datetime.utcnow() deprecated (Python 3.12+) → datetime.now(UTC)
- [ ] `mobile/lib/l10n/app_localizations.dart` — i18n manuel → migrer vers gen_l10n + .arb
- [ ] `mobile/lib/services/api_service.dart` — Pas de cert pinning
- [ ] `core/srs.py:67` — mastery_level = repetition // 2 sans CheckConstraint DB
- [ ] `mobile/lib/providers/auth_provider.dart` — Pas de timeout sur login/register/restoreSession

## Infrastructure manquante

- [ ] Tests backend (couverture <5%) — priorité : auth, SRS, friends
- [ ] Tests mobile (zéro widget tests)
- [ ] CI/CD backend (seulement Flutter web en CI)
- [ ] Alembic (voir HIGH ci-dessus)
- [ ] Logging structuré (voir HIGH ci-dessus)
