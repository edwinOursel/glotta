import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import '../models/user_profile.dart';
import '../services/api_service.dart';
import 'generation_provider.dart' show apiServiceProvider;

// ── Storage keys ──────────────────────────────────────────────────────────────

const _kAccessToken  = 'access_token';
const _kRefreshToken = 'refresh_token';

// ── State ─────────────────────────────────────────────────────────────────────

class AuthState {
  final UserProfile? user;
  final bool isLoading;
  final String? error;

  const AuthState({this.user, this.isLoading = false, this.error});

  bool get isAuthenticated => user != null;

  AuthState copyWith({
    UserProfile? user,
    bool? isLoading,
    String? error,
    bool clearUser = false,
    bool clearError = false,
  }) =>
      AuthState(
        user:      clearUser  ? null : user ?? this.user,
        isLoading: isLoading  ?? this.isLoading,
        error:     clearError ? null : error ?? this.error,
      );
}

// ── Notifier ──────────────────────────────────────────────────────────────────

class AuthNotifier extends StateNotifier<AuthState> {
  AuthNotifier(this._api, this._storage) : super(const AuthState());

  final ApiService _api;
  final FlutterSecureStorage _storage;

  // ── Boot: restore session from secure storage ─────────────────────────────

  Future<void> restoreSession() async {
    final token = await _storage.read(key: _kAccessToken);
    if (token == null) return;

    state = state.copyWith(isLoading: true);
    try {
      final profile = await _api.getProfile(accessToken: token);
      state = AuthState(user: profile);
    } catch (_) {
      // Token expired — try refresh
      await _tryRefresh();
    }
  }

  Future<void> _tryRefresh() async {
    final refresh = await _storage.read(key: _kRefreshToken);
    if (refresh == null) {
      state = const AuthState();
      return;
    }
    try {
      final newToken = await _api.refreshToken(refreshToken: refresh);
      await _storage.write(key: _kAccessToken, value: newToken);
      final profile = await _api.getProfile(accessToken: newToken);
      state = AuthState(user: profile);
    } catch (_) {
      await _storage.deleteAll();
      state = const AuthState();
    }
  }

  // ── Register ──────────────────────────────────────────────────────────────

  Future<void> register({
    required String email,
    required String password,
    String? username,
  }) async {
    state = state.copyWith(isLoading: true, clearError: true);
    try {
      final tokens = await _api.register(
        email: email,
        password: password,
        username: username,
      );
      await _persistTokens(tokens['access_token']!, tokens['refresh_token']!);
      final profile = await _api.getProfile(accessToken: tokens['access_token']!);
      state = AuthState(user: profile);
    } catch (e) {
      state = state.copyWith(isLoading: false, error: _message(e));
    }
  }

  // ── Login ─────────────────────────────────────────────────────────────────

  Future<void> login({
    required String email,
    required String password,
  }) async {
    state = state.copyWith(isLoading: true, clearError: true);
    try {
      final tokens = await _api.login(email: email, password: password);
      await _persistTokens(tokens['access_token']!, tokens['refresh_token']!);
      final profile = await _api.getProfile(accessToken: tokens['access_token']!);
      state = AuthState(user: profile);
    } catch (e) {
      state = state.copyWith(isLoading: false, error: _message(e));
    }
  }

  // ── Logout ────────────────────────────────────────────────────────────────

  Future<void> logout() async {
    await _storage.deleteAll();
    state = const AuthState();
  }

  // ── Update profile ────────────────────────────────────────────────────────

  Future<void> updateProfile({
    String? username,
    String? jlptLevel,
    String? constraintMode,
  }) async {
    final token = await _storage.read(key: _kAccessToken);
    if (token == null) return;

    try {
      final updated = await _api.updateProfile(
        accessToken:    token,
        username:       username,
        jlptLevel:      jlptLevel,
        constraintMode: constraintMode,
      );
      state = state.copyWith(user: updated);
    } catch (e) {
      state = state.copyWith(error: _message(e));
    }
  }

  // ── Token access (for ApiService calls in other providers) ────────────────

  Future<String?> getAccessToken() => _storage.read(key: _kAccessToken);

  // ── Helpers ───────────────────────────────────────────────────────────────

  Future<void> _persistTokens(String access, String refresh) async {
    await Future.wait([
      _storage.write(key: _kAccessToken,  value: access),
      _storage.write(key: _kRefreshToken, value: refresh),
    ]);
  }

  String _message(Object e) =>
      e is ApiException ? e.message : e.toString();
}

// ── Providers ─────────────────────────────────────────────────────────────────

final _secureStorage = const FlutterSecureStorage(
  aOptions: AndroidOptions(encryptedSharedPreferences: true),
);

final authProvider = StateNotifierProvider<AuthNotifier, AuthState>((ref) {
  final api = ref.watch(apiServiceProvider);
  return AuthNotifier(api, _secureStorage);
});

/// Convenience: expose the current user's access token to other providers.
final accessTokenProvider = FutureProvider<String?>((ref) {
  return ref.watch(authProvider.notifier).getAccessToken();
});
