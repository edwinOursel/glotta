import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../l10n/app_localizations.dart';
import '../../providers/auth_provider.dart';
import 'login_screen.dart';

class RegisterScreen extends ConsumerStatefulWidget {
  const RegisterScreen({super.key});

  @override
  ConsumerState<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends ConsumerState<RegisterScreen> {
  final _formKey         = GlobalKey<FormState>();
  final _emailCtrl       = TextEditingController();
  final _usernameCtrl    = TextEditingController();
  final _passwordCtrl    = TextEditingController();
  final _confirmCtrl     = TextEditingController();
  bool  _obscurePassword = true;

  String _selectedLevel = 'N5';
  static const _levels  = ['N5', 'N4', 'N3', 'N2', 'N1'];

  @override
  void dispose() {
    _emailCtrl.dispose();
    _usernameCtrl.dispose();
    _passwordCtrl.dispose();
    _confirmCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    await ref.read(authProvider.notifier).register(
      email:    _emailCtrl.text.trim(),
      password: _passwordCtrl.text,
      username: _usernameCtrl.text.trim().isNotEmpty
          ? _usernameCtrl.text.trim()
          : null,
    );
    // After registration, set the chosen JLPT level
    if (ref.read(authProvider).isAuthenticated) {
      await ref.read(authProvider.notifier).updateProfile(
        jlptLevel: _selectedLevel,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final l      = AppLocalizations.of(context);
    final auth   = ref.watch(authProvider);
    final theme  = Theme.of(context);
    final colors = theme.colorScheme;

    ref.listen<AuthState>(authProvider, (_, next) {
      if (next.error != null) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(next.error!), backgroundColor: colors.error),
        );
      }
    });

    return Scaffold(
      appBar: AppBar(
        title: Text(l.authCreateAccount),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pushReplacement(
            MaterialPageRoute(builder: (_) => const LoginScreen()),
          ),
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 24),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // ── Email ─────────────────────────────────────────────────
                TextFormField(
                  controller:      _emailCtrl,
                  keyboardType:    TextInputType.emailAddress,
                  textInputAction: TextInputAction.next,
                  decoration: InputDecoration(
                    labelText:  l.authEmail,
                    prefixIcon: const Icon(Icons.email_outlined),
                    border:     const OutlineInputBorder(),
                  ),
                  validator: (v) =>
                      v == null || !v.contains('@') ? l.validEmail : null,
                ),
                const SizedBox(height: 16),

                // ── Username (optional) ───────────────────────────────────
                TextFormField(
                  controller:      _usernameCtrl,
                  textInputAction: TextInputAction.next,
                  decoration: InputDecoration(
                    labelText:   l.authUsernameOptional,
                    prefixIcon:  const Icon(Icons.person_outline),
                    border:      const OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 16),

                // ── JLPT level ────────────────────────────────────────────
                DropdownButtonFormField<String>(
                  value:       _selectedLevel,
                  decoration:  InputDecoration(
                    labelText:  l.authStartingLevel,
                    prefixIcon: const Icon(Icons.school_outlined),
                    border:     const OutlineInputBorder(),
                  ),
                  items: _levels
                      .map((lvl) =>
                          DropdownMenuItem(value: lvl, child: Text(lvl)))
                      .toList(),
                  onChanged: (v) => setState(() => _selectedLevel = v!),
                ),
                const SizedBox(height: 8),
                Text(
                  l.authLevelHint,
                  style: theme.textTheme.bodySmall
                      ?.copyWith(color: colors.onSurfaceVariant),
                ),
                const SizedBox(height: 16),

                // ── Password ──────────────────────────────────────────────
                TextFormField(
                  controller:      _passwordCtrl,
                  obscureText:     _obscurePassword,
                  textInputAction: TextInputAction.next,
                  decoration: InputDecoration(
                    labelText:  l.authPassword,
                    prefixIcon: const Icon(Icons.lock_outline),
                    border:     const OutlineInputBorder(),
                    suffixIcon: IconButton(
                      icon: Icon(_obscurePassword
                          ? Icons.visibility_outlined
                          : Icons.visibility_off_outlined),
                      onPressed: () =>
                          setState(() => _obscurePassword = !_obscurePassword),
                    ),
                  ),
                  validator: (v) =>
                      v == null || v.length < 8 ? l.validPasswordLength : null,
                ),
                const SizedBox(height: 16),

                // ── Confirm password ──────────────────────────────────────
                TextFormField(
                  controller:      _confirmCtrl,
                  obscureText:     true,
                  textInputAction: TextInputAction.done,
                  onFieldSubmitted: (_) => _submit(),
                  decoration: InputDecoration(
                    labelText:  l.authConfirmPassword,
                    prefixIcon: const Icon(Icons.lock_outline),
                    border:     const OutlineInputBorder(),
                  ),
                  validator: (v) =>
                      v != _passwordCtrl.text ? l.validPasswordMismatch : null,
                ),
                const SizedBox(height: 28),

                // ── Submit ────────────────────────────────────────────────
                FilledButton(
                  onPressed: auth.isLoading ? null : _submit,
                  style: FilledButton.styleFrom(
                    minimumSize: const Size.fromHeight(52),
                  ),
                  child: auth.isLoading
                      ? const SizedBox(
                          height: 20,
                          width:  20,
                          child:  CircularProgressIndicator(strokeWidth: 2),
                        )
                      : Text(l.authCreateAccount,
                          style: const TextStyle(fontSize: 16)),
                ),
                const SizedBox(height: 20),

                // ── Login link ────────────────────────────────────────────
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      l.authAlreadyHaveAccount,
                      style: TextStyle(color: colors.onSurfaceVariant),
                    ),
                    GestureDetector(
                      onTap: () => Navigator.of(context).pushReplacement(
                        MaterialPageRoute(builder: (_) => const LoginScreen()),
                      ),
                      child: Text(
                        l.authSignIn,
                        style: TextStyle(
                          color:      colors.primary,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
