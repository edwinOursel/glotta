import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../l10n/app_localizations.dart';
import '../../providers/auth_provider.dart';
import 'register_screen.dart';

class LoginScreen extends ConsumerStatefulWidget {
  const LoginScreen({super.key});

  @override
  ConsumerState<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends ConsumerState<LoginScreen> {
  final _formKey       = GlobalKey<FormState>();
  final _emailCtrl     = TextEditingController();
  final _passwordCtrl  = TextEditingController();
  bool _obscurePassword = true;

  @override
  void dispose() {
    _emailCtrl.dispose();
    _passwordCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    await ref.read(authProvider.notifier).login(
      email:    _emailCtrl.text.trim(),
      password: _passwordCtrl.text,
    );
  }

  @override
  Widget build(BuildContext context) {
    final l       = AppLocalizations.of(context);
    final auth    = ref.watch(authProvider);
    final theme   = Theme.of(context);
    final colors  = theme.colorScheme;

    // Show backend error as snackbar
    ref.listen<AuthState>(authProvider, (_, next) {
      if (next.error != null) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(next.error!),
            backgroundColor: colors.error,
          ),
        );
      }
    });

    return Scaffold(
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 40),
            child: Form(
              key: _formKey,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // ── Logo / title ──────────────────────────────────────────
                  Icon(Icons.translate_rounded, size: 64, color: colors.primary),
                  const SizedBox(height: 12),
                  Text(
                    'Glotta',
                    style: theme.textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: colors.primary,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  Text(
                    l.appTagline,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: colors.onSurfaceVariant,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 40),

                  // ── Email ─────────────────────────────────────────────────
                  TextFormField(
                    controller:   _emailCtrl,
                    keyboardType: TextInputType.emailAddress,
                    textInputAction: TextInputAction.next,
                    decoration: InputDecoration(
                      labelText:   l.authEmail,
                      prefixIcon:  const Icon(Icons.email_outlined),
                      border:      const OutlineInputBorder(),
                    ),
                    validator: (v) =>
                        v == null || !v.contains('@') ? l.validEmail : null,
                  ),
                  const SizedBox(height: 16),

                  // ── Password ──────────────────────────────────────────────
                  TextFormField(
                    controller:      _passwordCtrl,
                    obscureText:     _obscurePassword,
                    textInputAction: TextInputAction.done,
                    onFieldSubmitted: (_) => _submit(),
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
                        v == null || v.isEmpty ? l.validPasswordRequired : null,
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
                        : Text(l.authSignIn,
                            style: const TextStyle(fontSize: 16)),
                  ),
                  const SizedBox(height: 20),

                  // ── Register link ─────────────────────────────────────────
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        l.authNoAccount,
                        style: TextStyle(color: colors.onSurfaceVariant),
                      ),
                      GestureDetector(
                        onTap: () => Navigator.of(context).pushReplacement(
                          MaterialPageRoute(
                            builder: (_) => const RegisterScreen(),
                          ),
                        ),
                        child: Text(
                          l.authSignUp,
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
      ),
    );
  }
}
