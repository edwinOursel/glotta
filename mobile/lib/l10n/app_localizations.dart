import 'package:flutter/material.dart';

// ── Abstract base ─────────────────────────────────────────────────────────────

abstract class AppLocalizations {
  // ── Auth ─────────────────────────────────────────────────────────────────
  String get appTagline;
  String get authEmail;
  String get authPassword;
  String get authConfirmPassword;
  String get authSignIn;
  String get authSignUp;
  String get authCreateAccount;
  String get authNoAccount;
  String get authAlreadyHaveAccount;
  String get authUsernameOptional;
  String get authStartingLevel;
  String get authLevelHint;

  // ── Validation ───────────────────────────────────────────────────────────
  String get validEmail;
  String get validPasswordRequired;
  String get validPasswordLength;
  String get validPasswordMismatch;
  String get validRequired;

  // ── Learn ────────────────────────────────────────────────────────────────
  String get learnTitle;
  String get learnEmptyTitle;
  String get learnEmptyHint;
  String get learnInputHint;
  String get learnGenerate;
  String get learnToggleConstraints;
  String get learnClearHistory;
  String get learnDismiss;
  String get learnJustNow;
  String learnMinutesAgo(int n);
  String learnHoursAgo(int n);
  String get learnPractising;
  String get learnSelectGrammarTheme;
  String learnGrammarThemeTapToChange(String name);
  String get learnPromptLabel;

  // ── Vocabulary ───────────────────────────────────────────────────────────
  String get vocabTitle;
  String get vocabSearch;
  String get vocabImportTooltip;
  String vocabImportLevel(String level);
  String get vocabAddFab;
  String get vocabAddSheetTitle;
  String get vocabWordField;
  String get vocabWordHint;
  String get vocabReadingField;
  String get vocabReadingHint;
  String get vocabMeaningField;
  String get vocabMeaningHint;
  String get vocabLevelField;
  String get vocabLevelOptional;
  String get vocabSaveButton;
  String vocabWordAdded(String word);
  String get vocabPractiseNow;
  String get vocabNoWords;
  String get vocabNoResults;
  String get vocabEmptyHint;
  String get vocabStatsWords;
  String get vocabStatsDue;
  String get vocabBadgeDue;
  String get vocabDeleteTitle;
  String vocabDeleteContent(String word);
  String get vocabDeleteButton;
  String get vocabCancel;
  String vocabImportConfirmTitle(String level);
  String vocabImportConfirmContent(String level);
  String get vocabImportButton;
  String vocabPractiseWord(String word);
  String get vocabLevelAll;

  // ── Grammar ──────────────────────────────────────────────────────────────
  String get grammarTitle;
  String get grammarClear;
  String get grammarCompleteFirst;
  String get grammarOk;
  String get grammarDescription;
  String get grammarExample;
  String get grammarPracticeFocus;
  String get grammarPractiseButton;
  String get grammarSessions;
  String get grammarMastery;

  // ── Progress ─────────────────────────────────────────────────────────────
  String get progressTitle;
  String get progressPlaceholder;

  // ── Settings ─────────────────────────────────────────────────────────────
  String get settingsTitle;
  String get settingsLanguage;
  String get settingsJlptLevel;
  String get settingsConstraintMode;
  String get settingsAbout;
  String get settingsAboutText;
  String get settingsSignOut;

  // ── Navigation ───────────────────────────────────────────────────────────
  // Kept in Japanese — intentional bilingual branding
  String get navLearn;
  String get navVocabulary;
  String get navProgress;
  String get navSettings;

  // ── Lookup ───────────────────────────────────────────────────────────────

  static AppLocalizations of(BuildContext context) =>
      Localizations.of<AppLocalizations>(context, AppLocalizations)!;

  static const LocalizationsDelegate<AppLocalizations> delegate =
      _AppLocalizationsDelegate();

  static const List<Locale> supportedLocales = [
    Locale('en'),
    Locale('fr'),
  ];
}

// ── Delegate ──────────────────────────────────────────────────────────────────

class _AppLocalizationsDelegate
    extends LocalizationsDelegate<AppLocalizations> {
  const _AppLocalizationsDelegate();

  @override
  bool isSupported(Locale locale) =>
      ['en', 'fr'].contains(locale.languageCode);

  @override
  Future<AppLocalizations> load(Locale locale) async =>
      locale.languageCode == 'fr' ? _FrLocalizations() : _EnLocalizations();

  @override
  bool shouldReload(_) => false;
}

// ══════════════════════════════════════════════════════════════════════════════
// English
// ══════════════════════════════════════════════════════════════════════════════

class _EnLocalizations extends AppLocalizations {
  @override String get appTagline           => 'Learn Japanese with constrained AI';
  @override String get authEmail            => 'Email';
  @override String get authPassword         => 'Password';
  @override String get authConfirmPassword  => 'Confirm password';
  @override String get authSignIn           => 'Sign in';
  @override String get authSignUp           => 'Sign up';
  @override String get authCreateAccount    => 'Create account';
  @override String get authNoAccount        => "Don't have an account? ";
  @override String get authAlreadyHaveAccount => 'Already have an account? ';
  @override String get authUsernameOptional => 'Username (optional)';
  @override String get authStartingLevel    => 'Starting JLPT level';
  @override String get authLevelHint        => 'N5 = beginner · N1 = advanced';

  @override String get validEmail           => 'Enter a valid email';
  @override String get validPasswordRequired => 'Enter your password';
  @override String get validPasswordLength  => 'Password must be at least 8 characters';
  @override String get validPasswordMismatch => 'Passwords do not match';
  @override String get validRequired        => 'Required';

  @override String get learnTitle           => '学習 - Learn';
  @override String get learnEmptyTitle      => 'Start learning!';
  @override String get learnEmptyHint       => 'Enter a Japanese prompt below to generate text';
  @override String get learnInputHint       => 'Enter a Japanese prompt… (e.g., 私は)';
  @override String get learnGenerate        => 'Generate';
  @override String get learnToggleConstraints => 'Toggle vocabulary constraints';
  @override String get learnClearHistory    => 'Clear history';
  @override String get learnDismiss         => 'Dismiss';
  @override String get learnJustNow         => 'Just now';
  @override String learnMinutesAgo(int n)   => '${n}m ago';
  @override String learnHoursAgo(int n)     => '${n}h ago';
  @override String get learnPractising      => 'Practising';
  @override String get learnSelectGrammarTheme => 'Select grammar theme';
  @override String learnGrammarThemeTapToChange(String name) => '$name — tap to change';
  @override String get learnPromptLabel     => 'Prompt';

  @override String get vocabTitle           => '単語帳 - Vocabulary';
  @override String get vocabSearch          => 'Search…';
  @override String get vocabImportTooltip   => 'Import JLPT vocabulary';
  @override String vocabImportLevel(String level) => 'Import $level';
  @override String get vocabAddFab          => 'New word';
  @override String get vocabAddSheetTitle   => 'Add a new word';
  @override String get vocabWordField       => 'Word *';
  @override String get vocabWordHint        => 'e.g. 猫、食べる、きれい';
  @override String get vocabReadingField    => 'Reading (furigana)';
  @override String get vocabReadingHint     => 'e.g. ねこ、たべる';
  @override String get vocabMeaningField    => 'Meaning (English)';
  @override String get vocabMeaningHint     => 'e.g. cat, to eat';
  @override String get vocabLevelField      => 'JLPT Level';
  @override String get vocabLevelOptional   => 'Optional';
  @override String get vocabSaveButton      => 'Add word';
  @override String vocabWordAdded(String word) => '「$word」added!';
  @override String get vocabPractiseNow     => 'Practise now';
  @override String get vocabNoWords         => 'No vocabulary yet';
  @override String get vocabNoResults       => 'No words match your search';
  @override String get vocabEmptyHint       => 'Tap + to add a word, or import a JLPT level.';
  @override String get vocabStatsWords      => 'Words';
  @override String get vocabStatsDue        => 'Due today';
  @override String get vocabBadgeDue        => 'Due';
  @override String get vocabDeleteTitle     => 'Delete word?';
  @override String vocabDeleteContent(String word) => 'Remove「$word」from your vocabulary?';
  @override String get vocabDeleteButton    => 'Delete';
  @override String get vocabCancel          => 'Cancel';
  @override String vocabImportConfirmTitle(String level) => 'Import $level vocabulary?';
  @override String vocabImportConfirmContent(String level) =>
      'This will add all $level words from the JLPT dataset to your personal vocabulary.';
  @override String get vocabImportButton    => 'Import';
  @override String vocabPractiseWord(String word) => 'Practise「$word」';
  @override String get vocabLevelAll        => 'All';

  @override String get grammarTitle         => '文法テーマ';
  @override String get grammarClear         => 'Clear';
  @override String get grammarCompleteFirst => 'Complete these themes first:';
  @override String get grammarOk            => 'OK';
  @override String get grammarDescription   => 'Description';
  @override String get grammarExample       => 'Example';
  @override String get grammarPracticeFocus => 'Practice focus';
  @override String get grammarPractiseButton => 'Practise this theme';
  @override String get grammarSessions      => 'Sessions';
  @override String get grammarMastery       => 'Mastery';

  @override String get progressTitle        => '進捗 - Progress';
  @override String get progressPlaceholder  => 'Progress tracking coming soon…';

  @override String get settingsTitle        => '設定 - Settings';
  @override String get settingsLanguage     => 'Language';
  @override String get settingsJlptLevel    => 'JLPT Level';
  @override String get settingsConstraintMode => 'Constraint Mode';
  @override String get settingsAbout        => 'About';
  @override String get settingsAboutText    =>
      'Language learning with constrained LLM generation.\n\n'
      'Learn Japanese by reading AI-generated text built exclusively '
      'from your personal vocabulary.';
  @override String get settingsSignOut      => 'Sign out';

  @override String get navLearn      => '学習';
  @override String get navVocabulary => '単語';
  @override String get navProgress   => '進捗';
  @override String get navSettings   => '設定';
}

// ══════════════════════════════════════════════════════════════════════════════
// French
// ══════════════════════════════════════════════════════════════════════════════

class _FrLocalizations extends AppLocalizations {
  @override String get appTagline           => 'Apprenez le japonais avec l\'IA contrainte';
  @override String get authEmail            => 'E-mail';
  @override String get authPassword         => 'Mot de passe';
  @override String get authConfirmPassword  => 'Confirmer le mot de passe';
  @override String get authSignIn           => 'Se connecter';
  @override String get authSignUp           => 'S\'inscrire';
  @override String get authCreateAccount    => 'Créer un compte';
  @override String get authNoAccount        => 'Pas encore de compte ? ';
  @override String get authAlreadyHaveAccount => 'Déjà un compte ? ';
  @override String get authUsernameOptional => 'Nom d\'utilisateur (facultatif)';
  @override String get authStartingLevel    => 'Niveau JLPT de départ';
  @override String get authLevelHint        => 'N5 = débutant · N1 = avancé';

  @override String get validEmail           => 'Entrez un e-mail valide';
  @override String get validPasswordRequired => 'Entrez votre mot de passe';
  @override String get validPasswordLength  => 'Le mot de passe doit contenir au moins 8 caractères';
  @override String get validPasswordMismatch => 'Les mots de passe ne correspondent pas';
  @override String get validRequired        => 'Requis';

  @override String get learnTitle           => '学習 - Apprendre';
  @override String get learnEmptyTitle      => 'Commencez à apprendre !';
  @override String get learnEmptyHint       => 'Entrez une amorce en japonais ci-dessous pour générer du texte';
  @override String get learnInputHint       => 'Amorce en japonais… (ex. 私は)';
  @override String get learnGenerate        => 'Générer';
  @override String get learnToggleConstraints => 'Activer / désactiver les contraintes';
  @override String get learnClearHistory    => 'Effacer l\'historique';
  @override String get learnDismiss         => 'Fermer';
  @override String get learnJustNow         => 'À l\'instant';
  @override String learnMinutesAgo(int n)   => 'Il y a ${n} min';
  @override String learnHoursAgo(int n)     => 'Il y a ${n} h';
  @override String get learnPractising      => 'Pratique de';
  @override String get learnSelectGrammarTheme => 'Choisir un thème de grammaire';
  @override String learnGrammarThemeTapToChange(String name) => '$name — appuyer pour changer';
  @override String get learnPromptLabel     => 'Amorce';

  @override String get vocabTitle           => '単語帳 - Vocabulaire';
  @override String get vocabSearch          => 'Chercher…';
  @override String get vocabImportTooltip   => 'Importer le vocabulaire JLPT';
  @override String vocabImportLevel(String level) => 'Importer $level';
  @override String get vocabAddFab          => 'Nouveau mot';
  @override String get vocabAddSheetTitle   => 'Ajouter un nouveau mot';
  @override String get vocabWordField       => 'Mot *';
  @override String get vocabWordHint        => 'ex. 猫、食べる、きれい';
  @override String get vocabReadingField    => 'Lecture (furigana)';
  @override String get vocabReadingHint     => 'ex. ねこ、たべる';
  @override String get vocabMeaningField    => 'Sens (anglais)';
  @override String get vocabMeaningHint     => 'ex. chat, manger';
  @override String get vocabLevelField      => 'Niveau JLPT';
  @override String get vocabLevelOptional   => 'Facultatif';
  @override String get vocabSaveButton      => 'Ajouter le mot';
  @override String vocabWordAdded(String word) => '« $word » ajouté !';
  @override String get vocabPractiseNow     => 'Pratiquer maintenant';
  @override String get vocabNoWords         => 'Aucun vocabulaire pour l\'instant';
  @override String get vocabNoResults       => 'Aucun mot ne correspond à la recherche';
  @override String get vocabEmptyHint       => 'Appuyez sur + pour ajouter un mot ou importer un niveau JLPT.';
  @override String get vocabStatsWords      => 'Mots';
  @override String get vocabStatsDue        => 'À revoir';
  @override String get vocabBadgeDue        => 'Dû';
  @override String get vocabDeleteTitle     => 'Supprimer le mot ?';
  @override String vocabDeleteContent(String word) => 'Retirer « $word » de votre vocabulaire ?';
  @override String get vocabDeleteButton    => 'Supprimer';
  @override String get vocabCancel          => 'Annuler';
  @override String vocabImportConfirmTitle(String level) => 'Importer le vocabulaire $level ?';
  @override String vocabImportConfirmContent(String level) =>
      'Cela ajoutera tous les mots $level du jeu de données JLPT à votre vocabulaire personnel.';
  @override String get vocabImportButton    => 'Importer';
  @override String vocabPractiseWord(String word) => 'Pratiquer « $word »';
  @override String get vocabLevelAll        => 'Tous';

  @override String get grammarTitle         => '文法テーマ';
  @override String get grammarClear         => 'Effacer';
  @override String get grammarCompleteFirst => 'Complétez d\'abord ces thèmes :';
  @override String get grammarOk            => 'D\'accord';
  @override String get grammarDescription   => 'Description';
  @override String get grammarExample       => 'Exemple';
  @override String get grammarPracticeFocus => 'Objectif de pratique';
  @override String get grammarPractiseButton => 'Pratiquer ce thème';
  @override String get grammarSessions      => 'Sessions';
  @override String get grammarMastery       => 'Maîtrise';

  @override String get progressTitle        => '進捗 - Progrès';
  @override String get progressPlaceholder  => 'Le suivi des progrès arrive bientôt…';

  @override String get settingsTitle        => '設定 - Paramètres';
  @override String get settingsLanguage     => 'Langue';
  @override String get settingsJlptLevel    => 'Niveau JLPT';
  @override String get settingsConstraintMode => 'Mode de contrainte';
  @override String get settingsAbout        => 'À propos';
  @override String get settingsAboutText    =>
      'Apprentissage du japonais par génération LLM contrainte.\n\n'
      'Lisez des textes générés par l\'IA construits exclusivement '
      'à partir de votre vocabulaire personnel.';
  @override String get settingsSignOut      => 'Se déconnecter';

  @override String get navLearn      => '学習';
  @override String get navVocabulary => '単語';
  @override String get navProgress   => '進捗';
  @override String get navSettings   => '設定';
}
