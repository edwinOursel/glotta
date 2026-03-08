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

  // ── Progress / Stats ─────────────────────────────────────────────────────
  String get progressTitle;
  String get progressStatsWords;
  String get progressStatsDue;
  String get progressStatsSessions;
  String get progressStatsStreak;
  String get progressStreakDays;
  String get progressLongestStreak;
  String get progressJlptProgress;
  String get progressGrammarProgress;
  String get progressTrophies;
  String get progressTrophiesUnlocked;
  String progressTrophyCount(int unlocked, int total);

  // ── Trophies ──────────────────────────────────────────────────────────────
  String get trophyNewUnlocked;
  // First steps
  String get trophyFirstWordName;
  String get trophyFirstWordDesc;
  String get trophyFirstGenerationName;
  String get trophyFirstGenerationDesc;
  String get trophyFirstThemeName;
  String get trophyFirstThemeDesc;
  // Vocabulary count
  String get trophyVocab10Name;
  String get trophyVocab10Desc;
  String get trophyVocab50Name;
  String get trophyVocab50Desc;
  String get trophyVocab100Name;
  String get trophyVocab100Desc;
  String get trophyVocab500Name;
  String get trophyVocab500Desc;
  // Vocabulary mastery
  String get trophyAccuracy80Name;
  String get trophyAccuracy80Desc;
  String get trophyMastered10Name;
  String get trophyMastered10Desc;
  String get trophyMastered50Name;
  String get trophyMastered50Desc;
  // Generation
  String get trophyGenerations10Name;
  String get trophyGenerations10Desc;
  String get trophyGenerations50Name;
  String get trophyGenerations50Desc;
  String get trophyGenerations200Name;
  String get trophyGenerations200Desc;
  // Grammar
  String get trophyThemes5Name;
  String get trophyThemes5Desc;
  String get trophyThemesMastered5Name;
  String get trophyThemesMastered5Desc;
  String get trophyN5CompleteName;
  String get trophyN5CompleteDesc;
  // Streaks
  String get trophyStreak3Name;
  String get trophyStreak3Desc;
  String get trophyStreak7Name;
  String get trophyStreak7Desc;
  String get trophyStreak30Name;
  String get trophyStreak30Desc;
  // Focus practice
  String get trophyFocusPractice10Name;
  String get trophyFocusPractice10Desc;

  // ── Review (SRS flashcard session) ───────────────────────────────────────
  String get reviewTitle;
  String get reviewStart;
  String get reviewNothingDue;
  String get reviewNothingDueHint;
  String reviewProgress(int current, int total);
  String get reviewFlipHint;
  String get reviewHard;
  String get reviewOk;
  String get reviewEasy;
  String get reviewSessionDone;
  String reviewSessionSummary(int reviewed, int correct);
  String get reviewBackToVocab;
  String get reviewAgain;

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

  // ── Nav label style setting ───────────────────────────────────────────────
  String get settingsNavStyle;
  String get settingsNavKanji;
  String get settingsNavHiragana;

  // ── Navigation (5th tab) ──────────────────────────────────────────────────
  String get navFriends;

  // ── Friends & Social ─────────────────────────────────────────────────────
  String get friendsTitle;
  String get friendsFriends;
  String get friendsRequests;
  String get friendsSent;
  String get friendsLeaderboard;
  String get friendsNoFriends;
  String get friendsNoFriendsHint;
  String get friendsNoPending;
  String get friendsAddFriend;
  String get friendsSearchHint;
  String get friendsSearchNoResults;
  String get friendsRequestSent;
  String get friendsRequestPending;
  String get friendsAlreadyFriends;
  String get friendsAccept;
  String get friendsDecline;
  String get friendsRemove;
  String get friendsRemoveTitle;
  String friendsRemoveContent(String name);
  String get friendsVocabSize;
  String get friendsMastered;
  String get friendsStreak;

  // ── Challenges ───────────────────────────────────────────────────────────
  String get challengesTitle;
  String get challengesActive;
  String get challengesHistory;
  String get challengesNone;
  String get challengesNoneHint;
  String get challengesSend;
  String get challengesAccept;
  String get challengesDecline;
  String get challengesTypeVocabSprint;
  String get challengesTypeMasteryRace;
  String get challengesTypeAccuracyDuel;
  String get challengesTypeVocabSprintDesc;
  String get challengesTypeMasteryRaceDesc;
  String get challengesTypeAccuracyDuelDesc;
  String get challengesDuration;
  String challengesDurationDays(int n);
  String get challengesVs;
  String get challengesYou;
  String get challengesWinner;
  String get challengesTie;
  String get challengesStatusPending;
  String get challengesStatusActive;
  String get challengesStatusCompleted;
  String get challengesStatusDeclined;
  String get challengesEndsIn;
  String challengesEndsInDays(int n);
  String challengesScore(double score, String type);

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

  @override String get progressTitle            => '進捗 - Progress';
  @override String get progressStatsWords       => 'Words';
  @override String get progressStatsDue         => 'Due today';
  @override String get progressStatsSessions    => 'Sessions';
  @override String get progressStatsStreak      => 'Streak';
  @override String get progressStreakDays       => 'days';
  @override String get progressLongestStreak    => 'Best streak';
  @override String get progressJlptProgress     => 'Vocabulary by JLPT level';
  @override String get progressGrammarProgress  => 'Grammar mastery';
  @override String get progressTrophies         => 'Trophies';
  @override String get progressTrophiesUnlocked => 'unlocked';
  @override String progressTrophyCount(int unlocked, int total) =>
      '$unlocked / $total trophies unlocked';

  @override String get trophyNewUnlocked          => 'Trophy unlocked!';
  @override String get trophyFirstWordName        => 'First Word';
  @override String get trophyFirstWordDesc        => 'Add your first word to the vocabulary';
  @override String get trophyFirstGenerationName  => 'First Text';
  @override String get trophyFirstGenerationDesc  => 'Generate your first Japanese text';
  @override String get trophyFirstThemeName       => 'Grammar Student';
  @override String get trophyFirstThemeDesc       => 'Select your first grammar theme';
  @override String get trophyVocab10Name          => 'Getting Started';
  @override String get trophyVocab10Desc          => 'Have 10 words in your vocabulary';
  @override String get trophyVocab50Name          => 'Bookworm';
  @override String get trophyVocab50Desc          => 'Have 50 words in your vocabulary';
  @override String get trophyVocab100Name         => 'Lexicon';
  @override String get trophyVocab100Desc         => 'Have 100 words in your vocabulary';
  @override String get trophyVocab500Name         => 'The Library';
  @override String get trophyVocab500Desc         => 'Have 500 words in your vocabulary';
  @override String get trophyAccuracy80Name       => 'Sharp Mind';
  @override String get trophyAccuracy80Desc       => 'Achieve 80%+ accuracy over 20+ reviewed words';
  @override String get trophyMastered10Name       => 'Star Collector';
  @override String get trophyMastered10Desc       => 'Fully master 10 words (level 5)';
  @override String get trophyMastered50Name       => 'Word Master';
  @override String get trophyMastered50Desc       => 'Fully master 50 words (level 5)';
  @override String get trophyGenerations10Name    => 'Scribbler';
  @override String get trophyGenerations10Desc    => 'Generate 10 Japanese texts';
  @override String get trophyGenerations50Name    => 'Author';
  @override String get trophyGenerations50Desc    => 'Generate 50 Japanese texts';
  @override String get trophyGenerations200Name   => 'Storyteller';
  @override String get trophyGenerations200Desc   => 'Generate 200 Japanese texts';
  @override String get trophyThemes5Name          => 'Polyglot';
  @override String get trophyThemes5Desc          => 'Practice 5 different grammar themes';
  @override String get trophyThemesMastered5Name  => 'Grammar Guru';
  @override String get trophyThemesMastered5Desc  => 'Fully master 5 grammar themes';
  @override String get trophyN5CompleteName       => 'N5 Champion';
  @override String get trophyN5CompleteDesc       => 'Master all N5 grammar themes';
  @override String get trophyStreak3Name          => 'On Fire';
  @override String get trophyStreak3Desc          => 'Study 3 days in a row';
  @override String get trophyStreak7Name          => 'Dedicated';
  @override String get trophyStreak7Desc          => 'Study 7 days in a row';
  @override String get trophyStreak30Name         => 'Unstoppable';
  @override String get trophyStreak30Desc         => 'Study 30 days in a row';
  @override String get trophyFocusPractice10Name  => 'Word Whisperer';
  @override String get trophyFocusPractice10Desc  => 'Practise a focus word 10 times';

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

  @override String get settingsNavStyle    => 'Navigation labels';
  @override String get settingsNavKanji    => 'Kanji (漢字)';
  @override String get settingsNavHiragana => 'Hiragana (ひらがな)';

  @override String get navFriends          => '友達';

  @override String get friendsTitle              => '友達 - Friends';
  @override String get friendsFriends            => 'Friends';
  @override String get friendsRequests           => 'Requests';
  @override String get friendsSent               => 'Sent';
  @override String get friendsLeaderboard        => 'Leaderboard';
  @override String get friendsNoFriends          => 'No friends yet';
  @override String get friendsNoFriendsHint      => 'Search for users to add friends and compete together!';
  @override String get friendsNoPending          => 'No pending requests';
  @override String get friendsAddFriend          => 'Add friend';
  @override String get friendsSearchHint         => 'Search by username or email…';
  @override String get friendsSearchNoResults    => 'No users found';
  @override String get friendsRequestSent        => 'Request sent!';
  @override String get friendsRequestPending     => 'Request pending';
  @override String get friendsAlreadyFriends     => 'Already friends';
  @override String get friendsAccept             => 'Accept';
  @override String get friendsDecline            => 'Decline';
  @override String get friendsRemove             => 'Remove friend';
  @override String get friendsRemoveTitle        => 'Remove friend?';
  @override String friendsRemoveContent(String name) => 'Remove $name from your friends list?';
  @override String get friendsVocabSize          => 'Words';
  @override String get friendsMastered           => 'Mastered';
  @override String get friendsStreak             => 'Streak';

  @override String get challengesTitle           => 'Challenges';
  @override String get challengesActive          => 'Active';
  @override String get challengesHistory         => 'History';
  @override String get challengesNone            => 'No active challenges';
  @override String get challengesNoneHint        => 'Challenge a friend to stay motivated!';
  @override String get challengesSend            => 'Send challenge';
  @override String get challengesAccept          => 'Accept';
  @override String get challengesDecline         => 'Decline';
  @override String get challengesTypeVocabSprint => 'Vocab Sprint';
  @override String get challengesTypeMasteryRace => 'Mastery Race';
  @override String get challengesTypeAccuracyDuel => 'Accuracy Duel';
  @override String get challengesTypeVocabSprintDesc =>
      'Who adds the most words during the challenge period?';
  @override String get challengesTypeMasteryRaceDesc =>
      'Who reaches the most mastered words (level 4+)?';
  @override String get challengesTypeAccuracyDuelDesc =>
      'Who has the highest review accuracy?';
  @override String get challengesDuration        => 'Duration';
  @override String challengesDurationDays(int n) => '$n days';
  @override String get challengesVs              => 'vs';
  @override String get challengesYou             => 'You';
  @override String get challengesWinner          => 'Winner';
  @override String get challengesTie             => 'Tie!';
  @override String get challengesStatusPending   => 'Waiting…';
  @override String get challengesStatusActive    => 'Active';
  @override String get challengesStatusCompleted => 'Finished';
  @override String get challengesStatusDeclined  => 'Declined';
  @override String get challengesEndsIn          => 'Ends in';
  @override String challengesEndsInDays(int n)   => '$n days left';
  @override String challengesScore(double score, String type) {
    if (type == 'accuracy_duel') return '${score.toStringAsFixed(1)}%';
    return score.toInt().toString();
  }

  @override String get reviewTitle           => 'Review';
  @override String get reviewStart           => 'Start review';
  @override String get reviewNothingDue      => 'All caught up!';
  @override String get reviewNothingDueHint  => 'No words due for review right now. Come back later!';
  @override String reviewProgress(int current, int total) => '$current / $total';
  @override String get reviewFlipHint        => 'Tap to reveal answer';
  @override String get reviewHard            => 'Hard';
  @override String get reviewOk             => 'OK';
  @override String get reviewEasy            => 'Easy';
  @override String get reviewSessionDone     => 'Session complete!';
  @override String reviewSessionSummary(int reviewed, int correct) =>
      '$correct / $reviewed correct';
  @override String get reviewBackToVocab     => 'Back to vocabulary';
  @override String get reviewAgain           => 'Review again';
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

  @override String get progressTitle            => '進捗 - Progrès';
  @override String get progressStatsWords       => 'Mots';
  @override String get progressStatsDue         => 'À revoir';
  @override String get progressStatsSessions    => 'Sessions';
  @override String get progressStatsStreak      => 'Série';
  @override String get progressStreakDays       => 'jours';
  @override String get progressLongestStreak    => 'Meilleure série';
  @override String get progressJlptProgress     => 'Vocabulaire par niveau JLPT';
  @override String get progressGrammarProgress  => 'Maîtrise de la grammaire';
  @override String get progressTrophies         => 'Trophées';
  @override String get progressTrophiesUnlocked => 'débloqués';
  @override String progressTrophyCount(int unlocked, int total) =>
      '$unlocked / $total trophées débloqués';

  @override String get trophyNewUnlocked          => 'Trophée débloqué !';
  @override String get trophyFirstWordName        => 'Premier mot';
  @override String get trophyFirstWordDesc        => 'Ajoutez votre premier mot au vocabulaire';
  @override String get trophyFirstGenerationName  => 'Premier texte';
  @override String get trophyFirstGenerationDesc  => 'Générez votre premier texte en japonais';
  @override String get trophyFirstThemeName       => 'Étudiant en grammaire';
  @override String get trophyFirstThemeDesc       => 'Sélectionnez votre premier thème grammatical';
  @override String get trophyVocab10Name          => 'Les premiers pas';
  @override String get trophyVocab10Desc          => 'Avoir 10 mots dans votre vocabulaire';
  @override String get trophyVocab50Name          => 'Lecteur assidu';
  @override String get trophyVocab50Desc          => 'Avoir 50 mots dans votre vocabulaire';
  @override String get trophyVocab100Name         => 'Lexique';
  @override String get trophyVocab100Desc         => 'Avoir 100 mots dans votre vocabulaire';
  @override String get trophyVocab500Name         => 'La bibliothèque';
  @override String get trophyVocab500Desc         => 'Avoir 500 mots dans votre vocabulaire';
  @override String get trophyAccuracy80Name       => 'Esprit vif';
  @override String get trophyAccuracy80Desc       => 'Atteindre 80 %+ de précision sur 20+ mots révisés';
  @override String get trophyMastered10Name       => 'Collectionneur d\'étoiles';
  @override String get trophyMastered10Desc       => 'Maîtriser parfaitement 10 mots (niveau 5)';
  @override String get trophyMastered50Name       => 'Maître des mots';
  @override String get trophyMastered50Desc       => 'Maîtriser parfaitement 50 mots (niveau 5)';
  @override String get trophyGenerations10Name    => 'Gribouilleur';
  @override String get trophyGenerations10Desc    => 'Générer 10 textes en japonais';
  @override String get trophyGenerations50Name    => 'Auteur';
  @override String get trophyGenerations50Desc    => 'Générer 50 textes en japonais';
  @override String get trophyGenerations200Name   => 'Conteur';
  @override String get trophyGenerations200Desc   => 'Générer 200 textes en japonais';
  @override String get trophyThemes5Name          => 'Polyglotte';
  @override String get trophyThemes5Desc          => 'Pratiquer 5 thèmes grammaticaux différents';
  @override String get trophyThemesMastered5Name  => 'Gourou de la grammaire';
  @override String get trophyThemesMastered5Desc  => 'Maîtriser parfaitement 5 thèmes grammaticaux';
  @override String get trophyN5CompleteName       => 'Champion N5';
  @override String get trophyN5CompleteDesc       => 'Maîtriser tous les thèmes grammaticaux N5';
  @override String get trophyStreak3Name          => 'En feu !';
  @override String get trophyStreak3Desc          => 'Étudier 3 jours consécutifs';
  @override String get trophyStreak7Name          => 'Persévérant';
  @override String get trophyStreak7Desc          => 'Étudier 7 jours consécutifs';
  @override String get trophyStreak30Name         => 'Inarrêtable';
  @override String get trophyStreak30Desc         => 'Étudier 30 jours consécutifs';
  @override String get trophyFocusPractice10Name  => 'Chuchoteur de mots';
  @override String get trophyFocusPractice10Desc  => 'Pratiquer un mot cible 10 fois';

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

  @override String get settingsNavStyle    => 'Labels de navigation';
  @override String get settingsNavKanji    => 'Kanji (漢字)';
  @override String get settingsNavHiragana => 'Hiragana (ひらがな)';

  @override String get navFriends          => '友達';

  @override String get friendsTitle              => '友達 - Amis';
  @override String get friendsFriends            => 'Amis';
  @override String get friendsRequests           => 'Demandes';
  @override String get friendsSent               => 'Envoyées';
  @override String get friendsLeaderboard        => 'Classement';
  @override String get friendsNoFriends          => 'Pas encore d\'amis';
  @override String get friendsNoFriendsHint      => 'Cherchez des utilisateurs pour ajouter des amis et vous mesurer à eux !';
  @override String get friendsNoPending          => 'Aucune demande en attente';
  @override String get friendsAddFriend          => 'Ajouter un ami';
  @override String get friendsSearchHint         => 'Chercher par nom d\'utilisateur ou e-mail…';
  @override String get friendsSearchNoResults    => 'Aucun utilisateur trouvé';
  @override String get friendsRequestSent        => 'Demande envoyée !';
  @override String get friendsRequestPending     => 'Demande en attente';
  @override String get friendsAlreadyFriends     => 'Déjà amis';
  @override String get friendsAccept             => 'Accepter';
  @override String get friendsDecline            => 'Refuser';
  @override String get friendsRemove             => 'Supprimer l\'ami';
  @override String get friendsRemoveTitle        => 'Supprimer l\'ami ?';
  @override String friendsRemoveContent(String name) => 'Retirer $name de votre liste d\'amis ?';
  @override String get friendsVocabSize          => 'Mots';
  @override String get friendsMastered           => 'Maîtrisés';
  @override String get friendsStreak             => 'Série';

  @override String get challengesTitle           => 'Défis';
  @override String get challengesActive          => 'Actifs';
  @override String get challengesHistory         => 'Historique';
  @override String get challengesNone            => 'Aucun défi actif';
  @override String get challengesNoneHint        => 'Défiez un ami pour rester motivé !';
  @override String get challengesSend            => 'Envoyer un défi';
  @override String get challengesAccept          => 'Accepter';
  @override String get challengesDecline         => 'Refuser';
  @override String get challengesTypeVocabSprint => 'Sprint de vocab';
  @override String get challengesTypeMasteryRace => 'Course à la maîtrise';
  @override String get challengesTypeAccuracyDuel => 'Duel de précision';
  @override String get challengesTypeVocabSprintDesc =>
      'Qui ajoute le plus de mots pendant la période du défi ?';
  @override String get challengesTypeMasteryRaceDesc =>
      'Qui atteint le plus de mots maîtrisés (niveau 4+) ?';
  @override String get challengesTypeAccuracyDuelDesc =>
      'Qui a la meilleure précision de révision ?';
  @override String get challengesDuration        => 'Durée';
  @override String challengesDurationDays(int n) => '$n jours';
  @override String get challengesVs              => 'contre';
  @override String get challengesYou             => 'Vous';
  @override String get challengesWinner          => 'Gagnant';
  @override String get challengesTie             => 'Égalité !';
  @override String get challengesStatusPending   => 'En attente…';
  @override String get challengesStatusActive    => 'Actif';
  @override String get challengesStatusCompleted => 'Terminé';
  @override String get challengesStatusDeclined  => 'Refusé';
  @override String get challengesEndsIn          => 'Se termine dans';
  @override String challengesEndsInDays(int n)   => '$n jours restants';
  @override String challengesScore(double score, String type) {
    if (type == 'accuracy_duel') return '${score.toStringAsFixed(1)} %';
    return score.toInt().toString();
  }

  @override String get reviewTitle           => 'Révision';
  @override String get reviewStart           => 'Commencer la révision';
  @override String get reviewNothingDue      => 'Tout est à jour !';
  @override String get reviewNothingDueHint  => 'Aucun mot à réviser pour l\'instant. Revenez plus tard !';
  @override String reviewProgress(int current, int total) => '$current / $total';
  @override String get reviewFlipHint        => 'Appuyez pour révéler la réponse';
  @override String get reviewHard            => 'Difficile';
  @override String get reviewOk             => 'OK';
  @override String get reviewEasy            => 'Facile';
  @override String get reviewSessionDone     => 'Session terminée !';
  @override String reviewSessionSummary(int reviewed, int correct) =>
      '$correct / $reviewed corrects';
  @override String get reviewBackToVocab     => 'Retour au vocabulaire';
  @override String get reviewAgain           => 'Réviser encore';
}
