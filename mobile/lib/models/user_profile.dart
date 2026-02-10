class UserProfile {
  final String id;
  final String email;
  final String? username;
  final String jlptLevel;
  final String constraintMode;
  final DateTime createdAt;
  final DateTime lastActive;

  const UserProfile({
    required this.id,
    required this.email,
    this.username,
    required this.jlptLevel,
    required this.constraintMode,
    required this.createdAt,
    required this.lastActive,
  });

  factory UserProfile.fromJson(Map<String, dynamic> json) => UserProfile(
        id:             json['id'] as String,
        email:          json['email'] as String,
        username:       json['username'] as String?,
        jlptLevel:      json['jlpt_level'] as String,
        constraintMode: json['constraint_mode'] as String,
        createdAt:      DateTime.parse(json['created_at'] as String),
        lastActive:     DateTime.parse(json['last_active'] as String),
      );

  UserProfile copyWith({
    String? username,
    String? jlptLevel,
    String? constraintMode,
  }) =>
      UserProfile(
        id:             id,
        email:          email,
        username:       username ?? this.username,
        jlptLevel:      jlptLevel ?? this.jlptLevel,
        constraintMode: constraintMode ?? this.constraintMode,
        createdAt:      createdAt,
        lastActive:     lastActive,
      );
}
