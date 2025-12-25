#!/usr/bin/env python3
"""
Script de démonstration pour Glotta - Apprentissage du japonais avec LLM contraint.

Ce script montre comment :
1. Créer un vocabulaire utilisateur
2. Charger un modèle japonais
3. Générer du texte avec contraintes de vocabulaire
4. Comparer génération contrainte vs non-contrainte
"""

from japanese_generator import JapaneseGenerator
from user_vocabulary import UserVocabulary


def demo_basic():
    """Démonstration basique : vocabulaire minimal."""
    print("\n" + "=" * 70)
    print("DEMO 1: Vocabulaire minimal (débutant absolu)")
    print("=" * 70)

    # Créer un vocabulaire très basique (niveau N5)
    generator = JapaneseGenerator(model_name="gpt2-small")
    vocab = generator.user_vocabulary

    # Ajouter quelques mots de base
    beginner_words = [
        # Pronoms
        "私", "あなた", "彼", "彼女",
        # Verbes de base
        "食べる", "飲む", "行く", "来る", "見る", "する", "いる", "ある",
        # Noms communs
        "猫", "犬", "本", "水", "お茶", "ご飯", "家", "学校",
        # Adjectifs
        "好き", "嫌い", "大きい", "小さい", "新しい", "古い",
        # Salutations
        "こんにちは", "さようなら", "ありがとう", "すみません",
    ]
    vocab.add_words(beginner_words)

    print(f"✓ Vocabulaire: {len(vocab)} éléments")
    print(f"✓ Token IDs autorisés: {len(vocab.get_allowed_token_ids())}")

    # Test de génération
    prompts = [
        "私は",
        "猫が",
        "こんにちは",
    ]

    for prompt in prompts:
        print(f"\n📝 Prompt: '{prompt}'")

        # Sans contraintes
        print("   🔓 Sans contraintes:")
        unconstrained = generator.generate(
            prompt,
            max_length=30,
            use_constraints=False,
            num_return_sequences=1
        )[0]
        print(f"      {unconstrained}")

        # Avec contraintes (mode hard)
        print("   🔒 Avec contraintes (hard):")
        constrained = generator.generate(
            prompt,
            max_length=30,
            use_constraints=True,
            num_return_sequences=1
        )[0]
        print(f"      {constrained}")


def demo_intermediate():
    """Démonstration intermédiaire : vocabulaire plus large."""
    print("\n" + "=" * 70)
    print("DEMO 2: Vocabulaire intermédiaire")
    print("=" * 70)

    generator = JapaneseGenerator(model_name="gpt2-small")
    vocab = generator.user_vocabulary

    # Charger depuis un fichier (ou créer)
    intermediate_words = [
        # Verbes
        "勉強する", "働く", "遊ぶ", "話す", "書く", "読む", "聞く", "買う",
        "作る", "会う", "待つ", "思う", "知る", "分かる",
        # Noms
        "友達", "先生", "学生", "会社", "仕事", "時間", "今日", "明日",
        "昨日", "朝", "昼", "夜", "週末", "月曜日", "日本", "東京",
        # Adjectifs/Adverbes
        "楽しい", "難しい", "簡単", "忙しい", "暇", "多い", "少ない",
        "とても", "少し", "全然", "たくさん", "もっと",
        # Conjonctions et expressions
        "でも", "そして", "だから", "それで", "もし", "たぶん",
    ]
    vocab.add_words(intermediate_words)

    print(f"✓ Vocabulaire: {len(vocab)} éléments")

    # Sauvegarder le vocabulaire
    vocab_file = "vocabulary_intermediate.json"
    vocab.save_to_file(vocab_file)
    print(f"✓ Vocabulaire sauvegardé dans {vocab_file}")

    # Test de génération
    prompt = "今日は"
    print(f"\n📝 Prompt: '{prompt}'")

    print("   🔒 Génération avec contraintes:")
    for i in range(3):
        result = generator.generate(
            prompt,
            max_length=40,
            use_constraints=True,
            temperature=0.8,
            num_return_sequences=1
        )[0]
        print(f"      {i+1}. {result}")


def demo_modes():
    """Démonstration des différents modes de contrainte."""
    print("\n" + "=" * 70)
    print("DEMO 3: Comparaison des modes de contrainte")
    print("=" * 70)

    generator = JapaneseGenerator(model_name="gpt2-small")
    vocab = generator.user_vocabulary

    # Vocabulaire très limité pour mieux voir la différence
    limited_words = ["私", "猫", "好き", "です", "とても", "かわいい"]
    vocab.add_words(limited_words)

    print(f"✓ Vocabulaire très limité: {limited_words}")

    prompt = "私は猫が"

    modes = ["hard", "soft", "adaptive"]

    for mode in modes:
        generator.set_constraint_mode(mode)
        print(f"\n🔧 Mode: {mode}")

        result = generator.generate(
            prompt,
            max_length=30,
            use_constraints=True,
            num_return_sequences=1
        )[0]
        print(f"   {result}")


def demo_interactive():
    """Lance le mode interactif."""
    print("\n" + "=" * 70)
    print("DEMO 4: Mode interactif")
    print("=" * 70)

    generator = JapaneseGenerator(model_name="gpt2-small")
    vocab = generator.user_vocabulary

    # Charger un vocabulaire si disponible
    vocab_file = "vocabulary_intermediate.json"
    try:
        vocab.load_from_file(vocab_file)
    except FileNotFoundError:
        # Utiliser un vocabulaire par défaut
        default_words = [
            "私", "あなた", "猫", "犬", "食べる", "飲む", "好き",
            "こんにちは", "ありがとう", "です", "ます", "する", "いる"
        ]
        vocab.add_words(default_words)
        print(f"⚠️  Fichier {vocab_file} non trouvé, utilisation vocabulaire minimal")

    # Lancer le mode interactif
    generator.interactive_mode()


def create_sample_vocabulary():
    """Crée un fichier de vocabulaire d'exemple."""
    print("\n" + "=" * 70)
    print("Création d'un vocabulaire d'exemple")
    print("=" * 70)

    from transformers import AutoTokenizer

    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small")
    vocab = UserVocabulary(tokenizer)

    # Ajouter du vocabulaire N5 basique
    n5_words = [
        # Nombres
        "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
        # Pronoms
        "私", "あなた", "彼", "彼女", "これ", "それ", "あれ", "ここ", "そこ", "あそこ",
        # Verbes fréquents
        "する", "いる", "ある", "行く", "来る", "見る", "食べる", "飲む", "話す",
        "聞く", "書く", "読む", "買う", "売る", "会う", "待つ", "座る", "立つ",
        # Noms communs
        "人", "男", "女", "子供", "犬", "猫", "本", "車", "家", "学校",
        "会社", "駅", "店", "水", "お茶", "ご飯", "パン", "肉", "魚",
        # Temps
        "今", "今日", "明日", "昨日", "朝", "昼", "夜", "時間", "分",
        # Adjectifs
        "良い", "悪い", "大きい", "小さい", "新しい", "古い", "高い", "安い",
        "長い", "短い", "多い", "少ない", "暑い", "寒い", "難しい", "簡単",
        # Expressions
        "こんにちは", "おはよう", "こんばんは", "さようなら", "ありがとう",
        "すみません", "ごめんなさい", "はい", "いいえ",
    ]

    vocab.add_words(n5_words)

    # Sauvegarder
    vocab.save_to_file("vocabulary_n5.json")
    print(f"\n✓ Vocabulaire N5 créé: {len(vocab)} éléments")
    print(f"✓ Fichier: vocabulary_n5.json")


if __name__ == "__main__":
    import sys

    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                         GLOTTA DEMO                            ║
    ║        Apprentissage du japonais avec LLM contraint            ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    if len(sys.argv) > 1:
        demo_choice = sys.argv[1]
    else:
        print("Demos disponibles:")
        print("  1. basic       - Vocabulaire minimal")
        print("  2. intermediate - Vocabulaire intermédiaire")
        print("  3. modes       - Comparaison des modes")
        print("  4. interactive - Mode interactif")
        print("  5. vocab       - Créer vocabulaire d'exemple")
        print()
        demo_choice = input("Choisir une démo (1-5): ").strip()

    try:
        if demo_choice in ["1", "basic"]:
            demo_basic()
        elif demo_choice in ["2", "intermediate"]:
            demo_intermediate()
        elif demo_choice in ["3", "modes"]:
            demo_modes()
        elif demo_choice in ["4", "interactive"]:
            demo_interactive()
        elif demo_choice in ["5", "vocab"]:
            create_sample_vocabulary()
        else:
            print("❌ Choix invalide")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Sayonara!")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
