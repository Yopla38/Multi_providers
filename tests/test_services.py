"""
Script de test complet pour la bibliothèque de services AI.
Teste tous les providers et fonctionnalités.
"""
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List

# Créer les dossiers de sortie
Path("test_output/images").mkdir(parents=True, exist_ok=True)
Path("test_output/videos").mkdir(parents=True, exist_ok=True)


def print_section(title: str):
    """Affiche une section formatée."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# ═══════════════════════════════════════════════════════════════════════
# TEST 1 : GÉNÉRATION D'IMAGES
# ═══════════════════════════════════════════════════════════════════════

def test_image_generation():
    """Test de génération d'images."""
    print_section("🎨 TEST 1 : Génération d'Images")

    from ai_services import media

    # Test 1.1: Image simple
    print("📍 Test 1.1: Génération simple")
    success = media.generate_image(
        prompt="A serene Japanese garden with cherry blossoms, koi pond, and stone lanterns, cinematic lighting, highly detailed",
        output_path="test_output/images/garden.png",
        loras=None  # Pas de LoRA pour ce test
    )

    if success:
        print("✅ Image générée : test_output/images/garden.png")
    else:
        print("❌ Échec de la génération")

    # Test 1.2: Image avec LoRAs (si configuré)
    print("\n📍 Test 1.2: Génération avec LoRAs")
    success = media.generate_image(
        prompt="Portrait of a woman, elegant, professional photo",
        output_path="test_output/images/portrait.png",
        loras={"nsfw": 0.0}  # LoRA désactivé
    )

    if success:
        print("✅ Image avec LoRA générée : test_output/images/portrait.png")
    else:
        print("❌ Échec de la génération avec LoRA")

    return success


# ═══════════════════════════════════════════════════════════════════════
# TEST 2 : ÉDITION D'IMAGES
# ═══════════════════════════════════════════════════════════════════════

def test_image_editing():
    """Test d'édition d'images."""
    print_section("✏️ TEST 2 : Édition d'Images")

    from ai_services import media

    # Vérifier que l'image de test existe
    base_image = "test_output/images/garden.png"
    if not Path(base_image).exists():
        print(f"⚠️ Image de base introuvable : {base_image}")
        print("   Générez d'abord une image avec test_image_generation()")
        return False

    print(f"📍 Édition de : {base_image}")
    success = media.generate_image(
        prompt="Add a traditional Japanese temple in the background, golden hour lighting",
        input_image=base_image,  # Active automatiquement l'édition
        output_path="test_output/images/garden_edited.png"
    )

    if success:
        print("✅ Image éditée : test_output/images/garden_edited.png")
    else:
        print("❌ Échec de l'édition")

    return success


# ═══════════════════════════════════════════════════════════════════════
# TEST 3 : COMPOSITION D'IMAGES
# ═══════════════════════════════════════════════════════════════════════

def test_image_composition():
    """Test de composition d'images."""
    print_section("🖼️ TEST 3 : Composition d'Images")

    from ai_services import media

    # Vérifier que les images sources existent
    images = [
        "test_output/images/garden.png",
        "test_output/images/portrait.png"
    ]

    missing = [img for img in images if not Path(img).exists()]
    if missing:
        print(f"⚠️ Images manquantes : {missing}")
        print("   Générez d'abord les images avec test_image_generation()")
        return False

    print(f"📍 Composition de {len(images)} images")
    success = media.compose_images(
        prompt="Combine these images into a harmonious scene",
        input_images=images,
        output_path="test_output/images/composed.png"
    )

    if success:
        print("✅ Image composée : test_output/images/composed.png")
    else:
        print("❌ Échec de la composition")

    return success


# ═══════════════════════════════════════════════════════════════════════
# TEST 4 : GÉNÉRATION DE VIDÉOS
# ═══════════════════════════════════════════════════════════════════════

def test_video_generation():
    """Test de génération de vidéos."""
    print_section("🎬 TEST 4 : Génération de Vidéos")

    from ai_services import media

    # Vérifier que l'image de départ existe
    start_image = "test_output/images/garden.png"
    if not Path(start_image).exists():
        print(f"⚠️ Image de départ introuvable : {start_image}")
        print("   Générez d'abord une image avec test_image_generation()")
        return False

    print(f"📍 Génération vidéo (81 frames)")
    success = media.generate_video(
        prompt="Camera slowly pans across the serene garden, petals gently falling",
        input_image=start_image,
        output_path="test_output/videos/garden_animation.mp4",
        num_frames=81,
        steps=8,
        resolution=576
    )

    if success:
        print("✅ Vidéo générée : test_output/videos/garden_animation.mp4")
    else:
        print("❌ Échec de la génération vidéo")

    return success


# ═══════════════════════════════════════════════════════════════════════
# TEST 5 : LLM TEXTE SIMPLE
# ═══════════════════════════════════════════════════════════════════════

def test_text_generation():
    """Test de génération de texte simple."""
    print_section("🤖 TEST 5 : Génération de Texte Simple")

    from ai_services import llm

    # Test 5.1: Réponse simple
    print("📍 Test 5.1: Réponse simple")
    response = llm.generate_text(
        prompt="Écris un haiku sur un jardin japonais."
    )
    print(f"\n📝 Réponse :\n{response}\n")

    # Test 5.2: Avec system prompt
    print("📍 Test 5.2: Avec system prompt")
    response = llm.generate_text(
        prompt="Explique-moi la photosynthèse en 2 phrases.",
        system_prompt="Tu es un professeur de biologie qui vulgarise pour des enfants de 10 ans."
    )
    print(f"\n📝 Réponse :\n{response}\n")

    # Test 5.3: Conversation avec historique
    print("📍 Test 5.3: Conversation (historique)")
    llm.clear_history()  # Réinitialiser

    response1 = llm.generate_text("Mon nom est Alice.")
    print(f"User: Mon nom est Alice.")
    print(f"Assistant: {response1}")

    response2 = llm.generate_text("Quel est mon nom ?")
    print(f"\nUser: Quel est mon nom ?")
    print(f"Assistant: {response2}\n")

    return True


# ═══════════════════════════════════════════════════════════════════════
# TEST 6 : LLM TEXTE STRUCTURÉ (PYDANTIC)
# ═══════════════════════════════════════════════════════════════════════

def test_structured_text():
    """Test de génération de texte structuré avec Pydantic."""
    print_section("📊 TEST 6 : Génération de Texte Structuré")

    from ai_services import llm

    # Définir les modèles Pydantic
    class MovieReview(BaseModel):
        """Critique de film structurée."""
        title: str = Field(description="Titre du film")
        rating: float = Field(ge=0, le=10, description="Note sur 10")
        summary: str = Field(description="Résumé en une phrase")
        pros: List[str] = Field(description="Points positifs")
        cons: List[str] = Field(description="Points négatifs")
        recommendation: str = Field(description="Recommandation finale")

    print("📍 Extraction structurée avec Pydantic")

    result = llm.generate_text(
        prompt="""
Analyse ce texte et extrait les informations :

"Inception est un chef-d'œuvre de Nolan. Les effets visuels sont époustouflants
et le casting est parfait. Cependant, l'intrigue est parfois confuse et le film
est un peu long (2h30). Je recommande vivement ce film aux amateurs de science-fiction."
        """,
        pydantic_model=MovieReview
    )

    print(f"\n📦 Résultat structuré :")
    print(f"   Titre: {result.title}")
    print(f"   Note: {result.rating}/10")
    print(f"   Résumé: {result.summary}")
    print(f"   Pros: {', '.join(result.pros)}")
    print(f"   Cons: {', '.join(result.cons)}")
    print(f"   Recommandation: {result.recommendation}\n")

    # Test avec un modèle plus complexe
    class TechnicalAnalysis(BaseModel):
        """Analyse technique structurée."""
        topic: str
        complexity_level: str = Field(description="beginner, intermediate, advanced")
        key_concepts: List[str]
        prerequisites: List[str]
        estimated_learning_time: str
        resources: List[str]

    print("📍 Analyse technique structurée")

    analysis = llm.generate_text(
        prompt="Analyse l'apprentissage du framework Django pour le développement web Python",
        pydantic_model=TechnicalAnalysis
    )

    print(f"\n📦 Analyse :")
    print(f"   Sujet: {analysis.topic}")
    print(f"   Niveau: {analysis.complexity_level}")
    print(f"   Concepts clés: {', '.join(analysis.key_concepts[:3])}...")
    print(f"   Temps estimé: {analysis.estimated_learning_time}\n")

    return True


# ═══════════════════════════════════════════════════════════════════════
# TEST 7 : ANALYSE D'IMAGES (MULTIMODAL)
# ═══════════════════════════════════════════════════════════════════════

def test_image_analysis():
    """Test d'analyse d'images avec LLaVA."""
    print_section("👁️ TEST 7 : Analyse d'Images (Multimodal)")

    from ai_services import llm

    # Vérifier que l'image existe
    test_image = "test_output/images/garden.png"
    if not Path(test_image).exists():
        print(f"⚠️ Image de test introuvable : {test_image}")
        print("   Générez d'abord une image avec test_image_generation()")
        return False

    # Test 7.1: Description simple
    print(f"📍 Test 7.1: Description de l'image")
    description = llm.analyze_image(
        prompt="Décris cette image en détail. Que vois-tu ?",
        image_path=test_image
    )
    print(f"\n📝 Description :\n{description}\n")

    # Test 7.2: Questions spécifiques
    print(f"📍 Test 7.2: Questions spécifiques")
    questions = [
        "Quelles couleurs dominent dans cette image ?",
        "Y a-t-il des éléments architecturaux visibles ?",
        "Quelle est l'ambiance générale de cette scène ?"
    ]

    for question in questions:
        answer = llm.analyze_image(
            prompt=question,
            image_path=test_image
        )
        print(f"❓ {question}")
        print(f"💬 {answer}\n")

    # Test 7.3: Analyse avec contexte expert
    print(f"📍 Test 7.3: Analyse expert")
    expert_analysis = llm.analyze_image(
        prompt="Analyse la composition photographique de cette image : règle des tiers, équilibre, point focal, etc.",
        image_path=test_image,
        system_prompt="Tu es un photographe professionnel expert en composition."
    )
    print(f"\n📸 Analyse photographique :\n{expert_analysis}\n")

    return True


# ═══════════════════════════════════════════════════════════════════════
# TEST 8 : WORKFLOW COMPLET
# ═══════════════════════════════════════════════════════════════════════

def test_complete_workflow():
    """Test d'un workflow complet combinant plusieurs services."""
    print_section("🔄 TEST 8 : Workflow Complet")

    from ai_services import media, llm

    # Étape 1: Générer une idée avec le LLM
    print("📍 Étape 1: Génération d'idée créative")

    class ImageConcept(BaseModel):
        """Concept d'image structuré."""
        subject: str = Field(description="Sujet principal")
        setting: str = Field(description="Environnement/décor")
        mood: str = Field(description="Ambiance/atmosphère")
        lighting: str = Field(description="Type d'éclairage")
        style: str = Field(description="Style artistique")
        full_prompt: str = Field(description="Prompt complet pour génération")

    concept = llm.generate_text(
        prompt="Crée un concept d'image artistique sur le thème 'harmonie entre nature et technologie'",
        pydantic_model=ImageConcept
    )

    print(f"\n💡 Concept généré :")
    print(f"   Sujet: {concept.subject}")
    print(f"   Décor: {concept.setting}")
    print(f"   Ambiance: {concept.mood}")
    print(f"   Style: {concept.style}")
    print(f"\n📝 Prompt complet :\n{concept.full_prompt}\n")

    # Étape 2: Générer l'image
    print("📍 Étape 2: Génération de l'image")
    image_path = "test_output/images/workflow_generated.png"

    success = media.generate_image(
        prompt=concept.full_prompt,
        output_path=image_path
    )

    if not success:
        print("❌ Échec de la génération d'image")
        return False

    print(f"✅ Image générée : {image_path}")

    # Étape 3: Analyser l'image générée
    print("\n📍 Étape 3: Analyse de l'image générée")

    analysis = llm.analyze_image(
        prompt="Analyse cette image : est-ce que le concept initial a été bien respecté ? Décris ce que tu vois.",
        image_path=image_path
    )

    print(f"\n🔍 Analyse :\n{analysis}\n")

    # Étape 4: Amélioration basée sur l'analyse
    print("📍 Étape 4: Génération d'une version améliorée")

    improvement_prompt = llm.generate_text(
        prompt=f"""
Voici le concept initial :
{concept.full_prompt}

Voici l'analyse de l'image générée :
{analysis}

Propose un prompt amélioré pour corriger les défauts et améliorer l'image.
Réponds uniquement avec le nouveau prompt, sans explication.
        """
    )

    print(f"\n✨ Prompt amélioré :\n{improvement_prompt}\n")

    success = media.generate_image(
        prompt=improvement_prompt,
        output_path="test_output/images/workflow_improved.png"
    )

    if success:
        print("✅ Image améliorée : test_output/images/workflow_improved.png")

    return success


# ═══════════════════════════════════════════════════════════════════════
# TEST 9 : ASYNC (OPTIONNEL)
# ═══════════════════════════════════════════════════════════════════════

async def test_async_operations():
    """Test des opérations asynchrones."""
    print_section("⚡ TEST 9 : Opérations Asynchrones")

    from ai_services import llm

    print("📍 Génération asynchrone simultanée")

    # Lancer plusieurs générations en parallèle
    tasks = [
        llm.generate_text_async("Écris un haiku sur l'hiver"),
        llm.generate_text_async("Écris un haiku sur l'été"),
        llm.generate_text_async("Écris un haiku sur l'automne")
    ]

    results = await asyncio.gather(*tasks)

    seasons = ["Hiver", "Été", "Automne"]
    for season, haiku in zip(seasons, results):
        print(f"\n🍂 {season} :")
        print(haiku)

    return True


# ═══════════════════════════════════════════════════════════════════════
# MENU PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

def print_menu():
    """Affiche le menu des tests."""
    print("\n" + "=" * 70)
    print("  🧪 MENU DE TEST - Bibliothèque de Services AI")
    print("=" * 70)
    print("\n📦 Tests Services Média (Images/Vidéos)")
    print("  1. Génération d'images")
    print("  2. Édition d'images")
    print("  3. Composition d'images")
    print("  4. Génération de vidéos")
    print("\n🤖 Tests LLM (Texte)")
    print("  5. Génération de texte simple")
    print("  6. Génération de texte structuré (Pydantic)")
    print("  7. Analyse d'images (Multimodal)")
    print("\n🔄 Tests Avancés")
    print("  8. Workflow complet")
    print("  9. Opérations asynchrones")
    print("\n⚡ Actions Rapides")
    print("  A. Tout tester (tests 1-8)")
    print("  Q. Quitter")
    print("\n" + "=" * 70)


def run_all_tests():
    """Exécute tous les tests de base."""
    print_section("🚀 EXÉCUTION DE TOUS LES TESTS")

    results = {}

    try:
        results["Image Generation"] = test_image_generation()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        results["Image Generation"] = False

    try:
        results["Image Editing"] = test_image_editing()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        results["Image Editing"] = False

    try:
        results["Image Composition"] = test_image_composition()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        results["Image Composition"] = False

    try:
        results["Video Generation"] = test_video_generation()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        results["Video Generation"] = False

    try:
        results["Text Generation"] = test_text_generation()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        results["Text Generation"] = False

    try:
        results["Structured Text"] = test_structured_text()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        results["Structured Text"] = False

    try:
        results["Image Analysis"] = test_image_analysis()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        results["Image Analysis"] = False

    try:
        results["Complete Workflow"] = test_complete_workflow()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        results["Complete Workflow"] = False

    # Résumé
    print_section("📊 RÉSUMÉ DES TESTS")

    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")

    total = len(results)
    passed = sum(1 for s in results.values() if s)
    print(f"\n🎯 Score: {passed}/{total} tests réussis")


def main():
    """Point d'entrée principal."""
    while True:
        print_menu()
        choice = input("\n👉 Votre choix : ").strip().upper()

        try:
            if choice == "1":
                test_image_generation()
            elif choice == "2":
                test_image_editing()
            elif choice == "3":
                test_image_composition()
            elif choice == "4":
                test_video_generation()
            elif choice == "5":
                test_text_generation()
            elif choice == "6":
                test_structured_text()
            elif choice == "7":
                test_image_analysis()
            elif choice == "8":
                test_complete_workflow()
            elif choice == "9":
                asyncio.run(test_async_operations())
            elif choice == "A":
                run_all_tests()
            elif choice == "Q":
                print("\n👋 Au revoir !\n")
                break
            else:
                print("\n⚠️ Choix invalide")

            input("\n⏸️  Appuyez sur Entrée pour continuer...")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Au revoir !\n")
            break
        except Exception as e:
            print(f"\n❌ ERREUR : {e}")
            import traceback
            traceback.print_exc()
            input("\n⏸️  Appuyez sur Entrée pour continuer...")


if __name__ == "__main__":
    main()