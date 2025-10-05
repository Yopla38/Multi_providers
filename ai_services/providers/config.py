"""
Configuration centralisée pour le pipeline vidéo.
Modifiez les chemins selon votre installation.
"""
import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════
# CHEMINS DES MODÈLES
# ═══════════════════════════════════════════════════════════

# LLaVA (Vision)
LLAVA_MODEL_PATH = "/home/yopla/montage_models/llm_models/python/models/multimodal/llava-v1.5-13b-Q6_K.gguf"
CLIP_LLAVA_MODEL_PATH = "/home/yopla/montage_models/llm_models/python/models/multimodal/mmproj-model-f16.gguf"

# DeepSeek R1 (LLM)
DEEPSEEK_MODEL_PATH = "/home/yopla/montage_models/llm_models/python/models/txt2txt/DeepSeek-R1-Distill-Qwen-14B-Q5_K_S/DeepSeek-R1-Distill-Qwen-14B-Q5_K_S.gguf"

# ═══════════════════════════════════════════════════════════
# COMFYUI
# ═══════════════════════════════════════════════════════════

COMFYUI_BASE_PATH = "/home/yopla/Applications/SD/comfyui/ComfyUI"
COMFYUI_URL = "http://127.0.0.1:8188"

# Chemins dérivés
COMFYUI_TEMP_DIR = os.path.join(COMFYUI_BASE_PATH, "temp")
COMFYUI_OUTPUT_DIR = os.path.join(COMFYUI_BASE_PATH, "output")

# Workflow vidéo
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
WORKFLOW_PATH = os.path.join(_PROJECT_ROOT, "WAN 2.2_loop_complete_api.json")

# ═══════════════════════════════════════════════════════════
# PARAMÈTRES VIDÉO
# ═══════════════════════════════════════════════════════════

# Fichier temporaire utilisé par le workflow pour la vidéo en cours
TEMP_VIDEO_FILENAME = "TempLoop.mp4"
TEMP_VIDEO_PATH = os.path.join(COMFYUI_TEMP_DIR, TEMP_VIDEO_FILENAME)

# Dossiers de sortie dans la racine du projet Cinemat
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUT_VIDEO_DIR = os.path.join(_PROJECT_ROOT, "output_videos")
TEMP_FRAMES_DIR = os.path.join(_PROJECT_ROOT, "temp_frames")

# ═══════════════════════════════════════════════════════════
# PARAMÈTRES DE GÉNÉRATION
# ═══════════════════════════════════════════════════════════

# Durée d'un segment vidéo (secondes)
SEGMENT_DURATION = 4

# Timeout pour la génération d'un segment (secondes)
VIDEO_GENERATION_TIMEOUT = 600  # 10 minutes

# Durée de stabilité du fichier pour considérer la génération terminée (secondes)
FILE_STABLE_DURATION = 10

# Nombre maximum de segments par séquence (limité pour éviter la dégradation)
MAX_SEGMENTS = 4

# Délai après déchargement d'un modèle (pour libération VRAM)
VRAM_RELEASE_DELAY = 3

# ═══════════════════════════════════════════════════════════
# VÉRIFICATION DE COHÉRENCE
# ═══════════════════════════════════════════════════════════

# Activer la vérification de cohérence avec LLaVA après chaque segment
# Si True : Analyse la dernière frame, compare avec l'attendu, ajuste le prompt suivant
# Si False : Génération plus rapide mais sans vérification
ENABLE_COHERENCE_VERIFICATION = True

# Seuil de tolérance pour considérer qu'un ajustement est nécessaire
COHERENCE_ADJUSTMENT_THRESHOLD = "acceptable"  # "good" | "acceptable" | "problematic"

# Mode de gestion des contradictions
# "auto" : Décision automatique basée sur l'analyse du LLM
# "interactive" : Demande à l'utilisateur en cas de contradiction
CONTRADICTION_HANDLING_MODE = "auto"  # "auto" | "interactive"

# Activer la détection de contradictions sémantiques
# Si True : Analyse si l'ajustement proposé contredit l'intention originale
# Si False : Applique toujours l'ajustement (comportement v1)
ENABLE_CONTRADICTION_DETECTION = True


# ═══════════════════════════════════════════════════════════
# MAPPING DES NŒUDS DU WORKFLOW
# ═══════════════════════════════════════════════════════════

NODE_MAPPING = {
    "input_image_node_id": 433,  # LoadImage
    "prompt_node_id": 59,  # Textbox (Prompt)
    "continue_video_node_id": 610,  # SimpleMathBoolean+ (Continue)
    "num_frames_node_id": 439,  # mxSlider (Frame Count)
    "steps_node_id": 440,  # mxSlider (Steps)
    "resolution_node_id": 601,  # mxSlider (Resolution)
}


# ═══════════════════════════════════════════════════════════
# CRÉATION DES DOSSIERS
# ═══════════════════════════════════════════════════════════

def ensure_directories():
    """Crée les dossiers nécessaires s'ils n'existent pas."""
    Path(OUTPUT_VIDEO_DIR).mkdir(parents=True, exist_ok=True)
    Path(TEMP_FRAMES_DIR).mkdir(parents=True, exist_ok=True)
    Path(COMFYUI_TEMP_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✅ Dossiers créés/vérifiés :")
    print(f"   - {OUTPUT_VIDEO_DIR}")
    print(f"   - {TEMP_FRAMES_DIR}")
    print(f"   - {COMFYUI_TEMP_DIR}")


# ═══════════════════════════════════════════════════════════
# VALIDATION DE LA CONFIGURATION
# ═══════════════════════════════════════════════════════════

def validate_config():
    """Vérifie que tous les fichiers critiques existent."""
    errors = []

    # Vérifier les modèles
    if not os.path.exists(LLAVA_MODEL_PATH):
        errors.append(f"❌ Modèle LLaVA introuvable : {LLAVA_MODEL_PATH}")

    if not os.path.exists(CLIP_LLAVA_MODEL_PATH):
        errors.append(f"❌ Modèle CLIP LLaVA introuvable : {CLIP_LLAVA_MODEL_PATH}")

    if not os.path.exists(DEEPSEEK_MODEL_PATH):
        errors.append(f"❌ Modèle DeepSeek introuvable : {DEEPSEEK_MODEL_PATH}")

    # Vérifier ComfyUI
    if not os.path.exists(COMFYUI_BASE_PATH):
        errors.append(f"❌ ComfyUI introuvable : {COMFYUI_BASE_PATH}")

    if not os.path.exists(WORKFLOW_PATH):
        errors.append(f"❌ Workflow introuvable : {WORKFLOW_PATH}")

    # Vérifier ffmpeg
    import shutil
    if not shutil.which("ffmpeg"):
        errors.append("❌ ffmpeg n'est pas installé ou pas dans le PATH")

    if errors:
        print("\n".join(errors))
        print("\n⚠️  Corrigez les erreurs ci-dessus dans src/video_pipeline/config.py")
        return False

    print("✅ Configuration validée")
    return True


if __name__ == "__main__":
    ensure_directories()
    validate_config()
