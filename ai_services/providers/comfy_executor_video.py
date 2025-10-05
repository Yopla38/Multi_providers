"""
Exécuteur spécialisé pour le workflow de génération vidéo.
Ce module est conçu pour interagir avec des workflows vidéo complexes sur ComfyUI.
"""
import json
import sys
import requests
import uuid
import os
import shutil
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from ai_services.config import settings

# --- Configuration Access ---
COMFYUI_CONFIG = settings.providers.comfyui
VIDEO_PIPELINE_CONFIG = settings.video_pipeline
SERVICE_CONFIG = settings.services.video_generation
NODE_MAPPING = SERVICE_CONFIG.node_mapping

COMFYUI_URL = str(COMFYUI_CONFIG.url)
COMFYUI_BASE_PATH = COMFYUI_CONFIG.base_path
TEMP_VIDEO_PATH = COMFYUI_BASE_PATH / "temp" / VIDEO_PIPELINE_CONFIG.temp_video_filename
VIDEO_GENERATION_TIMEOUT = VIDEO_PIPELINE_CONFIG.generation.timeout
FILE_STABLE_DURATION = VIDEO_PIPELINE_CONFIG.generation.file_stable_duration

# --- Helper Functions ---

def load_workflow(workflow_path: str) -> dict:
    """Charge un workflow JSON depuis un fichier."""
    with open(workflow_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def fix_anything_everywhere(workflow_api: dict) -> dict:
    """
    Injecte manuellement les connexions manquantes pour contourner les limitations
    des nœuds "Anything Everywhere" dans le format API.
    """
    print("   🔧 Résolution des broadcasts...")
    # Ces mappings sont spécifiques à "WAN 2.2_loop_complete_api.json"
    connections = {
        "16": {"t5": ["11", 0]},
        "28": {"vae": ["38", 0]},
        "538": {"vae": ["38", 0]},
        "579": {"text_embeds": ["16", 0], "feta_args": ["55", 0]},
        "581": {"text_embeds": ["16", 0], "feta_args": ["55", 0]},
    }
    for node_id, inputs in connections.items():
        if node_id in workflow_api:
            for input_name, connection in inputs.items():
                if input_name not in workflow_api[node_id]["inputs"]:
                    workflow_api[node_id]["inputs"][input_name] = connection
                    print(f"      ✓ Node {node_id}: ajout {input_name}")
    return workflow_api

def modify_workflow_api(
        workflow_api: dict,
        prompt: str,
        continue_video: bool,
        input_image_path: Optional[str],
        **kwargs
) -> dict:
    """Modifie le workflow avec les paramètres fournis."""
    import random
    workflow_copy = json.loads(json.dumps(workflow_api))
    print("📝 Configuration du workflow API :")

    def set_input(node_id_key, input_name, value):
        node_id = str(NODE_MAPPING.get(node_id_key))
        if node_id and node_id in workflow_copy:
            workflow_copy[node_id]["inputs"][input_name] = value
            print(f"   ✓ {node_id_key} ({input_name}) = {value}")

    set_input("prompt_node_id", "text", prompt)
    set_input("continue_video_node_id", "value", continue_video)
    set_input("num_frames_node_id", "Xi", kwargs.get("num_frames", 81))
    set_input("steps_node_id", "Xi", kwargs.get("steps", 8))
    set_input("resolution_node_id", "Xi", kwargs.get("resolution", 576))

    if input_image_path:
        set_input("input_image_node_id", "image", Path(input_image_path).name)

    # Gérer les seeds (peuvent rester hardcodés si non configurables)
    for seed_node in ["579", "581"]:
        if seed_node in workflow_copy:
            seed = random.randint(0, 2**32 - 1)
            workflow_copy[seed_node]["inputs"]["seed"] = seed
            print(f"   ✓ Seed ({seed_node}) = {seed}")

    return workflow_copy

def copy_image_to_comfyui(image_path: str):
    """Copie l'image dans le dossier input de ComfyUI."""
    comfyui_input = COMFYUI_BASE_PATH / "input"
    comfyui_input.mkdir(exist_ok=True)
    dest_path = comfyui_input / Path(image_path).name
    if not dest_path.exists():
        shutil.copy(image_path, dest_path)
        print(f"   ✓ Image copiée: {dest_path.name}")
    else:
        print(f"   ✓ Image déjà présente: {dest_path.name}")

def queue_prompt(workflow_api: dict, client_id: str) -> Optional[str]:
    """Envoie le workflow à l'API ComfyUI."""
    try:
        response = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow_api, "client_id": client_id}, timeout=30)
        response.raise_for_status()
        result = response.json()
        prompt_id = result.get('prompt_id')
        if prompt_id:
            print(f"   ✓ Workflow envoyé (prompt_id: {prompt_id})")
            return prompt_id
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Erreur d'API: {e}", file=sys.stderr)
    return None

def wait_for_completion() -> bool:
    """Attend que la vidéo soit générée en surveillant la stabilité du fichier."""
    print("   ⏳ Attente de la génération...")
    start_time = time.time()
    last_size = -1
    last_change_time = time.time()

    while time.time() - start_time < VIDEO_GENERATION_TIMEOUT:
        try:
            current_size = TEMP_VIDEO_PATH.stat().st_size if TEMP_VIDEO_PATH.exists() else -1
            if current_size != last_size:
                last_size = current_size
                last_change_time = time.time()
                if current_size > 0:
                    print(f"   📊 {int(time.time() - start_time)}s - {current_size / 1024 / 1024:.2f} MB")

            if current_size > 0 and (time.time() - last_change_time) >= FILE_STABLE_DURATION:
                print(f"   ✅ Vidéo stable ({current_size / 1024 / 1024:.2f} MB)")
                return True
        except OSError:
            pass
        time.sleep(2)

    print(f"   ❌ Timeout après {VIDEO_GENERATION_TIMEOUT}s", file=sys.stderr)
    return False

def extract_last_frame(video_path: str, output_path: str) -> bool:
    """Extrait la dernière frame d'une vidéo avec ffmpeg."""
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-sseof", "-0.5", "-i", video_path,
            "-update", "1", "-q:v", "1", "-y", output_path
        ], capture_output=True, check=True, timeout=30)
        if Path(output_path).exists():
            print("   ✓ Frame extraite")
            return True
    except Exception as e:
        print(f"   ❌ Erreur d'extraction de frame: {e}", file=sys.stderr)
    return False

# --- Main Execution Function ---

def execute_video_workflow(
        prompt: str,
        continue_video: bool = False,
        input_image_path: Optional[str] = None,
        **kwargs
) -> bool:
    """
    Orchestre l'exécution du workflow de génération vidéo.
    """
    print(f"\n{'─'*60}\n🎬 Exécution du workflow vidéo\n{'─'*60}")

    workflow_path = SERVICE_CONFIG.workflow
    if not workflow_path or not Path(workflow_path).exists():
        print(f"  [Erreur] Workflow vidéo introuvable: {workflow_path}", file=sys.stderr)
        return False

    if not continue_video:
        if TEMP_VIDEO_PATH.exists():
            TEMP_VIDEO_PATH.unlink()
            print("🗑️  Ancien fichier vidéo temporaire supprimé.")
        if not input_image_path:
            print("  [Erreur] Une image d'entrée est requise pour démarrer une nouvelle vidéo.", file=sys.stderr)
            return False
        try:
            copy_image_to_comfyui(input_image_path)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"   ❌ ERREUR lors de la préparation de l'image : {e}", file=sys.stderr)
            return False
    elif not TEMP_VIDEO_PATH.exists():
        print("⚠️  Continue=True mais aucun fichier vidéo précédent trouvé. La génération va échouer.", file=sys.stderr)
        return False

    workflow_api = load_workflow(workflow_path)
    workflow_api = fix_anything_everywhere(workflow_api)
    modified_workflow = modify_workflow_api(
        workflow_api, prompt, continue_video, input_image_path, **kwargs
    )

    print("🚀 Envoi à ComfyUI...")
    client_id = str(uuid.uuid4())
    if not queue_prompt(modified_workflow, client_id):
        return False

    success = wait_for_completion()
    if success:
        print("✅ Génération terminée avec succès.")
    else:
        print("❌ Échec de la génération.", file=sys.stderr)

    return success