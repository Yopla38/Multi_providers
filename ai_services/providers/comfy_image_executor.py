"""
Exécuteur ComfyUI pour la génération d'images.
Ce module interagit avec l'API de ComfyUI pour exécuter des workflows.
"""
import json
import requests
import uuid
import os
import copy
import random
import time
import sys
import shutil
from pathlib import Path
from ai_services.config import settings

# --- Configuration ---
COMFYUI_CONFIG = settings.providers.comfyui
COMFYUI_URL = str(COMFYUI_CONFIG.url)
COMFYUI_BASE_PATH = COMFYUI_CONFIG.base_path
TEMP_OUTPUT_DIR_NAME = "_pipeline_temp"

# --- Fonctions Utilitaires ---

def charger_json(chemin_fichier: str):
    """Charge un fichier JSON."""
    with open(chemin_fichier, 'r', encoding='utf-8') as f:
        return json.load(f)

def modifier_valeur_noeud_dans_liste(workflow: dict, id_noeud: int, index_widget: int, nouvelle_valeur):
    """Modifie une valeur dans les 'widgets_values' d'un nœud."""
    for node in workflow.get('nodes', []):
        if node.get('id') == id_noeud and 'widgets_values' in node and len(node['widgets_values']) > index_widget:
            node['widgets_values'][index_widget] = nouvelle_valeur
            return True
    return False

def transformer_workflow_pour_api(workflow: dict) -> dict:
    """Transforme un workflow UI en format de prompt pour l'API ComfyUI."""
    prompt_api = {}
    primitive_values = {str(node['id']): node['widgets_values'][0] for node in workflow['nodes'] if node['type'] == 'PrimitiveNode'}
    ui_only_node_types = {'Note', 'PrimitiveNode'}

    for node in workflow['nodes']:
        if node['type'] in ui_only_node_types:
            continue

        node_id_str = str(node['id'])
        api_node = {"class_type": node['type'], "inputs": {}}

        if 'widgets_values' in node:
            widget_names = [inp['name'] for inp in node.get('inputs', []) if 'widget' in inp]
            control_strings = ["fixed", "randomize", "increment", "decrement"]
            actual_values = [v for v in node['widgets_values'] if v not in control_strings]
            for name, value in zip(widget_names, actual_values):
                api_node['inputs'][name] = value

        for i, inp in enumerate(node.get('inputs', [])):
            input_name = inp['name']
            for link in workflow.get('links', []):
                if link[3] == node['id'] and link[4] == i:
                    source_node_id = str(link[1])
                    if source_node_id in primitive_values:
                        api_node['inputs'][input_name] = primitive_values[source_node_id]
                    else:
                        api_node['inputs'][input_name] = [source_node_id, link[2]]
                    break
        prompt_api[node_id_str] = api_node
    return prompt_api

def mettre_workflow_en_file(workflow_api: dict, client_id: str) -> dict | None:
    """Envoie le workflow à l'API ComfyUI."""
    try:
        reponse = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow_api, "client_id": client_id})
        reponse.raise_for_status()
        return reponse.json()
    except requests.exceptions.RequestException as e:
        print(f"  [Erreur Executor] : {e}", file=sys.stderr)
        return None

def safe_copy_image_to_comfy_input(source_path: str) -> str:
    """Copie une image vers le dossier input de ComfyUI de manière sécurisée."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Image source introuvable : {source_path}")

    comfy_input_dir = COMFYUI_BASE_PATH / "input"
    comfy_input_dir.mkdir(exist_ok=True)

    unique_name = f"pipeline_{uuid.uuid4().hex[:8]}{Path(source_path).suffix}"
    dest_path = comfy_input_dir / unique_name

    print("  [Executor] 📦 Copie de l'image vers ComfyUI/input/...")
    shutil.copy2(source_path, dest_path)

    # Vérification de la copie
    for _ in range(30):
        if dest_path.exists() and dest_path.stat().st_size == Path(source_path).stat().st_size:
            print("  [Executor] ✅ Image copiée et vérifiée")
            return unique_name
        time.sleep(0.1)
    raise RuntimeError("Timeout lors de la vérification de la copie de l'image")

def cleanup_comfy_input_file(filename: str):
    """Nettoie un fichier du dossier input de ComfyUI."""
    try:
        file_path = COMFYUI_BASE_PATH / "input" / filename
        if file_path.exists():
            file_path.unlink()
            print(f"  [Executor] 🧹 Nettoyé : {filename}")
    except Exception as e:
        print(f"  [Warning] Impossible de nettoyer {filename} : {e}")

def attendre_fichier(output_dir: Path, expected_prefix: str, timeout: int = 300) -> Path | None:
    """Attend la création d'un fichier de sortie dans un dossier."""
    print(f"  [Executor] Attente du fichier '{expected_prefix}' dans : {output_dir}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            for f in os.listdir(output_dir):
                if f.startswith(expected_prefix) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    found_path = output_dir / f
                    print(f"  [Executor] ✅ Fichier détecté : {f}")
                    return found_path
        except FileNotFoundError:
            pass
        time.sleep(2.0)
    print(f"  [Executor] ⏱ Timeout après {timeout}s.", file=sys.stderr)
    return None

def execute_image_workflow(
        service_name: str,
        final_output_path: str,
        prompt: str,
        input_image: str = None,
        loras: dict = None
):
    """
    Exécute un workflow ComfyUI pour générer une image en utilisant la configuration centralisée.
    """
    client_id = str(uuid.uuid4())
    copied_input_filename = None

    try:
        # --- Récupérer la configuration du service ---
        service_config = settings.services.get(service_name)
        if not service_config or not service_config.workflow or not service_config.node_mapping:
            print(f"  [Erreur] Service '{service_name}' ou sa configuration (workflow, node_mapping) est introuvable.", file=sys.stderr)
            return False

        workflow_path = Path(service_config.workflow)
        mapping = service_config.node_mapping

        # --- Préparation ---
        temp_dir_abs = COMFYUI_BASE_PATH / "output" / TEMP_OUTPUT_DIR_NAME
        temp_dir_abs.mkdir(exist_ok=True)
        temp_prefix = str(uuid.uuid4())

        workflow = charger_json(str(workflow_path))
        workflow_copy = copy.deepcopy(workflow)

        print(f"\n[Executor] Configuration du workflow pour le service : {service_name}")

        # --- Modification du workflow ---
        modifier_valeur_noeud_dans_liste(workflow_copy, mapping['prompt_node_id'], 0, prompt)
        modifier_valeur_noeud_dans_liste(workflow_copy, mapping['seed_node_id'], 0, random.randint(0, 2**32 - 1))
        modifier_valeur_noeud_dans_liste(workflow_copy, mapping['save_node_id'], 0, str(temp_dir_abs / temp_prefix))

        if mapping.get('input_image_node_id') and input_image:
            copied_input_filename = safe_copy_image_to_comfy_input(input_image)
            modifier_valeur_noeud_dans_liste(workflow_copy, mapping['input_image_node_id'], 0, copied_input_filename)
            print(f"  - Image d'entrée : {copied_input_filename}")

        if loras and "lora_nodes" in mapping:
            for lora_name, strength in loras.items():
                if lora_name in mapping["lora_nodes"]:
                    node_id = mapping["lora_nodes"][lora_name]
                    modifier_valeur_noeud_dans_liste(workflow_copy, node_id, 1, float(strength))
                    modifier_valeur_noeud_dans_liste(workflow_copy, node_id, 2, float(strength))
                    print(f"  - LoRA '{lora_name}' réglée sur {strength}")

        # --- Exécution ---
        workflow_api = transformer_workflow_pour_api(workflow_copy)
        resultat = mettre_workflow_en_file(workflow_api, client_id)
        if not resultat:
            return False

        print("  [Executor] ✅ Workflow envoyé. Attente du fichier...")
        temp_file_path = attendre_fichier(temp_dir_abs, temp_prefix)
        if not temp_file_path:
            return False

        # --- Finalisation ---
        Path(final_output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_file_path), final_output_path)
        print(f"  [Executor] 🚚 Fichier déplacé vers : {final_output_path}")

        return True

    except Exception as e:
        print(f"  [Erreur Executor] Exception : {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

    finally:
        if copied_input_filename:
            cleanup_comfy_input_file(copied_input_filename)