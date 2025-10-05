"""
Exécuteur ComfyUI pour images (copié de comfy_executor.py original).
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

# --- Configuration ---
COMFYUI_BASE_PATH = os.path.abspath("/home/yopla/Applications/SD/comfyui/ComfyUI/")
COMFYUI_URL = "http://127.0.0.1:8188"
TEMP_OUTPUT_DIR_NAME = "_pipeline_temp"

# --- Mapping des Noeuds ---
NODE_MAPPING = {
    "flux_cinemat.json": {
        "prompt_node_id": 6,
        "save_node_id": 9,
        "seed_node_id": 25,
        "input_image_node_id": None,
        "lora_nodes": {}
    },
    "flux_kontext_master.json": {
        "prompt_node_id": 6,
        "save_node_id": 9,
        "seed_node_id": 25,
        "input_image_node_id": 41,
        "lora_nodes": {
            "nsfw": 59,
            "change_angle": 60
        }
    }
}


# --- Fonctions Utilitaires ---

def charger_json(chemin_fichier):
    with open(chemin_fichier, 'r', encoding='utf-8') as f:
        return json.load(f)


def modifier_valeur_noeud_dans_liste(workflow, id_noeud, index_widget, nouvelle_valeur):
    for node in workflow.get('nodes', []):
        if node.get('id') == id_noeud and 'widgets_values' in node and len(node['widgets_values']) > index_widget:
            node['widgets_values'][index_widget] = nouvelle_valeur
            return True
    return False


def transformer_workflow_pour_api(workflow):
    """Transforme un workflow JSON en format de prompt API."""
    prompt_api = {}

    primitive_values = {}
    for node in workflow['nodes']:
        if node['type'] == 'PrimitiveNode':
            primitive_values[str(node['id'])] = node['widgets_values'][0]

    ui_only_node_types = {'Note', 'PrimitiveNode'}

    for node in workflow['nodes']:
        if node['type'] in ui_only_node_types:
            continue

        node_id_str = str(node['id'])
        api_node = {"class_type": node['type'], "inputs": {}}

        if 'widgets_values' in node:
            widget_names = [inp['name'] for inp in node.get('inputs', []) if 'widget' in inp]

            # Filtre les chaînes de contrôle
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


def mettre_workflow_en_file(workflow_api, client_id):
    try:
        reponse = requests.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow_api, "client_id": client_id}
        )
        reponse.raise_for_status()
        return reponse.json()
    except requests.exceptions.RequestException as e:
        print(f"  [Erreur Executor] : {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"  [Détail] : {e.response.json()}", file=sys.stderr)
            except:
                pass
        return None


def safe_copy_image_to_comfy_input(source_path: str) -> str:
    """
    Copie une image vers le dossier input de ComfyUI de manière sécurisée.

    Args:
        source_path: Chemin source de l'image

    Returns:
        Nom du fichier dans ComfyUI/input/ (juste le nom, pas le chemin)
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Image source introuvable : {source_path}")

    # Dossier input de ComfyUI
    comfy_input_dir = os.path.join(COMFYUI_BASE_PATH, "input")
    os.makedirs(comfy_input_dir, exist_ok=True)

    # Générer un nom unique pour éviter les conflits
    import uuid
    file_ext = os.path.splitext(source_path)[1]
    unique_name = f"pipeline_{uuid.uuid4().hex[:8]}{file_ext}"
    dest_path = os.path.join(comfy_input_dir, unique_name)

    print(f"  [Executor] 📦 Copie de l'image vers ComfyUI/input/...")

    # Taille source
    src_size = os.path.getsize(source_path)
    src_size_mb = src_size / 1024 / 1024
    print(f"  [Executor] 📊 Taille : {src_size_mb:.2f} MB")

    # Copier
    import shutil
    try:
        shutil.copy2(source_path, dest_path)
    except Exception as e:
        print(f"  [Erreur] Échec de la copie : {e}", file=sys.stderr)
        raise

    # Vérifier l'intégrité
    print(f"  [Executor] 🔍 Vérification...")
    max_attempts = 30  # 3 secondes max
    for attempt in range(max_attempts):
        try:
            if os.path.exists(dest_path):
                current_size = os.path.getsize(dest_path)
                if current_size == src_size:
                    print(f"  [Executor] ✅ Image copiée et vérifiée")
                    return unique_name  # Retourner JUSTE le nom, pas le chemin
        except OSError:
            pass
        time.sleep(0.1)

    raise RuntimeError("Timeout lors de la vérification de la copie")


def cleanup_comfy_input_file(filename: str):
    """Nettoie un fichier temporaire du dossier input."""
    try:
        file_path = os.path.join(COMFYUI_BASE_PATH, "input", filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"  [Executor] 🧹 Nettoyé : {filename}")
    except Exception as e:
        print(f"  [Warning] Impossible de nettoyer {filename} : {e}")

def attendre_fichier(output_dir, expected_prefix, timeout=300):
    print(f"  [Executor] Attente du fichier '{expected_prefix}' dans : {output_dir}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            for f in os.listdir(output_dir):
                if f.startswith(expected_prefix) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    found_path = os.path.join(output_dir, f)
                    print(f"  [Executor] ✅ Fichier détecté : {f}")
                    return found_path
        except FileNotFoundError:
            pass
        time.sleep(2.0)
    print(f"  [Executor] ⏱ Timeout après {timeout}s.", file=sys.stderr)
    return None


def execute_image_workflow(
        workflow_name: str,
        final_output_path: str,
        prompt: str,
        input_image: str = None,
        loras: dict = None
):
    """
    Exécute un workflow ComfyUI pour générer une image.

    Args:
        workflow_name: Nom du workflow (ex: "flux_cinemat.json")
        final_output_path: Chemin de sortie final (ex: "/project/Characters/Alice/01_base_nude.png")
        prompt: Le prompt textuel
        input_image: Chemin ABSOLU de l'image d'entrée (optionnel)
        loras: Dict des LoRAs (ex: {"change_angle": 1.0, "nsfw": 0.0})

    Returns:
        True si succès
    """
    client_id = str(uuid.uuid4())
    copied_input_filename = None  # Pour le nettoyage

    try:
        if workflow_name not in NODE_MAPPING:
            print(f"  [Erreur] Workflow '{workflow_name}' non configuré.", file=sys.stderr)
            return False

        # Créer le dossier temporaire
        temp_dir_abs = os.path.join(COMFYUI_BASE_PATH, "output", TEMP_OUTPUT_DIR_NAME)
        os.makedirs(temp_dir_abs, exist_ok=True)
        temp_prefix = str(uuid.uuid4())

        # Charger le workflow
        workflow_path = os.path.join(COMFYUI_BASE_PATH, "workflows", workflow_name)
        if not os.path.exists(workflow_path):
            print(f"  [Erreur] Workflow introuvable : {workflow_path}", file=sys.stderr)
            return False

        workflow = charger_json(workflow_path)
        workflow_copy = copy.deepcopy(workflow)
        mapping = NODE_MAPPING[workflow_name]

        print(f"\n[Executor] Configuration du workflow : {workflow_name}")

        # Modifier les paramètres de base
        modifier_valeur_noeud_dans_liste(workflow_copy, mapping['prompt_node_id'], 0, prompt)
        modifier_valeur_noeud_dans_liste(workflow_copy, mapping['seed_node_id'], 0, random.randint(0, 2 ** 32 - 1))
        modifier_valeur_noeud_dans_liste(workflow_copy, mapping['save_node_id'], 0,
                                         os.path.join(temp_dir_abs, temp_prefix))

        # ═══════════════════════════════════════════════════════════
        # GESTION DE L'IMAGE D'ENTRÉE (CORRECTION CRITIQUE)
        # ═══════════════════════════════════════════════════════════
        if mapping['input_image_node_id'] and input_image:
            # Copier l'image vers ComfyUI/input/ de manière sécurisée
            copied_input_filename = safe_copy_image_to_comfy_input(input_image)

            # Modifier le workflow avec JUSTE le nom du fichier (pas le chemin)
            modifier_valeur_noeud_dans_liste(
                workflow_copy,
                mapping['input_image_node_id'],
                0,
                copied_input_filename  # Juste le nom : "pipeline_abc123.png"
            )
            print(f"  - Image d'entrée : {copied_input_filename}")

        # Configurer les LoRAs
        if loras and "lora_nodes" in mapping:
            for lora_name, strength in loras.items():
                if lora_name in mapping["lora_nodes"]:
                    node_id = mapping["lora_nodes"][lora_name]
                    modifier_valeur_noeud_dans_liste(workflow_copy, node_id, 1, float(strength))
                    modifier_valeur_noeud_dans_liste(workflow_copy, node_id, 2, float(strength))
                    print(f"  - LoRA '{lora_name}' réglée sur {strength}")

        # Transformer et envoyer
        workflow_api = transformer_workflow_pour_api(workflow_copy)
        resultat = mettre_workflow_en_file(workflow_api, client_id)

        if not resultat:
            print("  [Erreur] Échec de l'envoi du workflow.", file=sys.stderr)
            return False

        print(f"  [Executor] ✅ Workflow envoyé. Attente du fichier...")

        # Attendre le fichier
        temp_file_path = attendre_fichier(temp_dir_abs, temp_prefix)
        if not temp_file_path:
            print("  [Erreur] Le fichier de sortie n'a pas été généré.", file=sys.stderr)
            return False

        # Déplacer vers la destination finale
        final_dir = os.path.dirname(final_output_path)
        os.makedirs(final_dir, exist_ok=True)
        shutil.move(temp_file_path, final_output_path)
        print(f"  [Executor] 🚚 Fichier déplacé vers : {final_output_path}")

        return True

    except Exception as e:
        print(f"  [Erreur Executor] Exception : {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Nettoyer l'image temporaire dans input/
        if copied_input_filename:
            cleanup_comfy_input_file(copied_input_filename)