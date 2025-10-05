"""
Exécuteur spécialisé pour le workflow Wan Video 2.2.
VERSION SIMPLIFIÉE : Modification directe du JSON UI.
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

from .config import (
    COMFYUI_URL,
    COMFYUI_BASE_PATH,
    NODE_MAPPING,
    FILE_STABLE_DURATION,
    VIDEO_GENERATION_TIMEOUT,
    TEMP_VIDEO_PATH
)


class ComfyVideoExecutor:
    """Exécuteur pour le workflow Wan Video 2.2."""

    def __init__(self):
        self.client_id = str(uuid.uuid4())
        self.base_url = COMFYUI_URL

    def load_workflow(self, workflow_path: str) -> dict:
        """Charge un workflow JSON au format API."""
        with open(workflow_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def fix_anything_everywhere(self, workflow_api: dict) -> dict:
        """
        Corrige le workflow API en résolvant les broadcasts "Anything Everywhere".

        Les nodes "Anything Everywhere" ne fonctionnent pas dans le format API.
        On doit injecter manuellement les connexions manquantes.
        """
        print(f"   🔧 Résolution des broadcasts...")

        # Hardcoder les connexions connues pour ce workflow
        # (Détectées en analysant les nodes "Anything Everywhere")

        # Node 16 (WanVideoTextEncode) : manque le T5 encoder
        if "16" in workflow_api:
            if "t5" not in workflow_api["16"]["inputs"]:
                workflow_api["16"]["inputs"]["t5"] = ["11", 0]
                print(f"      ✓ Node 16 : ajout t5")

        # Node 28 (WanVideoDecode) : manque le VAE
        if "28" in workflow_api:
            if "vae" not in workflow_api["28"]["inputs"]:
                workflow_api["28"]["inputs"]["vae"] = ["38", 0]
                print(f"      ✓ Node 28 : ajout vae")

        # Node 538 (WanVideoImageToVideoEncode) : manque le VAE
        if "538" in workflow_api:
            if "vae" not in workflow_api["538"]["inputs"]:
                workflow_api["538"]["inputs"]["vae"] = ["38", 0]
                print(f"      ✓ Node 538 : ajout vae")

        # Node 579 (WanVideoSampler high) : manque text_embeds et feta_args
        if "579" in workflow_api:
            if "text_embeds" not in workflow_api["579"]["inputs"]:
                workflow_api["579"]["inputs"]["text_embeds"] = ["16", 0]
                print(f"      ✓ Node 579 : ajout text_embeds")
            if "feta_args" not in workflow_api["579"]["inputs"]:
                workflow_api["579"]["inputs"]["feta_args"] = ["55", 0]
                print(f"      ✓ Node 579 : ajout feta_args")

        # Node 581 (WanVideoSampler low) : manque text_embeds et feta_args
        if "581" in workflow_api:
            if "text_embeds" not in workflow_api["581"]["inputs"]:
                workflow_api["581"]["inputs"]["text_embeds"] = ["16", 0]
                print(f"      ✓ Node 581 : ajout text_embeds")
            if "feta_args" not in workflow_api["581"]["inputs"]:
                workflow_api["581"]["inputs"]["feta_args"] = ["55", 0]
                print(f"      ✓ Node 581 : ajout feta_args")

        # Supprimer les nodes "Anything Everywhere" (optionnel, ils sont ignorés de toute façon)
        for node_id in ["57", "226", "229", "230"]:
            if node_id in workflow_api:
                del workflow_api[node_id]

        return workflow_api

    def modify_workflow_api(
            self,
            workflow_api: dict,
            prompt: str,
            continue_video: bool = False,
            input_image_path: Optional[str] = None,
            num_frames: int = 81,
            steps: int = 8,
            resolution: int = 576
    ) -> dict:
        """Modifie un workflow au format API."""
        import random

        workflow_copy = json.loads(json.dumps(workflow_api))

        print(f"📝 Configuration du workflow API :")

        # 1. Prompt (Node 59)
        if "59" in workflow_copy:
            workflow_copy["59"]["inputs"]["text"] = prompt
            print(f"   ✓ Prompt modifié")

        # 2. Continue video (Node 610)
        if "610" in workflow_copy:
            workflow_copy["610"]["inputs"]["value"] = continue_video
            print(f"   ✓ Continue = {continue_video}")

        # 3. Image d'entrée (Node 433)
        if input_image_path and "433" in workflow_copy:
            filename = os.path.basename(input_image_path)
            workflow_copy["433"]["inputs"]["image"] = filename
            print(f"   ✓ Nom de fichier image défini dans le workflow : {filename}")
        else:
            print(f"   ⚠️  Avertissement: Aucun chemin d'image fourni à modify_workflow_api.")

        # 4. Résolution (Node 601)
        if "601" in workflow_copy:
            workflow_copy["601"]["inputs"]["Xi"] = resolution
            print(f"   ✓ Résolution = {resolution}")

        # 5. Frames (Node 439)
        if "439" in workflow_copy:
            workflow_copy["439"]["inputs"]["Xi"] = num_frames
            print(f"   ✓ Frames = {num_frames}")

        # 6. Steps (Node 440)
        if "440" in workflow_copy:
            workflow_copy["440"]["inputs"]["Xi"] = steps
            print(f"   ✓ Steps = {steps}")

        # 7. Seeds aléatoires
        if "579" in workflow_copy:
            new_seed = random.randint(0, 2 ** 32 - 1)
            workflow_copy["579"]["inputs"]["seed"] = new_seed
            print(f"   ✓ Seed (579) = {new_seed}")

        if "581" in workflow_copy:
            new_seed_581 = random.randint(0, 2 ** 32 - 1)
            workflow_copy["581"]["inputs"]["seed"] = new_seed_581
            print(f"   ✓ Seed (581) = {new_seed_581}")

        return workflow_copy

    def copy_image_to_comfyui(self, image_path: str) -> str:
        """Copie l'image dans le dossier input de ComfyUI."""
        comfyui_input = os.path.join(COMFYUI_BASE_PATH, "input")
        os.makedirs(comfyui_input, exist_ok=True)
        dest_path = os.path.join(comfyui_input, os.path.basename(image_path))

        # Ajout d'une simple vérification pour éviter de copier si inutile
        if not os.path.exists(dest_path):
            shutil.copy(image_path, dest_path)
            print(f"   ✓ Image copiée: {os.path.basename(image_path)}")
        else:
            print(f"   ✓ Image déjà présente: {os.path.basename(image_path)}")

        return os.path.basename(image_path)

    def queue_prompt(self, workflow_api: dict) -> Optional[str]:
        """Envoie le workflow API à ComfyUI."""
        try:
            response = requests.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow_api, "client_id": self.client_id},
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            prompt_id = result.get('prompt_id')

            if prompt_id:
                print(f"   ✓ Workflow envoyé (prompt_id: {prompt_id})")
                return prompt_id
            return None

        except requests.exceptions.RequestException as e:
            print(f"   ❌ Erreur : {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    print(f"   Détail : {e.response.json()}")
                except:
                    pass
            return None

    def wait_for_completion(
            self,
            video_path: str = TEMP_VIDEO_PATH,
            timeout: int = VIDEO_GENERATION_TIMEOUT
    ) -> bool:
        """
        Attend que la vidéo soit complètement générée.

        Logique améliorée :
        1. Enregistrer la taille initiale du fichier (si existe)
        2. Attendre que la taille CHANGE (preuve que ComfyUI a commencé)
        3. Puis attendre que la taille se STABILISE (génération terminée)
        """
        print(f"   ⏳ Attente de la génération...")

        start_time = time.time()

        # ═══════════════════════════════════════════════════════════
        # ÉTAPE 1 : Enregistrer la taille initiale
        # ═══════════════════════════════════════════════════════════
        initial_size = None
        if os.path.exists(video_path):
            try:
                initial_size = os.path.getsize(video_path)
                print(f"   📦 Fichier existant détecté ({initial_size / 1024 / 1024:.2f} MB)")
            except OSError:
                initial_size = None

        # ═══════════════════════════════════════════════════════════
        # ÉTAPE 2 : Attendre que ComfyUI commence à modifier le fichier
        # ═══════════════════════════════════════════════════════════
        generation_started = False

        if initial_size is not None:
            print(f"   🔍 Attente du début de modification...")
            while time.time() - start_time < timeout:
                if os.path.exists(video_path):
                    try:
                        current_size = os.path.getsize(video_path)
                        if current_size != initial_size:
                            print(f"   🎬 Génération démarrée (taille change: {current_size / 1024 / 1024:.2f} MB)")
                            generation_started = True
                            break
                    except OSError:
                        pass
                time.sleep(1)

            if not generation_started:
                print(f"   ❌ Timeout : le fichier n'a pas été modifié")
                return False

        # ═══════════════════════════════════════════════════════════
        # ÉTAPE 3 : Attendre que la taille se stabilise
        # ═══════════════════════════════════════════════════════════
        last_size = initial_size if initial_size else 0
        last_change_time = time.time()

        print(f"   📊 Surveillance de la stabilité...")

        while time.time() - start_time < timeout:
            if os.path.exists(video_path):
                try:
                    current_size = os.path.getsize(video_path)

                    if current_size != last_size:
                        # La taille a changé, la génération continue
                        last_size = current_size
                        last_change_time = time.time()
                        elapsed = int(time.time() - start_time)
                        print(f"   📊 {elapsed}s - {current_size / 1024 / 1024:.2f} MB")

                    # Vérifier la stabilité
                    stable_time = time.time() - last_change_time
                    if stable_time >= FILE_STABLE_DURATION and current_size > 0:
                        print(f"   ✅ Vidéo stable ({current_size / 1024 / 1024:.2f} MB)")
                        return True

                except OSError:
                    # Fichier en cours d'écriture, on continue
                    pass

            time.sleep(2)

        print(f"   ❌ Timeout après {timeout}s")
        return False

    def extract_last_frame(self, video_path: str, output_path: str) -> bool:
        """Extrait la dernière frame."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            subprocess.run([
                "ffmpeg", "-sseof", "-0.5", "-i", video_path,
                "-update", "1", "-q:v", "1", "-y", output_path
            ], capture_output=True, check=True, timeout=30)

            if Path(output_path).exists():
                print(f"   ✓ Frame extraite")
                return True
            return False

        except Exception as e:
            print(f"   ❌ Erreur : {e}")
            return False

    def execute_workflow(
            self,
            workflow_path: str,
            prompt: str,
            continue_video: bool = False,
            input_image_path: Optional[str] = None,
            **kwargs
    ) -> bool:
        """Exécute le workflow."""
        print(f"\n{'─' * 60}")
        print(f"🎬 Exécution")
        print(f"{'─' * 60}")

        # Supprimer l'ancien fichier SEULEMENT si ce n'est PAS une continuation
        if not continue_video and os.path.exists(TEMP_VIDEO_PATH):
            os.remove(TEMP_VIDEO_PATH)
            print(f"🗑️  Ancien fichier supprimé (nouveau segment)")
        elif continue_video and os.path.exists(TEMP_VIDEO_PATH):
            print(f"♻️  Fichier existant conservé (continuation)")
        elif continue_video and not os.path.exists(TEMP_VIDEO_PATH):
            print(f"⚠️  ATTENTION : Continue=True mais aucun fichier vidéo précédent trouvé !")
            print(f"   Chemin attendu : {TEMP_VIDEO_PATH}")
            print(f"   → La génération va probablement échouer.")

        if not continue_video:
            try:
                self.copy_image_to_comfyui(input_image_path)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"   ❌ ERREUR lors de la préparation de l'image : {e}", file=sys.stderr)
                return False

        # Charger le workflow API
        workflow_api = self.load_workflow(workflow_path)

        # Corriger les "Anything Everywhere"
        workflow_api = self.fix_anything_everywhere(workflow_api)

        # Modifier le workflow API
        modified_workflow = self.modify_workflow_api(
            workflow_api,
            prompt=prompt,
            continue_video=continue_video,
            input_image_path=input_image_path,
            **kwargs
        )

        # VÉRIFICATION DE SÉCURITÉ : Si continue_video, vérifier TempLoop.mp4
        if continue_video:
            if not os.path.exists(TEMP_VIDEO_PATH):
                print(f"   ❌ ERREUR : Continue=True mais TempLoop.mp4 introuvable")
                return False

            # Vérifier que le fichier n'est pas en cours d'écriture
            print(f"   🔍 Vérification de TempLoop.mp4...")
            initial_size = os.path.getsize(TEMP_VIDEO_PATH)
            time.sleep(1)  # Attendre 1 seconde

            try:
                final_size = os.path.getsize(TEMP_VIDEO_PATH)
                if initial_size != final_size:
                    print(f"   ⚠️  TempLoop.mp4 en cours de modification, attente...")
                    time.sleep(3)
                    final_size = os.path.getsize(TEMP_VIDEO_PATH)

                print(f"   ✅ TempLoop.mp4 stable ({final_size / 1024 / 1024:.2f} MB)")
            except OSError as e:
                print(f"   ❌ Erreur d'accès à TempLoop.mp4 : {e}")
                return False

        # Envoyer
        print(f"🚀 Envoi à ComfyUI...")
        prompt_id = self.queue_prompt(modified_workflow)

        if not prompt_id:
            return False

        # Attendre
        success = self.wait_for_completion()

        if success:
            print(f"✅ Génération terminée")
        else:
            print(f"❌ Échec")

        return success