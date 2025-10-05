"""
Wrappers pour les services ComfyUI.
Ces classes servent d'interface standardisée pour les gestionnaires (managers)
afin d'interagir avec les exécuteurs ComfyUI.
"""
from typing import Optional, Dict
from tenacity import retry, wait_exponential, stop_after_attempt

# --- Image Provider ---

class ComfyUIImageProvider:
    """Wrapper pour l'exécuteur de génération d'images ComfyUI."""

    def __init__(self):
        self._executor = None

    @property
    def executor(self):
        if self._executor is None:
            # Importation paresseuse de l'exécuteur
            from .comfy_image_executor import execute_image_workflow
            self._executor = execute_image_workflow
        return self._executor

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def generate(
        self,
        service_name: str,  # Modifié: utilise le nom du service
        output_path: str,
        prompt: str,
        input_image: Optional[str] = None,
        loras: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> bool:
        """Génère une image via ComfyUI en utilisant la configuration du service."""
        print(f"🎨 ComfyUI - Lancement du service d'image : {service_name}")

        return self.executor(
            service_name=service_name,
            final_output_path=output_path,
            prompt=prompt,
            input_image=input_image,
            loras=loras,
            **kwargs
        )

# --- Video Provider ---

class ComfyUIVideoProvider:
    """Wrapper pour l'exécuteur de génération de vidéos ComfyUI."""

    def __init__(self):
        self._executor = None

    @property
    def executor(self):
        if self._executor is None:
            # Importation paresseuse de l'exécuteur vidéo
            from .comfy_executor_video import execute_video_workflow
            self._executor = execute_video_workflow
        return self._executor

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def generate(
        self,
        prompt: str,
        input_image: Optional[str] = None,
        continue_video: bool = False,
        **kwargs
    ) -> bool:
        """Génère une vidéo via ComfyUI."""
        print(f"🎬 ComfyUI - Lancement du service vidéo")

        # Le workflow_path n'est plus nécessaire, il est lu depuis la config
        return self.executor(
            prompt=prompt,
            input_image_path=input_image,
            continue_video=continue_video,
            **kwargs
        )