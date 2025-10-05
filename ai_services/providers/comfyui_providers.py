"""
Wrappers pour les services ComfyUI existants.
Réutilise votre code sans modification.
"""
import os
from typing import Optional, Dict
from tenacity import retry, wait_exponential, stop_after_attempt


class ComfyUIImageProvider:
    """Wrapper pour vos executors ComfyUI existants."""

    def __init__(self, config: dict):
        self.config = config
        # Import lazy pour éviter de charger si pas utilisé
        self._executor = None

    @property
    def executor(self):
        if self._executor is None:
            # NOTE: This will fail unless the user has this module in their PYTHONPATH
            from ai_services.providers.comfy_image_executor import execute_image_workflow
            self._executor = execute_image_workflow
        return self._executor

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def generate(
        self,
        workflow_name: str,
        output_path: str,
        prompt: str,
        input_image: Optional[str] = None,
        loras: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> bool:
        """Génère une image via ComfyUI."""
        print(f"🎨 ComfyUI - {workflow_name}")

        return self.executor(
            workflow_name=workflow_name,
            final_output_path=output_path,
            prompt=prompt,
            input_image=input_image,
            loras=loras,
            **kwargs
        )


class ComfyUIVideoProvider:
    """Wrapper pour votre ComfyVideoExecutor."""

    def __init__(self, config: dict):
        self.config = config
        self._executor = None

    @property
    def executor(self):
        if self._executor is None:
            # NOTE: This will fail unless the user has this module in their PYTHONPATH
            from comfy_executor_video import ComfyVideoExecutor
            self._executor = ComfyVideoExecutor()
        return self._executor

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def generate(
        self,
        workflow_path: str,
        prompt: str,
        input_image: Optional[str] = None,
        num_frames: int = 81,
        continue_video: bool = False,
        **kwargs
    ) -> bool:
        """Génère une vidéo via ComfyUI."""
        print(f"🎬 ComfyUI Video - {num_frames} frames")

        return self.executor.execute_workflow(
            workflow_path=workflow_path,
            prompt=prompt,
            input_image_path=input_image,
            continue_video=continue_video,
            num_frames=num_frames,
            **kwargs
        )