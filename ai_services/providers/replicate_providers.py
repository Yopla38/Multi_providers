"""
Wrapper pour Replicate (réutilise votre ImageGenerator_replicate).
"""
from typing import Optional, Dict, List
from tenacity import retry, wait_exponential, stop_after_attempt


class ReplicateMediaProvider:
    """Wrapper pour vos services Replicate existants."""

    def __init__(self, config: dict, api_token: str):
        self.config = config
        self._generator = None
        self.api_token = api_token

    @property
    def generator(self):
        if self._generator is None:
            # NOTE: This will fail unless the user has this module in their PYTHONPATH
            from replicate_image import ImageGenerator_replicate
            self._generator = ImageGenerator_replicate(
                api_token=self.api_token,
                max_workers=10
            )
        return self._generator

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def generate_image(
        self,
        prompt: str,
        output_path: str,
        model_id: Optional[str] = None,
        input_image: Optional[str] = None,
        loras: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> bool:
        """Génère une image via Replicate."""
        print(f"🌐 Replicate - {model_id or 'default model'}")

        if not model_id:
            model_id = self.config.get("default_models", {}).get("image")

        outputs = self.generator.generate_image(
            prompt=prompt,
            output_file=output_path,
            model_id=model_id,
            lora_model=loras,
            **kwargs
        )

        return len(outputs) > 0

    def __del__(self):
        if self._generator:
            self._generator.__exit__(None, None, None)