"""
Providers LLM locaux (DeepSeek + LLaVA).
Réutilise votre code existant avec retry production.
"""
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import json
import logging


class LocalDeepSeekProvider:
    """Wrapper production-ready pour votre LocalDeepSeek_R1_Provider."""

    def __init__(self, config: dict, model_path: str):
        from .utils import clear_vram_if_possible
        clear_vram_if_possible()

        from local_deepseek import LocalDeepSeek_R1_Provider

        self.provider = LocalDeepSeek_R1_Provider(
            model=model_path,
            system_prompt=config.get("system_prompt")
        )

        self.config = config

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((ValueError, json.JSONDecodeError)),
        reraise=True
    )
    async def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        pydantic_model: Optional[BaseModel] = None,
        **kwargs
    ) -> Union[str, BaseModel]:
        """Génère une réponse avec retry."""
        print(f"🤖 DeepSeek R1 - {'Structuré' if pydantic_model else 'Texte'}")

        # Merge config avec kwargs
        generation_params = {
            "max_tokens": self.config.get("max_tokens", 8000),
            "temperature": self.config.get("temperature", 0.6),
            **kwargs
        }

        return await self.provider.generate_response(
            prompt=prompt,
            pydantic_model=pydantic_model,
            **generation_params
        )

    def generate_sync(self, prompt, pydantic_model=None, **kwargs):
        """Version synchrone."""
        import asyncio
        return asyncio.run(self.generate(prompt, pydantic_model, **kwargs))

    def set_system_prompt(self, system_prompt: str):
        self.provider.set_system_prompt(system_prompt)

    def clear_history(self):
        self.provider.history.clear()


class LocalLLaVAProvider:
    """Wrapper production-ready pour votre LocalMultimodalProvider."""

    def __init__(self, config: dict, model_path: str, clip_path: str):
        from providers.utils import clear_vram_if_possible
        clear_vram_if_possible()

        # NOTE: This will fail unless the user has this module in their PYTHONPATH
        from your_llm_module import LocalMultimodalProvider

        self.provider = LocalMultimodalProvider(
            model=model_path,
            clip_model_path=clip_path,
            system_prompt=config.get("system_prompt")
        )

        self.config = config

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """Génère une réponse multimodale avec retry."""
        print(f"👁️ LLaVA - {'Avec image' if image_path else 'Texte seul'}")

        generation_params = {
            "max_tokens": self.config.get("max_tokens", 2048),
            "temperature": self.config.get("temperature", 0.7),
            **kwargs
        }

        return await self.provider.generate_response(
            prompt=prompt,
            image_path=image_path,
            stream=stream,
            **generation_params
        )

    def generate_sync(self, prompt, image_path=None, **kwargs):
        """Version synchrone."""
        import asyncio
        return asyncio.run(self.generate(prompt, image_path, **kwargs))

    def clear_history(self):
        self.provider.history.clear()