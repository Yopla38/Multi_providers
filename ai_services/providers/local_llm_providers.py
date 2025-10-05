"""
Providers pour les modèles de langage locaux (DeepSeek et LLaVA).
Ces classes servent de wrappers robustes pour les exécuteurs de modèles,
en ajoutant des fonctionnalités comme les tentatives (retries).
"""
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import json
import logging
import asyncio

from ..config import LocalDeepSeekConfig, LocalLlavaConfig

class LocalDeepSeekProvider:
    """Wrapper de production pour le LocalDeepSeek_R1_Provider."""

    def __init__(self, config: LocalDeepSeekConfig):
        from .utils import clear_vram_if_possible
        clear_vram_if_possible()

        # L'exécuteur sous-jacent est maintenant auto-configuré via les settings globaux
        from .local_deepseek import LocalDeepSeek_R1_Provider
        self.provider = LocalDeepSeek_R1_Provider()
        self.config = config

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((ValueError, json.JSONDecodeError)),
        reraise=True
    )
    async def generate_response(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        pydantic_model: Optional[BaseModel] = None,
        **kwargs
    ) -> Union[str, BaseModel]:
        """Génère une réponse avec une logique de retry."""
        print(f"🤖 DeepSeek R1 - Structuré: {pydantic_model is not None}")

        # Fusionne la configuration du provider avec les arguments d'appel
        generation_params = {
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            **kwargs
        }

        return await self.provider.generate_response(
            prompt=prompt, pydantic_model=pydantic_model, **generation_params
        )

    def generate_response_sync(self, prompt, pydantic_model=None, **kwargs):
        """Wrapper synchrone pour la génération de réponse."""
        return asyncio.run(self.generate_response(prompt, pydantic_model, **kwargs))

    def set_system_prompt(self, system_prompt: str):
        self.provider.set_system_prompt(system_prompt)

    def clear_history(self):
        self.provider.history.clear()


class LocalLLaVAProvider:
    """Wrapper de production pour le LocalMultimodalProvider."""

    def __init__(self, config: LocalLlavaConfig):
        from .utils import clear_vram_if_possible
        clear_vram_if_possible()

        from .local_deepseek import LocalMultimodalProvider
        self.provider = LocalMultimodalProvider()
        self.config = config

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def generate_response(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """Génère une réponse multimodale avec une logique de retry."""
        print(f"👁️ LLaVA - Image: {image_path is not None}")

        generation_params = {
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            **kwargs
        }

        return await self.provider.generate_response(
            prompt=prompt, image_path=image_path, **generation_params
        )

    def generate_response_sync(self, prompt, image_path=None, **kwargs):
        """Wrapper synchrone pour la génération de réponse."""
        return asyncio.run(self.generate_response(prompt, image_path, **kwargs))

    def set_system_prompt(self, system_prompt: str):
        self.provider.set_system_prompt(system_prompt)

    def clear_history(self):
        self.provider.history.clear()