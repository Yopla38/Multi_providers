"""
Gestionnaire unifié pour les services LLM (texte et multimodal).
"""
import yaml
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel
from .providers.utils import load_api_keys


class LLMManager:
    """
    Gestionnaire unique pour les LLM.
    Support texte simple et multimodal.
    """

    def __init__(
        self,
        config_path: str = "ai_services/config.yaml",
        secrets_path: str = "ai_services/secrets.env"
    ):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.api_keys = load_api_keys(secrets_path)
        self._providers = {}

    def _get_provider(self, service_name: str):
        """Charge un provider LLM à la demande."""
        if service_name in self._providers:
            return self._providers[service_name]

        service_config = self.config["llm_services"][service_name]
        provider_name = service_config["provider"]
        provider_config = self.config["providers"].get(provider_name, {})

        if provider_name == "local_deepseek":
            from .providers.local_llm_providers import LocalDeepSeekProvider

            self._providers[service_name] = LocalDeepSeekProvider(
                config=provider_config,
                model_path=service_config["model_path"]
            )

        elif provider_name == "local_llava":
            from .providers.local_llm_providers import LocalLLaVAProvider

            self._providers[service_name] = LocalLLaVAProvider(
                config=provider_config,
                model_path=service_config["model_path"],
                clip_path=service_config["clip_path"]
            )

        return self._providers[service_name]

    def generate_text(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        pydantic_model: Optional[BaseModel] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Union[str, BaseModel]:
        """
        Génère du texte (synchrone).

        Args:
            prompt: Prompt ou liste de messages
            pydantic_model: Modèle Pydantic pour réponse structurée
            system_prompt: Prompt système (optionnel)
            **kwargs: Paramètres additionnels
        """
        provider = self._get_provider("text_generation")

        if system_prompt:
            provider.set_system_prompt(system_prompt)

        print(f"\n{'='*60}")
        print(f"🤖 Génération Texte")
        print(f"   Structuré: {pydantic_model is not None}")
        print(f"{'='*60}")

        return provider.generate_sync(
            prompt=prompt,
            pydantic_model=pydantic_model,
            **kwargs
        )

    async def generate_text_async(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        pydantic_model: Optional[BaseModel] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Union[str, BaseModel]:
        """Version asynchrone de generate_text."""
        provider = self._get_provider("text_generation")

        if system_prompt:
            provider.set_system_prompt(system_prompt)

        return await provider.generate(
            prompt=prompt,
            pydantic_model=pydantic_model,
            **kwargs
        )

    def analyze_image(
        self,
        prompt: str,
        image_path: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Analyse une image avec LLaVA (synchrone).

        Args:
            prompt: Question/instruction
            image_path: Chemin de l'image
            system_prompt: Prompt système (optionnel)
            **kwargs: Paramètres additionnels
        """
        provider = self._get_provider("multimodal")

        if system_prompt:
            provider.provider.system_prompt = system_prompt

        print(f"\n{'='*60}")
        print(f"👁️ Analyse Image")
        print(f"   Image: {Path(image_path).name}")
        print(f"{'='*60}")

        return provider.generate_sync(
            prompt=prompt,
            image_path=image_path,
            **kwargs
        )

    async def analyze_image_async(
        self,
        prompt: str,
        image_path: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Version asynchrone de analyze_image."""
        provider = self._get_provider("multimodal")

        if system_prompt:
            provider.provider.system_prompt = system_prompt

        return await provider.generate(
            prompt=prompt,
            image_path=image_path,
            **kwargs
        )

    def clear_history(self, service: str = "text_generation"):
        """Efface l'historique de conversation."""
        if service in self._providers:
            self._providers[service].clear_history()


# Instance globale
llm = LLMManager()