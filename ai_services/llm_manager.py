"""
Gestionnaire unifié pour les services LLM (texte et multimodal).
Fournit une interface simple pour la génération de texte, l'analyse d'images,
et la gestion de l'historique de conversation.
"""
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel

from .config import settings

class LLMManager:
    """
    Gestionnaire unique pour les LLM.
    Route les appels vers le bon fournisseur en se basant sur la configuration.
    """
    def __init__(self):
        self.config = settings
        self._providers = {}

    def _get_provider(self, service_name: str):
        """Charge un fournisseur LLM à la demande (lazy loading)."""
        if service_name in self._providers:
            return self._providers[service_name]

        try:
            service_config = self.config.services[service_name]
        except KeyError:
            raise ValueError(f"Service LLM '{service_name}' non trouvé dans la configuration.")

        provider_name = service_config.provider
        provider_settings = getattr(self.config.providers, provider_name)

        if provider_name == "local_deepseek":
            from .providers.local_llm_providers import LocalDeepSeekProvider
            self._providers[service_name] = LocalDeepSeekProvider(config=provider_settings)
        elif provider_name == "local_llava":
            from .providers.local_llm_providers import LocalLLaVAProvider
            self._providers[service_name] = LocalLLaVAProvider(config=provider_settings)
        else:
            raise ValueError(f"Fournisseur LLM inconnu : {provider_name}")

        return self._providers[service_name]

    def generate_text(
        self, prompt: Union[str, List[Dict[str, str]]],
        pydantic_model: Optional[BaseModel] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Union[str, BaseModel]:
        """Génère du texte de manière synchrone."""
        provider = self._get_provider("text_generation")
        if system_prompt:
            provider.set_system_prompt(system_prompt)

        print(f"\n{'='*60}\n🤖 Génération Texte | Structuré: {pydantic_model is not None}\n{'='*60}")

        # Correction: utilise la méthode synchrone de la classe de base
        return provider.generate_response_sync(
            prompt=prompt, pydantic_model=pydantic_model, **kwargs
        )

    async def generate_text_async(
        self, prompt: Union[str, List[Dict[str, str]]],
        pydantic_model: Optional[BaseModel] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Union[str, BaseModel]:
        """Génère du texte de manière asynchrone."""
        provider = self._get_provider("text_generation")
        if system_prompt:
            provider.set_system_prompt(system_prompt)

        return await provider.generate_response(
            prompt=prompt, pydantic_model=pydantic_model, **kwargs
        )

    def analyze_image(
        self, prompt: str, image_path: str, system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        """Analyse une image de manière synchrone."""
        provider = self._get_provider("multimodal")
        if system_prompt:
            provider.set_system_prompt(system_prompt)

        print(f"\n{'='*60}\n👁️ Analyse Image | Image: {Path(image_path).name}\n{'='*60}")

        return provider.generate_response_sync(
            prompt=prompt, image_path=image_path, **kwargs
        )

    async def analyze_image_async(
        self, prompt: str, image_path: str, system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        """Analyse une image de manière asynchrone."""
        provider = self._get_provider("multimodal")
        if system_prompt:
            provider.set_system_prompt(system_prompt)

        return await provider.generate_response(
            prompt=prompt, image_path=image_path, **kwargs
        )

    def clear_history(self, service: str = "text_generation"):
        """Efface l'historique de conversation du service spécifié."""
        if service in self._providers:
            self._providers[service].history.clear()
            print(f"🧹 Historique pour le service '{service}' effacé.")

# Instance globale pour un accès simplifié
llm = LLMManager()