"""
Gestionnaire unifié pour les services média (images/vidéos).
Ce module fournit une interface simple pour accéder aux fonctionnalités
de génération d'images et de vidéos, en routant les appels vers le
fournisseur (provider) approprié défini dans la configuration.
"""
from pathlib import Path
from typing import Optional, Dict, List

from .config import settings
from .providers.utils import load_api_keys

class MediaManager:
    """
    Gestionnaire unique pour images et vidéos.
    Route automatiquement les appels vers le bon fournisseur (provider)
    en se basant sur la configuration globale.
    """
    def __init__(self):
        # Utilise la configuration globale chargée via Pydantic
        self.config = settings
        self.api_keys = load_api_keys() # Charge depuis .env par défaut
        self._providers = {}

    def _get_provider(self, provider_name: str):
        """Charge un fournisseur à la demande (lazy loading)."""
        if provider_name in self._providers:
            return self._providers[provider_name]

        provider_config = self.config.providers.model_dump().get(provider_name, {})

        if provider_name == "comfyui":
            from .providers.comfyui_providers import ComfyUIImageProvider, ComfyUIVideoProvider
            self._providers["comfyui"] = {
                "image": ComfyUIImageProvider(),
                "video": ComfyUIVideoProvider()
            }
        elif provider_name == "replicate":
            from .providers.replicate_providers import ReplicateMediaProvider
            api_token = self.api_keys.get("REPLICATE_API_TOKEN")
            if not api_token:
                raise ValueError("REPLICATE_API_TOKEN manquant dans le fichier .env")
            self._providers["replicate"] = ReplicateMediaProvider(provider_config, api_token)

        if provider_name not in self._providers:
            raise ValueError(f"Fournisseur inconnu : {provider_name}")

        return self._providers[provider_name]

    def generate_image(
        self,
        prompt: str,
        output_path: str,
        input_image: Optional[str] = None,
        loras: Optional[Dict[str, float]] = None,
        service_override: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Génère une image en utilisant le service approprié."""
        service_name = service_override or ("image_editing" if input_image else "image_generation")

        try:
            service_config = self.config.services[service_name]
        except KeyError:
            raise ValueError(f"Service '{service_name}' non trouvé dans la configuration.")

        provider_name = service_config.provider

        print(f"\n{'='*60}\n🎨 Génération Image - Service: {service_name}\n   Provider: {provider_name}\n   Output: {Path(output_path).name}\n{'='*60}")

        provider = self._get_provider(provider_name)

        if provider_name == "comfyui":
            # Le provider refactorisé attend `service_name` pour trouver sa config
            return provider["image"].generate(
                service_name=service_name,
                output_path=output_path,
                prompt=prompt,
                input_image=input_image,
                loras=loras,
                **kwargs
            )
        elif provider_name == "replicate":
            return provider.generate_image(
                prompt=prompt,
                output_path=output_path,
                input_image=input_image,
                loras=loras,
                model_id=service_config.model, # Utilise le champ 'model'
                **kwargs
            )

        raise ValueError(f"Provider inconnu : {provider_name}")

    def compose_images(self, prompt: str, input_images: List[str], output_path: str, **kwargs) -> bool:
        """Compose plusieurs images en utilisant le service de composition."""
        print(f"\n{'='*60}\n🖼️  Composition - {len(input_images)} images\n{'='*60}")
        # La logique de composition est gérée par le service 'image_composition'
        return self.generate_image(
            prompt=prompt,
            output_path=output_path,
            input_image=input_images[0], # Le workflow gère les multiples images
            service_override="image_composition",
            **kwargs
        )

    def generate_video(self, prompt: str, output_path: str, input_image: Optional[str] = None, continue_video: bool = False, **kwargs) -> bool:
        """Génère une vidéo."""
        service_name = "video_generation"
        service_config = self.config.services[service_name]
        provider_name = service_config.provider

        print(f"\n{'='*60}\n🎬 Génération Vidéo\n   Provider: {provider_name}\n   Continue: {continue_video}\n{'='*60}")

        provider = self._get_provider(provider_name)

        if provider_name == "comfyui":
            # Le provider refactorisé n'a plus besoin de `workflow_path`
            return provider["video"].generate(
                prompt=prompt,
                input_image=input_image,
                continue_video=continue_video,
                **kwargs
            )
        elif provider_name == "replicate":
            raise NotImplementedError("La génération de vidéo avec Replicate n'est pas encore implémentée.")

        raise ValueError(f"Provider inconnu : {provider_name}")

# Instance globale pour un accès simplifié
media = MediaManager()