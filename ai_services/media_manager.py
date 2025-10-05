"""
Gestionnaire unifié pour les services média (images/vidéos).
"""
import yaml
import os
from pathlib import Path
from typing import Optional, Dict, List
from .providers.utils import load_api_keys


class MediaManager:
    """
    Gestionnaire unique pour images et vidéos.
    Route automatiquement vers le bon provider.
    """

    def __init__(
        self,
        config_path: str = "ai_services/config.yaml",
        secrets_path: str = "ai_services/secrets.env"
    ):
        # Charger configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Charger clés API
        self.api_keys = load_api_keys(secrets_path)

        # Lazy loading des providers
        self._providers = {}

    def _get_provider(self, provider_name: str):
        """Charge un provider à la demande."""
        if provider_name in self._providers:
            return self._providers[provider_name]

        provider_config = self.config["providers"].get(provider_name, {})

        if provider_name == "comfyui":
            from ai_services.providers.comfyui_providers import ComfyUIImageProvider, ComfyUIVideoProvider
            self._providers["comfyui"] = {
                "image": ComfyUIImageProvider(provider_config),
                "video": ComfyUIVideoProvider(provider_config)
            }

        elif provider_name == "replicate":
            from ai_services.providers.replicate_providers import ReplicateMediaProvider
            api_token = self.api_keys.get("REPLICATE_API_TOKEN")
            if not api_token:
                raise ValueError("REPLICATE_API_TOKEN manquant dans secrets.env")

            self._providers["replicate"] = ReplicateMediaProvider(
                provider_config,
                api_token
            )

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
        """
        Génère une image.

        Args:
            prompt: Le prompt
            output_path: Chemin de sortie
            input_image: Image d'entrée (optionnel, active l'édition)
            loras: Dict des LoRAs
            service_override: Force un service spécifique
            **kwargs: Paramètres additionnels
        """
        # Déterminer le service
        if service_override:
            service_name = service_override
        elif input_image:
            service_name = "image_editing"
        else:
            service_name = "image_generation"

        service_config = self.config["media_services"][service_name]
        provider_name = service_config["provider"]

        print(f"\n{'='*60}")
        print(f"🎨 Génération Image - {service_name}")
        print(f"   Provider: {provider_name}")
        print(f"   Output: {Path(output_path).name}")
        print(f"{'='*60}")

        provider = self._get_provider(provider_name)

        if provider_name == "comfyui":
            workflow = service_config["workflow"]
            return provider["image"].generate(
                workflow_name=workflow,
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
                model_id=service_config.get("model"),
                **kwargs
            )

        raise ValueError(f"Provider inconnu : {provider_name}")

    def compose_images(
        self,
        prompt: str,
        input_images: List[str],
        output_path: str,
        **kwargs
    ) -> bool:
        """Compose plusieurs images."""
        service_config = self.config["media_services"]["image_composition"]
        provider_name = service_config["provider"]

        print(f"\n{'='*60}")
        print(f"🖼️  Composition - {len(input_images)} images")
        print(f"   Provider: {provider_name}")
        print(f"{'='*60}")

        # Pour l'instant, utilise la première image
        # À adapter selon votre workflow de composition
        return self.generate_image(
            prompt=prompt,
            output_path=output_path,
            input_image=input_images[0],
            service_override="image_composition",
            **kwargs
        )

    def generate_video(
        self,
        prompt: str,
        output_path: str,
        input_image: Optional[str] = None,
        num_frames: int = 81,
        continue_video: bool = False,
        **kwargs
    ) -> bool:
        """Génère une vidéo."""
        service_config = self.config["media_services"]["video_generation"]
        provider_name = service_config["provider"]

        print(f"\n{'='*60}")
        print(f"🎬 Génération Vidéo - {num_frames} frames")
        print(f"   Provider: {provider_name}")
        print(f"   Continue: {continue_video}")
        print(f"{'='*60}")

        provider = self._get_provider(provider_name)

        if provider_name == "comfyui":
            workflow_path = os.path.join(
                self.config["providers"]["comfyui"]["base_path"],
                "workflows",
                service_config["workflow"]
            )

            return provider["video"].generate(
                workflow_path=workflow_path,
                prompt=prompt,
                input_image=input_image,
                num_frames=num_frames,
                continue_video=continue_video,
                **kwargs
            )

        elif provider_name == "replicate":
            # À implémenter selon l'API Replicate vidéo
            raise NotImplementedError("Vidéo Replicate pas encore implémentée")

        raise ValueError(f"Provider inconnu : {provider_name}")


# Instance globale
media = MediaManager()