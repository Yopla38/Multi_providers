"""
Wrapper pour les services Replicate.
Cette classe fournit une interface standardisée pour interagir avec l'API
de Replicate pour la génération d'images et potentiellement d'autres médias.
"""
from typing import Optional, Dict
from tenacity import retry, wait_exponential, stop_after_attempt
from .utils import load_api_keys
from ..config import settings

class ReplicateMediaProvider:
    """Wrapper pour les services Replicate."""

    def __init__(self):
        # Accède à la configuration globale via l'objet settings
        self.config = settings.providers.replicate
        self.api_token = load_api_keys().get("REPLICATE_API_TOKEN")
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN manquant dans le fichier .env ou l'environnement.")
        self._generator = None

    @property
    def generator(self):
        """Charge paresseusement le générateur d'images Replicate."""
        if self._generator is None:
            # Correction de l'import pour être relatif au package
            from .replicate_image import ImageGenerator_replicate
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
        """
        Génère une image via Replicate en utilisant le modèle spécifié.
        Si aucun modèle n'est fourni, utilise le modèle par défaut de la configuration.
        """
        # Si aucun model_id n'est passé, utiliser le modèle par défaut pour l'image
        final_model_id = model_id or self.config.default_models.get("image")
        if not final_model_id:
            raise ValueError("Aucun modèle Replicate spécifié et aucun modèle par défaut trouvé.")

        print(f"🌐 Replicate - Modèle: {final_model_id}")

        # Le générateur sous-jacent gère la logique d'appel à Replicate
        outputs = self.generator.generate_image(
            prompt=prompt,
            output_file=output_path,
            model_id=final_model_id,
            lora_model=loras,
            **kwargs
        )

        # Retourne True si au moins une image a été générée avec succès
        return len(outputs) > 0

    def __del__(self):
        """S'assure que les ressources sont nettoyées à la destruction de l'objet."""
        if self._generator:
            self._generator.__exit__(None, None, None)