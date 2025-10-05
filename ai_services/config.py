import os
import yaml
import shutil
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, DirectoryPath, FilePath, HttpUrl
from typing import Dict, Optional, Any

# ==============================================================================
# CONFIGURATION LOADER
# ==============================================================================
# Ce module charge, valide et expose la configuration depuis config.yaml.
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

class ServiceConfig(BaseModel):
    """Configuration d'un service (ex: image_generation)."""
    provider: str
    workflow: Optional[str] = None  # Pour les providers basés sur des workflows (ex: ComfyUI)
    model: Optional[str] = None     # Pour les providers basés sur des modèles (ex: Replicate)
    node_mapping: Optional[Dict[str, Any]] = None

class ComfyUIConfig(BaseModel):
    """Configuration pour le fournisseur ComfyUI."""
    base_path: DirectoryPath
    url: HttpUrl

    @field_validator('base_path')
    @classmethod
    def resolve_base_path(cls, v):
        # Si le chemin est relatif, le résoudre par rapport à la racine du projet
        if not Path(v).is_absolute():
            return (PROJECT_ROOT / v).resolve()
        return Path(v).resolve()

class ReplicateConfig(BaseModel):
    """Configuration pour le fournisseur Replicate."""
    default_models: Dict[str, str]

class LocalDeepSeekConfig(BaseModel):
    """Configuration pour le fournisseur DeepSeek local."""
    model_path: FilePath
    n_ctx: int = 4096
    temperature: float = 0.7
    max_tokens: int = 2048

    @field_validator('model_path')
    @classmethod
    def resolve_model_path(cls, v):
        if not Path(v).is_absolute():
            return (PROJECT_ROOT / v).resolve()
        return Path(v).resolve()

class LocalLlavaConfig(BaseModel):
    """Configuration pour le fournisseur LLaVA local."""
    model_path: FilePath
    clip_path: FilePath
    n_ctx: int = 4096
    temperature: float = 0.7
    max_tokens: int = 2048

    @field_validator('model_path', 'clip_path')
    @classmethod
    def resolve_model_paths(cls, v):
        if not Path(v).is_absolute():
            return (PROJECT_ROOT / v).resolve()
        return Path(v).resolve()

class ProvidersConfig(BaseModel):
    """Ensemble des configurations pour tous les fournisseurs."""
    comfyui: ComfyUIConfig
    replicate: ReplicateConfig
    local_deepseek: LocalDeepSeekConfig
    local_llava: LocalLlavaConfig

class VideoGenerationConfig(BaseModel):
    """Paramètres pour la génération de segments vidéo."""
    segment_duration: int
    timeout: int
    file_stable_duration: int
    max_segments: int
    vram_release_delay: int

class CoherenceCheckConfig(BaseModel):
    """Paramètres pour la vérification de cohérence."""
    enabled: bool
    adjustment_threshold: str
    contradiction_handling_mode: str
    enable_contradiction_detection: bool

class VideoPipelineConfig(BaseModel):
    """Configuration complète du pipeline vidéo."""
    output_dir: str
    temp_frames_dir: str
    temp_video_filename: str
    generation: VideoGenerationConfig
    coherence_check: CoherenceCheckConfig

    @property
    def output_path(self) -> Path:
        return PROJECT_ROOT / self.output_dir

    @property
    def temp_frames_path(self) -> Path:
        return PROJECT_ROOT / self.temp_frames_dir

class ValidationConfig(BaseModel):
    """Configuration pour les validations au démarrage."""
    check_ffmpeg: bool

class AppConfig(BaseModel):
    """Modèle Pydantic racine pour l'ensemble de la configuration."""
    services: Dict[str, ServiceConfig]
    providers: ProvidersConfig
    video_pipeline: VideoPipelineConfig
    validation: ValidationConfig

    @field_validator('services')
    @classmethod
    def resolve_workflow_paths(cls, services: Dict[str, ServiceConfig]):
        """Résout les chemins des workflows relatifs à la racine du projet."""
        workflows_dir = PROJECT_ROOT / 'workflows'
        for service_name, service_config in services.items():
            if service_config.provider == 'comfyui' and service_config.workflow:
                workflow_path = workflows_dir / service_config.workflow
                if not workflow_path.is_file():
                    raise ValueError(f"Workflow file not found for service '{service_name}': {workflow_path}")
                service_config.workflow = str(workflow_path.resolve())
        return services

def load_config() -> AppConfig:
    """Charge, valide et retourne la configuration de l'application."""
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file 'config.yaml' not found in {config_path.parent}")

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    config = AppConfig(**config_data)

    if config.validation.check_ffmpeg:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("Validation Error: ffmpeg is not installed or not in the system's PATH.")

    config.video_pipeline.output_path.mkdir(parents=True, exist_ok=True)
    config.video_pipeline.temp_frames_path.mkdir(parents=True, exist_ok=True)

    comfyui_temp_dir = config.providers.comfyui.base_path / "temp"
    comfyui_output_dir = config.providers.comfyui.base_path / "output"
    comfyui_temp_dir.mkdir(parents=True, exist_ok=True)
    comfyui_output_dir.mkdir(parents=True, exist_ok=True)

    return config

try:
    settings = load_config()
    print("✅ Configuration loaded and validated successfully.")
except (FileNotFoundError, ValueError, RuntimeError) as e:
    print(f"❌ CONFIGURATION ERROR: {e}")
    settings = None
except Exception as e:
    print(f"❌ An unexpected error occurred during configuration loading: {e}")
    settings = None

if __name__ == "__main__":
    if settings:
        import json
        print(json.dumps(settings.model_dump(mode='json'), indent=2))