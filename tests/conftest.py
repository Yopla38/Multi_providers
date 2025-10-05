import pytest
import shutil
import tempfile
from pathlib import Path
import os

# --- Fixtures pour l'environnement de test ---

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Retourne le chemin racine du projet."""
    return Path(__file__).parent.parent.resolve()

@pytest.fixture(scope="session")
def mock_workflows_dir(project_root: Path) -> Path:
    """Crée un dossier de workflows de test."""
    d = project_root / "workflows"
    d.mkdir(exist_ok=True)
    # Créer un faux fichier workflow
    (d / "test_workflow.json").write_text('{"mock": true}')
    return d

@pytest.fixture(scope="session")
def mock_models_dir(project_root: Path) -> Path:
    """Crée un dossier de modèles de test avec de faux fichiers."""
    d = project_root / "models"
    d.mkdir(exist_ok=True)
    (d / "deepseek.gguf").touch()
    (d / "llava.gguf").touch()
    (d / "llava_clip.gguf").touch()
    return d

@pytest.fixture(scope="session")
def mock_comfyui_dir(project_root: Path) -> Path:
    """Crée une fausse arborescence pour ComfyUI."""
    d = project_root / "test_comfyui"
    (d / "input").mkdir(parents=True, exist_ok=True)
    (d / "output").mkdir(parents=True, exist_ok=True)
    (d / "temp").mkdir(parents=True, exist_ok=True)
    return d

# --- Fixture principale de configuration ---

@pytest.fixture(scope="session")
def mock_config_data(mock_workflows_dir, mock_models_dir, mock_comfyui_dir) -> dict:
    """
    Fournit un dictionnaire de configuration complet et valide pour les tests.
    Pointe vers les fichiers et dossiers de test créés par les autres fixtures.
    """
    return {
        "services": {
            "image_generation_comfy": {
                "provider": "comfyui",
                "workflow": "test_workflow.json",
                "node_mapping": {"prompt_node_id": 1}
            },
            "image_generation_replicate": {
                "provider": "replicate",
                "model": "test/replicate-model:version"
            },
            "video_generation": {
                "provider": "comfyui",
                "workflow": "test_workflow.json",
                "node_mapping": {"prompt_node_id": 1}
            },
            "text_generation": {"provider": "local_deepseek"},
            "multimodal": {"provider": "local_llava"}
        },
        "providers": {
            "comfyui": {
                "base_path": str(mock_comfyui_dir),
                "url": "http://127.0.0.1:8188"
            },
            "replicate": {
                "default_models": {
                    "image": "default/replicate-image:version",
                    "video": "default/replicate-video:version"
                }
            },
            "local_deepseek": {
                "model_path": str(mock_models_dir / "deepseek.gguf"),
                "n_ctx": 2048, "temperature": 0.1, "max_tokens": 512
            },
            "local_llava": {
                "model_path": str(mock_models_dir / "llava.gguf"),
                "clip_path": str(mock_models_dir / "llava_clip.gguf"),
                "n_ctx": 2048, "temperature": 0.1, "max_tokens": 512
            }
        },
        "video_pipeline": {
            "output_dir": "test_output/videos",
            "temp_frames_dir": "test_output/temp_frames",
            "temp_video_filename": "test_video.mp4",
            "generation": {
                "segment_duration": 1, "timeout": 30, "file_stable_duration": 1,
                "max_segments": 2, "vram_release_delay": 0
            },
            "coherence_check": {
                "enabled": False, "adjustment_threshold": "good",
                "contradiction_handling_mode": "auto", "enable_contradiction_detection": False
            }
        },
        "validation": {"check_ffmpeg": False}
    }

# --- Fixture pour patcher la configuration ---

@pytest.fixture(autouse=True)
def patch_config(monkeypatch, mock_config_data):
    """
    Fixture auto-utilisée qui intercepte l'appel à `load_config`
    et le remplace par une fonction qui retourne notre configuration de test.
    Ceci assure que tous les tests utilisent la configuration mockée.
    """
    from ai_services import config
    from ai_services.config import AppConfig

    # Créer une instance de AppConfig à partir de nos données de test
    # C'est ici que Pydantic valide notre configuration de test
    mock_settings_object = AppConfig(**mock_config_data)

    # Remplacer l'objet 'settings' dans le module config
    monkeypatch.setattr(config, 'settings', mock_settings_object)

    # Remplacer aussi la fonction de chargement pour être sûr
    def mock_load_config():
        return mock_settings_object

    monkeypatch.setattr(config, 'load_config', mock_load_config)

    # On peut aussi patcher les utilitaires si nécessaire
    from ai_services.providers import utils
    def mock_load_api_keys():
        return {"REPLICATE_API_TOKEN": "r8_test_token"}

    monkeypatch.setattr(utils, 'load_api_keys', mock_load_api_keys)

    yield mock_settings_object

    # Nettoyage après les tests (si nécessaire)
    shutil.rmtree(mock_config_data["providers"]["comfyui"]["base_path"], ignore_errors=True)
    shutil.rmtree(Path(mock_config_data["providers"]["local_deepseek"]["model_path"]).parent, ignore_errors=True)
    shutil.rmtree(Path(mock_config_data["services"]["video_generation"]["workflow"]).parent, ignore_errors=True)