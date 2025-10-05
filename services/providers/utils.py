"""
Utilitaires communs pour tous les providers.
"""
import subprocess
import gc
import torch
from typing import Optional


def clear_vram_if_possible():
    """Libère la VRAM PyTorch."""
    try:
        gc.collect()
        if torch.cuda.is_available():
            print("🧹 Nettoyage de la VRAM...")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("✅ VRAM nettoyée.")
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ Échec du nettoyage VRAM : {e}")


def get_optimal_n_gpu_layers(default: int = 0) -> int:
    """Détermine le nombre optimal de couches GPU."""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader']
        )
        free_memory = int(output.decode('utf-8').split('\n')[0].strip())

        if free_memory >= 16000:
            return 35
        elif free_memory >= 8000:
            return 20
        elif free_memory >= 4000:
            return 12
        else:
            return 6
    except Exception as e:
        print(f"⚠️ Aucun GPU détecté : {e}")
        return default


def load_api_keys(secrets_path: str = "secrets.env") -> dict:
    """Charge les clés API depuis un fichier .env."""
    import os
    from pathlib import Path

    keys = {}
    secrets_file = Path(secrets_path)

    if not secrets_file.exists():
        print(f"⚠️ Fichier {secrets_path} introuvable")
        return keys

    with open(secrets_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                keys[key.strip()] = value.strip()

    return keys