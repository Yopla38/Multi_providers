# 🎯 Bibliothèque Complète de Services AI

Voici une architecture **production-ready** qui unifie tout :

## 📁 Structure

```
services/
├── __init__.py
├── config.yaml                 # Configuration des services
├── secrets.env                 # Clés API (gitignored)
├── media_manager.py           # Gestion images/vidéos
├── llm_manager.py             # Gestion LLM/texte
└── providers/
    ├── __init__.py
    ├── comfyui_providers.py   # Wrapper ComfyUI
    ├── replicate_providers.py # Wrapper Replicate
    ├── local_llm_providers.py # DeepSeek + LLaVA
    └── utils.py               # Fonctions communes
```

---

## 1. **config.yaml** - Configuration Principale

```yaml
# Configuration des services média (images/vidéos)
media_services:
  image_generation:
    provider: comfyui
    workflow: flux_cinemat.json
    
  image_editing:
    provider: comfyui
    workflow: flux_kontext_master.json
    
  image_composition:
    provider: comfyui
    workflow: flux_kontext_master.json
    
  video_generation:
    provider: comfyui
    workflow: wan_video_22.json

# Configuration des services LLM (texte)
llm_services:
  text_generation:
    provider: local_deepseek
    model_path: /home/yopla/montage_models/llm_models/python/models/txt2txt/DeepSeek-R1-Distill-Qwen-14B-Q5_K_S/DeepSeek-R1-Distill-Qwen-14B-Q5_K_S.gguf
    
  multimodal:
    provider: local_llava
    model_path: /home/yopla/Documents/llm_models/python/models/multimodal/llava-v1.5-13b-Q6_K.gguf
    clip_path: /home/yopla/Documents/llm_models/python/models/multimodal/mmproj-model-f16.gguf

# Configuration des providers
providers:
  comfyui:
    base_path: /home/yopla/Applications/SD/comfyui/ComfyUI/
    url: http://127.0.0.1:8188
    
  replicate:
    default_models:
      image: bytedance/sdxl-lightning-4step
      video: some-video-model
      
  local_deepseek:
    n_ctx: 8192
    temperature: 0.6
    max_tokens: 8000
    
  local_llava:
    n_ctx: 4096
    temperature: 0.7
    max_tokens: 2048
```

---

## 2. **secrets.env** - Clés API

```bash
# Replicate
REPLICATE_API_TOKEN=r8_YOUR_TOKEN_HERE

# OpenAI (optionnel)
OPENAI_API_KEY=sk-YOUR_KEY_HERE

# Autres services
STABILITY_AI_KEY=sk-YOUR_KEY
```

---

## 3. **providers/utils.py** - Utilitaires Communs

```python
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
```

---

## 4. **providers/comfyui_providers.py** - Wrapper ComfyUI

```python
"""
Wrappers pour les services ComfyUI existants.
Réutilise votre code sans modification.
"""
import os
from typing import Optional, Dict
from tenacity import retry, wait_exponential, stop_after_attempt


class ComfyUIImageProvider:
    """Wrapper pour vos executors ComfyUI existants."""
    
    def __init__(self, config: dict):
        self.config = config
        # Import lazy pour éviter de charger si pas utilisé
        self._executor = None
    
    @property
    def executor(self):
        if self._executor is None:
            from comfy_image_executor import execute_image_workflow
            self._executor = execute_image_workflow
        return self._executor
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def generate(
        self,
        workflow_name: str,
        output_path: str,
        prompt: str,
        input_image: Optional[str] = None,
        loras: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> bool:
        """Génère une image via ComfyUI."""
        print(f"🎨 ComfyUI - {workflow_name}")
        
        return self.executor(
            workflow_name=workflow_name,
            final_output_path=output_path,
            prompt=prompt,
            input_image=input_image,
            loras=loras,
            **kwargs
        )


class ComfyUIVideoProvider:
    """Wrapper pour votre ComfyVideoExecutor."""
    
    def __init__(self, config: dict):
        self.config = config
        self._executor = None
    
    @property
    def executor(self):
        if self._executor is None:
            from comfy_executor_video import ComfyVideoExecutor
            self._executor = ComfyVideoExecutor()
        return self._executor
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def generate(
        self,
        workflow_path: str,
        prompt: str,
        input_image: Optional[str] = None,
        num_frames: int = 81,
        continue_video: bool = False,
        **kwargs
    ) -> bool:
        """Génère une vidéo via ComfyUI."""
        print(f"🎬 ComfyUI Video - {num_frames} frames")
        
        return self.executor.execute_workflow(
            workflow_path=workflow_path,
            prompt=prompt,
            input_image_path=input_image,
            continue_video=continue_video,
            num_frames=num_frames,
            **kwargs
        )
```

---

## 5. **providers/replicate_providers.py** - Wrapper Replicate

```python
"""
Wrapper pour Replicate (réutilise votre ImageGenerator_replicate).
"""
from typing import Optional, Dict, List
from tenacity import retry, wait_exponential, stop_after_attempt


class ReplicateMediaProvider:
    """Wrapper pour vos services Replicate existants."""
    
    def __init__(self, config: dict, api_token: str):
        self.config = config
        self._generator = None
        self.api_token = api_token
    
    @property
    def generator(self):
        if self._generator is None:
            from image_service import ImageGenerator_replicate
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
        """Génère une image via Replicate."""
        print(f"🌐 Replicate - {model_id or 'default model'}")
        
        if not model_id:
            model_id = self.config.get("default_models", {}).get("image")
        
        outputs = self.generator.generate_image(
            prompt=prompt,
            output_file=output_path,
            model_id=model_id,
            lora_model=loras,
            **kwargs
        )
        
        return len(outputs) > 0
    
    def __del__(self):
        if self._generator:
            self._generator.__exit__(None, None, None)
```

---

## 6. **providers/local_llm_providers.py** - LLM Locaux

```python
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
        from providers.utils import clear_vram_if_possible
        clear_vram_if_possible()
        
        # Import votre classe existante
        # Note: Je suppose que votre code est dans un module accessible
        from your_llm_module import LocalDeepSeek_R1_Provider
        
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
```

---

## 7. **media_manager.py** - Gestionnaire Images/Vidéos

```python
"""
Gestionnaire unifié pour les services média (images/vidéos).
"""
import yaml
import os
from pathlib import Path
from typing import Optional, Dict, List
from providers.utils import load_api_keys


class MediaManager:
    """
    Gestionnaire unique pour images et vidéos.
    Route automatiquement vers le bon provider.
    """
    
    def __init__(
        self,
        config_path: str = "services/config.yaml",
        secrets_path: str = "services/secrets.env"
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
            from providers.comfyui_providers import ComfyUIImageProvider, ComfyUIVideoProvider
            self._providers["comfyui"] = {
                "image": ComfyUIImageProvider(provider_config),
                "video": ComfyUIVideoProvider(provider_config)
            }
        
        elif provider_name == "replicate":
            from providers.replicate_providers import ReplicateMediaProvider
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
```

---

## 8. **llm_manager.py** - Gestionnaire LLM

```python
"""
Gestionnaire unifié pour les services LLM (texte et multimodal).
"""
import yaml
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel
from providers.utils import load_api_keys


class LLMManager:
    """
    Gestionnaire unique pour les LLM.
    Support texte simple et multimodal.
    """
    
    def __init__(
        self,
        config_path: str = "services/config.yaml",
        secrets_path: str = "services/secrets.env"
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
            from providers.local_llm_providers import LocalDeepSeekProvider
            
            self._providers[service_name] = LocalDeepSeekProvider(
                config=provider_config,
                model_path=service_config["model_path"]
            )
        
        elif provider_name == "local_llava":
            from providers.local_llm_providers import LocalLLaVAProvider
            
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
```

---

## 9. **__init__.py** - Interface Simple

```python
"""
Interface simplifiée pour la bibliothèque de services AI.
"""
from .media_manager import media
from .llm_manager import llm

__all__ = ['media', 'llm']

__version__ = "1.0.0"
```

---

## 🚀 Utilisation Complète

### Exemple 1: Images

```python
from services import media

# Génération simple
media.generate_image(
    prompt="A beautiful landscape",
    output_path="output/landscape.png",
    loras={"nsfw": 0.5}
)

# Édition (détecte automatiquement grâce à input_image)
media.generate_image(
    prompt="Add a sunset",
    input_image="input/base.png",
    output_path="output/edited.png"
)

# Composition
media.compose_images(
    prompt="Combine characters",
    input_images=["char1.png", "char2.png"],
    output_path="output/composed.png"
)

# Vidéo
media.generate_video(
    prompt="A moving landscape",
    input_image="input/start.png",
    output_path="output/video.mp4",
    num_frames=81
)
```

### Exemple 2: LLM Texte

```python
from services import llm
from pydantic import BaseModel
from typing import List

# Texte simple
response = llm.generate_text("Écris-moi une histoire courte sur un chat.")
print(response)

# Réponse structurée
class Analysis(BaseModel):
    summary: str
    key_points: List[str]
    sentiment: str

result = llm.generate_text(
    prompt="Analyse ce texte: 'Les ventes ont augmenté de 15%...'",
    pydantic_model=Analysis
)
print(result.summary)
print(result.key_points)
```

### Exemple 3: Multimodal

```python
from services import llm

# Analyser une image
description = llm.analyze_image(
    prompt="Décris cette image en détail",
    image_path="input/photo.jpg"
)
print(description)

# Analyse technique d'un graphique (votre cas d'usage)
value = llm.analyze_image(
    prompt="Quelle est la valeur de εzz lorsque Rs = 0.4?",
    image_path="input/graph.jpg",
    system_prompt="Tu es un expert en analyse de graphiques scientifiques."
)
print(value)
```

### Exemple 4: Workflow Complet

```python
from services import media, llm

# 1. Générer une image
media.generate_image(
    prompt="A character in a forest",
    output_path="output/character.png"
)

# 2. Analyser l'image générée
description = llm.analyze_image(
    prompt="Décris ce personnage en détail",
    image_path="output/character.png"
)

# 3. Générer une vidéo basée sur l'analyse
media.generate_video(
    prompt=f"Animate this character: {description}",
    input_image="output/character.png",
    output_path="output/animation.mp4"
)
```

---

## ✨ Points Clés

✅ **Un seul fichier à modifier** : `config.yaml`  
✅ **Clés API séparées** : `secrets.env` (gitignored)  
✅ **Retry automatique** : `tenacity` intégré partout  
✅ **Lazy loading** : Les modèles ne se chargent que si utilisés  
✅ **Gestion VRAM** : Nettoyage automatique  
✅ **Production-ready** : Logging, erreurs, retry  
✅ **Réutilise votre code** : Wrappers autour de l'existant  
✅ **Interface unifiée** : `media` et `llm`  
