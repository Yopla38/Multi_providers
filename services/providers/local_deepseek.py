from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
import asyncio
import base64
import gc
import io
import json
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union, AsyncGenerator, Tuple

import instructor
import requests
import torch
from PIL import Image
from pydantic import BaseModel
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from llama_cpp import Llama
from openai import OpenAI

# CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FORCE_CUBLAS=on -DLLAVA_BUILD=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade

LLAVA_MODEL_PATH = "/home/yopla/Documents/llm_models/python/models/multimodal/llava-v1.5-13b-Q6_K.gguf"
CLIP_LLAVA_MODEL_PATH = "/home/yopla/Documents/llm_models/python/models/multimodal/mmproj-model-f16.gguf"


@dataclass
class Message:
    """Classe représentant un message dans l'historique"""
    role: str  # 'user' ou 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class ConversationHistory:
    def __init__(self, max_messages: int = 10):
        self.messages: List[Message] = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_formated_messages(self) -> List[Dict[str, str]]:
        """Retourne l'historique formaté pour les APIs de chat."""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def clear(self):
        self.messages.clear()


def clear_vram_if_possible():
    """
    Tente de libérer la VRAM de PyTorch si le module est disponible.
    Appelée avant de charger un nouveau modèle lourd en mémoire.
    """
    try:
        import gc
        import torch

        # Étape 1: Forcer le garbage collector de Python
        gc.collect()

        # Étape 2: Vider le cache de PyTorch si CUDA est disponible
        if torch.cuda.is_available():
            print("🧹 Nettoyage de la VRAM (cache PyTorch)...")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("✅ VRAM nettoyée.")

    except ImportError:
        # Si torch n'est pas installé, on ne fait rien
        pass
    except Exception as e:
        print(f"⚠️ Avertissement : Échec de la tentative de nettoyage de la VRAM : {e}")


class LLMProvider(ABC):
    def __init__(self):
        self.history = ConversationHistory()
        self.log_file = 'GPT_log.json'

    @abstractmethod
    async def generate_response(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        pass

    @abstractmethod
    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=None) -> Dict[str, Any]:
        pass

    def write_log(self, receive_text=None):
        if self.log_file:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log = f"Timestamp : {timestamp}\n{receive_text}"
            with open(self.log_file, 'w') as f:
                f.write(log)

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings pour une liste de textes"""
        pass

    def generate_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """Version synchrone pour générer des embeddings"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_embeddings(texts))
        finally:
            loop.close()

    def generate_response_sync(self, messages, **kwargs):
        """Version synchrone pour générer une réponse"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_response(messages, **kwargs))
        finally:
            loop.close()

    def set_system_prompt(self, system_prompt):
        pass


def get_optimal_n_gpu_layers(default: int = 0) -> int:
    """
    Détermine dynamiquement le nombre optimal de couches à exécuter sur le GPU
    """
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader']
        )
        free_memory_str = output.decode('utf-8').split('\n')[0].strip()
        free_memory = int(free_memory_str)

        if free_memory >= 16000:
            return 35
        elif free_memory >= 8000:
            return 20
        elif free_memory >= 4000:
            return 12
        else:
            return 6
    except Exception as e:
        print("Aucun GPU détecté ou erreur lors de la détection de la mémoire GPU :", e)
        return default


# Wrapper personnalisé pour llama-cpp-python compatible avec l'API OpenAI
class LlamaCppOpenAIWrapper:
    """Wrapper pour rendre llama-cpp-python compatible avec l'interface OpenAI attendue par Instructor"""

    def __init__(self, llama_model: Llama):
        self.llama_model = llama_model
        self.base_url = "http://localhost:8000"  # URL fictive pour la compatibilité
        self.api_key = "local"  # Clé fictive pour la compatibilité
        self.chat = self
        self.completions = self

    def create(self, messages: List[Dict[str, str]], **kwargs):
        """Interface compatible OpenAI pour les chat completions"""

        # Extraire response_model des kwargs s'il existe
        response_model = kwargs.pop('response_model', None)

        # Convertir les messages au format DeepSeek R1
        formatted_prompt = self._format_deepseek_r1_prompt(messages)

        # Paramètres recommandés pour DeepSeek R1
        generation_params = {
            "prompt": formatted_prompt,
            "max_tokens": kwargs.get("max_tokens", 4000),
            "temperature": kwargs.get("temperature", 0.6),  # Recommandation DeepSeek
            "top_p": kwargs.get("top_p", 0.95),
            "stop": [
                "<|begin_of_sentence|>",
                "<|end_of_sentence|>",
                "<|User|>",
                "<|Assistant",
                "<｜User｜>",
                "<｜end▁of▁sentence｜>"
            ],
            "echo": False
        }

        # Générer la réponse
        response = self.llama_model(**generation_params)

        # Extraire le texte de la réponse
        response_text = response["choices"][0]["text"].strip()

        # Pour DeepSeek R1, extraire le contenu après les balises de raisonnement
        if response_model:
            # Extraire le contenu de raisonnement et la réponse finale
            reasoning_content, final_content = self._extract_reasoning_and_content(response_text)

            # Créer un mock de la réponse OpenAI avec reasoning_content
            class MockMessage:
                def __init__(self, content, reasoning_content=None):
                    self.content = content
                    self.role = "assistant"
                    self.reasoning_content = reasoning_content

            class MockChoice:
                def __init__(self, content, reasoning_content=None):
                    self.message = MockMessage(content, reasoning_content)
                    self.finish_reason = "stop"
                    self.index = 0

            class MockCompletion:
                def __init__(self, content, reasoning_content=None):
                    self.choices = [MockChoice(content, reasoning_content)]
                    self.id = f"chatcmpl-{hash(content) % 1000000}"
                    self.object = "chat.completion"
                    self.created = int(datetime.now().timestamp())
                    self.model = "deepseek-r1"

            return MockCompletion(final_content, reasoning_content)
        else:
            # Réponse simple sans structure
            class MockMessage:
                def __init__(self, content):
                    self.content = content
                    self.role = "assistant"

            class MockChoice:
                def __init__(self, content):
                    self.message = MockMessage(content)
                    self.finish_reason = "stop"
                    self.index = 0

            class MockCompletion:
                def __init__(self, content):
                    self.choices = [MockChoice(content)]
                    self.id = f"chatcmpl-{hash(content) % 1000000}"
                    self.object = "chat.completion"
                    self.created = int(datetime.now().timestamp())
                    self.model = "deepseek-r1"

            return MockCompletion(response_text)

    def _format_deepseek_r1_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Formate les messages selon le template DeepSeek-R1"""
        formatted = ""

        for i, message in enumerate(messages):
            is_last = (i == len(messages) - 1)

            if message["role"] == "user":
                formatted += f"<｜User｜>{message['content']}\n\n"
            elif message["role"] == "assistant":
                formatted += f"<｜Assistant｜>{message['content']}"
                if not is_last:
                    formatted += "<｜end▁of▁sentence｜>\n\n"
            elif message["role"] == "system":
                formatted = message['content'] + "\n\n" + formatted

        # Si le dernier message n'est pas de l'assistant, ajouter le prompt
        if messages and messages[-1]["role"] != "assistant":
            formatted += "<｜Assistant｜>"

        return formatted

    def _extract_reasoning_and_content(self, text: str) -> tuple[str, str]:
        """
        Extrait le contenu de raisonnement (balises <think>) et le contenu final pour DeepSeek R1
        """
        # Extraire le contenu de raisonnement entre <think> et </think>
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, text, re.DOTALL)
        reasoning_content = "\n".join(think_matches).strip() if think_matches else ""

        # Extraire le contenu final après les balises de raisonnement
        # Supprimer tout le contenu entre <think> et </think>
        final_content = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()

        return reasoning_content, final_content


class LocalDeepSeek_R1_Provider(LLMProvider):
    def __init__(self, model: str, api_key: Optional[str] = None, system_prompt: Optional[str] = None):
        super().__init__()
        clear_vram_if_possible()
        self.system_prompt = system_prompt
        self.executor = ThreadPoolExecutor(max_workers=1)

        gpu_layers = get_optimal_n_gpu_layers(default=0)
        print(f"Nombre de couches gpu utilisées : {str(gpu_layers)}")

        # Charger le modèle Llama avec les paramètres recommandés pour DeepSeek-R1
        self.llm = Llama(
            model_path=model,
            n_gpu_layers=gpu_layers,
            n_ctx=8192,  # Contexte adapté pour DeepSeek-R1
            verbose=False,
            chat_format=None,  # Pas de format prédéfini
        )

        # Créer le wrapper OpenAI-compatible
        self.openai_wrapper = LlamaCppOpenAIWrapper(self.llm)

        # Configurer Instructor avec le mode MD_JSON comme recommandé pour DeepSeek
        self.instructor_client = instructor.patch(
            self.openai_wrapper,
            mode=instructor.Mode.MD_JSON  # Mode recommandé pour DeepSeek
        )

    @retry(
        wait=wait_random_exponential(min=1, max=10),
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
        """
        Génère une réponse structurée ou non avec DeepSeek-R1 et Instructor
        """
        # Convertir le prompt en liste de messages
        if isinstance(prompt, str):
            self.history.add_message("user", prompt)
            messages = []
            # Éviter d'ajouter un message système selon les recommandations DeepSeek
            if self.system_prompt:
                # Intégrer le système dans le message utilisateur plutôt que comme message système séparé
                formatted_messages = self.history.get_formated_messages()
                if formatted_messages:
                    first_user_msg = formatted_messages[0].copy()
                    first_user_msg["content"] = f"{self.system_prompt}\n\n{first_user_msg['content']}"
                    messages = [first_user_msg] + formatted_messages[1:]
                else:
                    messages = formatted_messages
            else:
                messages = self.history.get_formated_messages()
        else:
            messages = prompt.copy()
            # Éviter les messages système séparés
            if self.system_prompt and not any(msg.get("role") == "system" for msg in messages):
                # Intégrer dans le premier message utilisateur
                for msg in messages:
                    if msg["role"] == "user":
                        msg["content"] = f"{self.system_prompt}\n\n{msg['content']}"
                        break

            # Ajouter à l'historique
            for msg in prompt:
                if msg["role"] == "user":
                    self.history.add_message("user", msg["content"])

        loop = asyncio.get_event_loop()

        try:
            if pydantic_model:
                # Utiliser Instructor pour les réponses structurées
                response = await loop.run_in_executor(
                    self.executor,
                    lambda: self.instructor_client.chat.completions.create(
                        messages=messages,
                        response_model=pydantic_model,
                        max_tokens=kwargs.get("max_tokens", 8000),
                        temperature=kwargs.get("temperature", 0.6)  # Recommandation DeepSeek
                    )
                )

                # Ajouter la réponse à l'historique
                if hasattr(response, 'model_dump_json'):
                    self.history.add_message("assistant", response.model_dump_json(indent=2))
                else:
                    self.history.add_message("assistant", str(response))
                return response
            else:
                # Réponse texte simple - ne pas passer response_model=None explicitement
                response = await loop.run_in_executor(
                    self.executor,
                    lambda: self.openai_wrapper.create(
                        messages=messages,
                        max_tokens=kwargs.get("max_tokens", 8000),
                        temperature=kwargs.get("temperature", 0.6)
                    )
                )

                response_text = response.choices[0].message.content
                self.history.add_message("assistant", response_text)
                return response_text

        except Exception as e:
            logging.error(f"Erreur lors de la génération avec DeepSeek-R1 et Instructor : {e}")
            # Fallback simple
            try:
                fallback_response = await loop.run_in_executor(
                    self.executor,
                    lambda: self.openai_wrapper.create(
                        messages=messages,
                        max_tokens=kwargs.get("max_tokens", 4000),
                        temperature=kwargs.get("temperature", 0.6)
                    )
                )
                fallback_text = fallback_response.choices[0].message.content
                self.history.add_message("assistant", fallback_text)
                return fallback_text
            except Exception as fallback_error:
                logging.error(f"Erreur de fallback: {fallback_error}")
                raise e

    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=None) -> Dict[str, Any]:
        """
        Génère une réponse pour l'interface humaine
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.openai_wrapper.create(messages=messages)
            )

            response_text = response.choices[0].message.content

            return {
                "choices": [{
                    "message": {
                        "content": response_text,
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 100,  # Approximation
                    "completion_tokens": len(response_text) // 4,
                    "total_tokens": 100 + len(response_text) // 4
                }
            }
        except Exception as e:
            logging.error(f"Erreur dans generate_response_for_humain: {e}")
            return {
                "choices": [{
                    "message": {
                        "content": f"Erreur: {str(e)}",
                        "role": "assistant"
                    },
                    "finish_reason": "error"
                }]
            }

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Génère des embeddings (non supporté par ce modèle)
        """
        logging.warning("Les embeddings ne sont pas supportés par DeepSeek-R1")
        return []

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Exemple d'utilisation avec Instructor
if __name__ == "__main__":
    # Test du provider DeepSeek-R1 avec Instructor
    provider = LocalDeepSeek_R1_Provider(
        model="/home/yopla/montage_models/llm_models/python/models/txt2txt/DeepSeek-R1-Distill-Qwen-14B-Q5_K_S/DeepSeek-R1-Distill-Qwen-14B-Q5_K_S.gguf",
        system_prompt="Vous êtes un assistant IA utile et précis."
    )


    # Modèles Pydantic pour tester Instructor
    class SimpleResponse(BaseModel):
        answer: str
        confidence: float
        reasoning: str


    class ComplexAnalysis(BaseModel):
        summary: str
        key_points: List[str]
        sentiment: str
        confidence_score: float
        recommendations: List[str]


    async def test_instructor():
        print("=== Test avec Instructor (Version corrigée) ===")

        # Test 1: Réponse structurée simple
        print("\n1. Test réponse structurée simple:")
        try:
            response = await provider.generate_response(
                "Analysez la phrase: 'Le temps est magnifique aujourd'hui'",
                pydantic_model=SimpleResponse
            )
            print(f"Type: {type(response)}")
            if hasattr(response, 'model_dump_json'):
                print(f"Réponse: {response.model_dump_json(indent=2)}")
            else:
                print(f"Réponse: {response}")
        except Exception as e:
            print(f"Erreur test 1: {e}")

        # Test 2: Réponse structurée complexe
        print("\n2. Test réponse structurée complexe:")
        try:
            complex_response = await provider.generate_response(
                "Analysez ce texte: 'Les ventes ont augmenté de 15% ce trimestre grâce à notre nouvelle stratégie marketing. Cependant, les coûts de production ont également augmenté.'",
                pydantic_model=ComplexAnalysis
            )
            if hasattr(complex_response, 'model_dump_json'):
                print(f"Réponse complexe: {complex_response.model_dump_json(indent=2)}")
            else:
                print(f"Réponse complexe: {complex_response}")
        except Exception as e:
            print(f"Erreur test 2: {e}")

        # Test 3: Réponse simple (sans structure)
        print("\n3. Test réponse simple:")
        try:
            simple_response = await provider.generate_response(
                "Racontez-moi une courte histoire sur un chat."
            )
            print(f"Histoire: {simple_response}")
        except Exception as e:
            print(f"Erreur test 3: {e}")

        # Test 4: Conversation avec historique
        print("\n4. Test conversation avec historique:")
        try:
            provider.history.clear()  # Reset l'historique

            response1 = await provider.generate_response("Je m'appelle Alice.")
            print(f"Réponse 1: {response1}")

            response2 = await provider.generate_response(
                "Quel est mon nom et donnez-moi des informations sur ce prénom",
                pydantic_model=SimpleResponse
            )
            if hasattr(response2, 'model_dump_json'):
                print(f"Réponse 2 (structurée): {response2.model_dump_json(indent=2)}")
            else:
                print(f"Réponse 2 (structurée): {response2}")
        except Exception as e:
            print(f"Erreur test 4: {e}")

    # Exécuter le test
    # asyncio.run(test_instructor())


def _check_llama_cpp_available() -> bool:
    try:
        import llama_cpp
        return True
    except ImportError:
        return False


# ---------------------------------------------------------
#                   LLAVA


class LocalMultimodalProvider(LLMProvider):
    def __init__(self, model: str, clip_model_path: str, api_key: Optional[str] = None, cached_model=None,
                 system_prompt: Optional[str] = None,
                 structured_response: Optional[str] = None):

        super().__init__()

        clear_vram_if_possible()

        if not _check_llama_cpp_available():
            raise ImportError(
                """Ce provider nécessite llama_cpp. Installez-le avec: CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FORCE_CUBLAS=on -DLLAVA_BUILD=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
"""
            )

        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler

        self.system_prompt = system_prompt
        self.structured_response = structured_response
        self.executor = ThreadPoolExecutor(max_workers=1)

        if cached_model:
            self.llm = cached_model
        else:
            gpu_layers = get_optimal_n_gpu_layers(default=-1)  # -1 pour tout mettre sur le GPU
            print(f"Nombre de couches gpu utilisées : {str(gpu_layers)}")

            chat_handler = Llava15ChatHandler(clip_model_path=clip_model_path, verbose=False)

            # 2. Initialiser Llama en lui passant le chat_handler
            self.llm = Llama(
                model_path=model,
                chat_handler=chat_handler,
                n_gpu_layers=gpu_layers,
                n_ctx=4096,
                verbose=False
            )

    def get_model(self):
        return self.llm

    def _prepare_image(self, image_path: str) -> str:
        """Charge une image et l'encode en base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise IOError(f"Erreur lors du chargement de l'image {image_path}: {e}")

    async def generate_response(
            self,
            prompt: str,
            image_path: Optional[str] = None,  # Nouveau paramètre pour l'image
            pydantic_model: Optional[BaseModel] = None,
            stream: bool = False,
            callback=None,
            **kwargs
    ) -> Union[str, BaseModel]:
        """
        Génère une réponse, avec support optionnel de l'image et du streaming.
        """
        self.history.add_message("user", prompt)

        # Construction des messages pour le modèle multimodal
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Construction du contenu utilisateur (texte + image)
        user_content = [{"type": "text", "text": prompt}]
        if image_path:
            try:
                base64_image = self._prepare_image(image_path)
                image_url = f"data:image/jpeg;base64,{base64_image}"
                user_content.append({"type": "image_url", "image_url": {"url": image_url}})
            except IOError as e:
                # Gérer l'erreur si l'image ne peut pas être chargée
                # On peut choisir de continuer sans l'image ou de lever une exception
                logging.error(str(e))
                # Pour cet exemple, on continue sans l'image
                pass

        messages.append({"role": "user", "content": user_content})

        try:
            loop = asyncio.get_event_loop()
            response_generator = await loop.run_in_executor(
                self.executor,
                lambda: self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=kwargs.get("max_tokens", 2048),
                    temperature=kwargs.get("temperature", 0.7),
                    stream=stream
                )
            )

            if stream:
                full_response = ""

                async def stream_handler():
                    nonlocal full_response
                    for chunk in response_generator:
                        content = chunk["choices"][0].get("delta", {}).get("content")
                        if content:
                            full_response += content
                            if callback:
                                callback(content)
                            yield content
                    self.history.add_message("assistant", full_response.strip())

                # Retourne un générateur asynchrone pour le streaming
                return "".join([chunk async for chunk in stream_handler()])

            else:
                response_text = response_generator["choices"][0]["message"]["content"]
                self.history.add_message("assistant", response_text)
                return response_text

        except Exception as e:
            raise Exception(f"Erreur lors de la génération de la réponse multimodale: {str(e)}")

    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=None) -> Dict[str, Any]:
        """SPECTER ne génère pas de texte."""
        logging.warning("Not implemented")
        return {"error": "Not implemented"}

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


async def extract_value_from_data(image_path, legende):
    # Langue pour les prompts
    LANGUAGE = "français"

    print("Initialisation du fournisseur multimodal...")
    provider = LocalMultimodalProvider(
        model=LLAVA_MODEL_PATH,
        clip_model_path=CLIP_LLAVA_MODEL_PATH,
        system_prompt=f"Tu es un expert en analyse de données visuelles."
    )

    # La légende est une information cruciale qu'on fournit dans les deux étapes

    # --- ÉTAPE 1 : FORCER L'ANALYSE ET LA DESCRIPTION ---
    print("\n--- ÉTAPE 1 : Demande de description du graphique ---")
    prompt_description = (
        f"Voici la légende du graphique : {legende}\n\n"
        "Ta première tâche est de décrire ce graphique en détail, sans encore extraire de valeur numérique précise. "
        "Ta description doit inclure :\n"
        "1. Les axes X et Y, y compris leurs noms et l'intervalle approximatif de leurs valeurs.\n"
        "2. Les différentes courbes présentes (couleur et style, ex: 'ligne continue bleue').\n"
        "3. À quoi chaque courbe correspond d'après la légende fournie.\n"
        "4. La tendance générale de chaque courbe (ex: 'commence à X, augmente jusqu'à un maximum, puis diminue').\n\n"
        f"Fournis cette description entièrement et uniquement en {LANGUAGE}."
    )

    description_reponse = await provider.generate_response(
        prompt=prompt_description,
        image_path=image_path,
        stream=False  # Pas besoin de stream pour la description
    )

    print("\n[RÉPONSE DU MODÈLE - DESCRIPTION] :")
    print(description_reponse)
    # À ce stade, le prompt et la description sont dans l'historique du 'provider'.

    # --- ÉTAPE 2 : POSER LA QUESTION CIBLÉE ---
    print("\n--- ÉTAPE 2 : Demande d'extraction de la valeur spécifique ---")
    prompt_extraction = (
        "Parfait. Maintenant, en te basant sur ta description précédente et en analysant à nouveau le graphique très attentivement :\n"
        "Quelle est la valeur numérique approximative de εzz lorsque Rs est égal à 0.4 ? Pour trouver la bonne valeur, vous devez tracer une ligne verticale en partant de l'abscisse jusqu'à la courbe puis un droite horizontale de la courbe à l'ordonnée. la valeur sur l'axe correspond à la valeur à retourner\n"
        "**Réponds UNIQUEMENT avec la valeur numérique en pourcent.** Ne formule aucune phrase."
    )

    # Note : Pas besoin de renvoyer l'image_path !
    # Le chat handler de LLaVA a déjà traité l'image lors du premier appel.
    valeur_extraite = await provider.generate_response(
        prompt=prompt_extraction,
        stream=False
    )

    print("\n[RÉPONSE DU MODÈLE - VALEUR EXTRAITE] :")
    print(valeur_extraite)

    # Vous pouvez maintenant essayer de convertir la réponse en nombre
    try:
        valeur_numerique = float(valeur_extraite.strip().replace(',', '.'))
        print(f"\nValeur numérique parsée avec succès : {valeur_numerique}")
    except ValueError:
        print(f"\nImpossible de convertir la réponse '{valeur_extraite}' en nombre.")

    del provider


# Test en cours ...
if __name__ == "__main__":
    legende = """ εzz (blue line) and εxx (red line). Strain(Rs)"""
    asyncio.run(extract_value_from_data(
        "/home/yopla/Documents/llm_models/python/models/multimodal/test_figure/courbe.jpg",
        legende=legende))