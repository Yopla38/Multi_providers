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

from ai_services.config import settings

@dataclass
class Message:
    """Classe représentant un message dans l'historique"""
    role: str
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
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def clear(self):
        self.messages.clear()


def clear_vram_if_possible():
    """Tente de libérer la VRAM de PyTorch."""
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            print("🧹 Nettoyage de la VRAM (cache PyTorch)...")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("✅ VRAM nettoyée.")
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ Avertissement : Échec du nettoyage de la VRAM : {e}")


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
        pass

    def generate_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_embeddings(texts))
        finally:
            loop.close()

    def generate_response_sync(self, messages, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_response(messages, **kwargs))
        finally:
            loop.close()

    def set_system_prompt(self, system_prompt):
        pass


def get_optimal_n_gpu_layers(default: int = 0) -> int:
    """Détermine dynamiquement le nombre optimal de couches GPU."""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader']
        )
        free_memory = int(output.decode('utf-8').strip().split('\n')[0])
        if free_memory >= 16000: return 35
        if free_memory >= 8000: return 20
        if free_memory >= 4000: return 12
        return 6
    except Exception as e:
        print("Aucun GPU détecté ou erreur lors de la détection de la mémoire GPU :", e)
        return default


class LlamaCppOpenAIWrapper:
    """Wrapper pour rendre llama-cpp-python compatible avec l'API OpenAI."""
    def __init__(self, llama_model: Llama):
        self.llama_model = llama_model
        self.chat = self
        self.completions = self

    def create(self, messages: List[Dict[str, str]], **kwargs):
        formatted_prompt = self._format_deepseek_r1_prompt(messages)
        generation_params = {
            "prompt": formatted_prompt,
            "max_tokens": kwargs.get("max_tokens", 4000),
            "temperature": kwargs.get("temperature", 0.6),
            "top_p": kwargs.get("top_p", 0.95),
            "stop": ["<|begin_of_sentence|>", "<|end_of_sentence|>", "<|User|>", "<|Assistant", "<｜User｜>", "<｜end of sentence｜>"],
            "echo": False
        }
        response = self.llama_model(**generation_params)
        response_text = response["choices"][0]["text"].strip()

        class MockMessage:
            def __init__(self, content):
                self.content = content
                self.role = "assistant"
        class MockChoice:
            def __init__(self, content):
                self.message = MockMessage(content)
        class MockCompletion:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
        return MockCompletion(response_text)

    def _format_deepseek_r1_prompt(self, messages: List[Dict[str, str]]) -> str:
        formatted = ""
        for i, message in enumerate(messages):
            if message["role"] == "user":
                formatted += f"<｜User｜>{message['content']}\n\n"
            elif message["role"] == "assistant":
                formatted += f"<｜Assistant｜>{message['content']}"
                if not (i == len(messages) - 1):
                    formatted += "<｜end of sentence｜>\n\n"
            elif message["role"] == "system":
                formatted = message['content'] + "\n\n" + formatted
        if messages and messages[-1]["role"] != "assistant":
            formatted += "<｜Assistant｜>"
        return formatted


class LocalDeepSeek_R1_Provider(LLMProvider):
    def __init__(self, system_prompt: Optional[str] = None):
        super().__init__()
        clear_vram_if_possible()

        self.config = settings.providers.local_deepseek
        self.system_prompt = system_prompt
        self.executor = ThreadPoolExecutor(max_workers=1)

        gpu_layers = get_optimal_n_gpu_layers(default=0)
        print(f"Nombre de couches gpu utilisées : {str(gpu_layers)}")

        self.llm = Llama(
            model_path=str(self.config.model_path),
            n_gpu_layers=gpu_layers,
            n_ctx=self.config.n_ctx,
            verbose=False,
            chat_format=None,
        )
        self.openai_wrapper = LlamaCppOpenAIWrapper(self.llm)
        self.instructor_client = instructor.patch(self.openai_wrapper, mode=instructor.Mode.MD_JSON)

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3), reraise=True)
    async def generate_response(
            self, prompt: Union[str, List[Dict[str, str]]], pydantic_model: Optional[BaseModel] = None, **kwargs
    ) -> Union[str, BaseModel]:
        if isinstance(prompt, str):
            self.history.add_message("user", prompt)
            messages = self.history.get_formated_messages()
        else:
            messages = prompt
            for msg in prompt:
                if msg["role"] == "user": self.history.add_message("user", msg["content"])

        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
             messages.insert(0, {"role": "system", "content": self.system_prompt})

        loop = asyncio.get_event_loop()
        try:
            task = lambda: self.instructor_client.chat.completions.create(
                messages=messages,
                response_model=pydantic_model,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature)
            )
            response = await loop.run_in_executor(self.executor, task)

            if pydantic_model:
                self.history.add_message("assistant", response.model_dump_json(indent=2))
                return response
            else:
                response_text = response.choices[0].message.content
                self.history.add_message("assistant", response_text)
                return response_text
        except Exception as e:
            logging.error(f"Erreur lors de la génération avec DeepSeek-R1 : {e}")
            raise e

    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=None) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(self.executor, lambda: self.openai_wrapper.create(messages=messages))
        response_text = response.choices[0].message.content
        return {"choices": [{"message": {"content": response_text, "role": "assistant"}}]}

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        logging.warning("Les embeddings ne sont pas supportés par DeepSeek-R1")
        return []

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def __del__(self):
        if hasattr(self, 'executor'): self.executor.shutdown(wait=False)


def _check_llama_cpp_available() -> bool:
    try:
        import llama_cpp; return True
    except ImportError: return False


class LocalMultimodalProvider(LLMProvider):
    def __init__(self, cached_model=None, system_prompt: Optional[str] = None):
        super().__init__()
        clear_vram_if_possible()
        if not _check_llama_cpp_available():
            raise ImportError("Ce provider nécessite llama_cpp.")

        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler

        self.config = settings.providers.local_llava
        self.system_prompt = system_prompt
        self.executor = ThreadPoolExecutor(max_workers=1)

        if cached_model:
            self.llm = cached_model
        else:
            gpu_layers = get_optimal_n_gpu_layers(default=-1)
            print(f"Nombre de couches gpu utilisées : {str(gpu_layers)}")
            chat_handler = Llava15ChatHandler(clip_model_path=str(self.config.clip_path), verbose=False)
            self.llm = Llama(
                model_path=str(self.config.model_path),
                chat_handler=chat_handler,
                n_gpu_layers=gpu_layers,
                n_ctx=self.config.n_ctx,
                verbose=False
            )

    def get_model(self):
        return self.llm

    def _prepare_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def generate_response(
            self, prompt: str, image_path: Optional[str] = None, pydantic_model: Optional[BaseModel] = None, **kwargs
    ) -> Union[str, BaseModel]:
        self.history.add_message("user", prompt)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        user_content = [{"type": "text", "text": prompt}]
        if image_path:
            base64_image = self._prepare_image(image_path)
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        messages.append({"role": "user", "content": user_content})

        loop = asyncio.get_event_loop()
        response_generator = await loop.run_in_executor(
            self.executor,
            lambda: self.llm.create_chat_completion(
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature)
            )
        )
        response_text = response_generator["choices"][0]["message"]["content"]
        self.history.add_message("assistant", response_text)
        return response_text

    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=None) -> Dict[str, Any]:
        logging.warning("Not implemented")
        return {"error": "Not implemented"}

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass

    def __del__(self):
        if hasattr(self, 'executor'): self.executor.shutdown(wait=False)


if __name__ == "__main__":
    async def test_deepseek_provider():
        print("=== Test du provider DeepSeek-R1 ===")
        provider = LocalDeepSeek_R1_Provider(system_prompt="Vous êtes un assistant IA utile.")
        class SimpleResponse(BaseModel):
            answer: str
        response = await provider.generate_response(
            "Quelle est la capitale de la France ?", pydantic_model=SimpleResponse
        )
        print(f"Réponse structurée: {response.model_dump_json(indent=2)}")

        text_response = await provider.generate_response("Raconte une blague.")
        print(f"Réponse texte: {text_response}")

    async def test_llava_provider():
        print("\n=== Test du provider LLaVA ===")
        provider = LocalMultimodalProvider(system_prompt="Tu es un expert en analyse d'images.")
        # Créez un fichier image factice pour le test
        try:
            from PIL import Image
            test_image_path = "test_image.png"
            Image.new('RGB', (100, 100), color = 'red').save(test_image_path)
            response = await provider.generate_response("Que vois-tu sur cette image ?", image_path=test_image_path)
            print(f"Analyse d'image: {response}")
            os.remove(test_image_path)
        except Exception as e:
            print(f"Erreur pendant le test LLaVA (PIL est-il installé?): {e}")

    # Décommentez pour exécuter les tests
    # asyncio.run(test_deepseek_provider())
    # asyncio.run(test_llava_provider())