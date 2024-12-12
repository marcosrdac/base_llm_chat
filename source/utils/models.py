import openai
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils.credentials import get_credentials
from functools import partial
from gpt4all import Embed4All
from utils.paths import GENERATOR_MODELS_DIR, EMBEDDER_MODELS_DIR
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


CHEAP_TEXT_GENERATOR = "gpt-4o-mini"
SMART_TEXT_GENERATOR = "gpt-4o"
# LOCAL_TEXT_GENERATOR = "phi-3-mini-4k-instruct"
LOCAL_TEXT_GENERATOR = "gpt2"
LOCAL_TEXT_EMBEDDER = "all-minilm-l6-v2"

DEFAULT_TEXT_GENERATOR = CHEAP_TEXT_GENERATOR
DEFAULT_TEXT_EMBEDDER = LOCAL_TEXT_EMBEDDER


models = {
    "gpt-4o-mini": {
        "type": "text-generator",
        "input_size": 128_000,
        "provider": "openai",
    },
    "gpt-4o": {
        "type": "text-generator",
        "input_size": 128_000,
        "provider": "openai",
    },
    "text-embedding-3-small": {
        "type": "text-embedder",
        "input_size": 8_191,
        "embedding_size": 512,
        "embedding_sizes": range(1, 512),
        "provider": "openai",
    },
    "text-embedding-3-large": {
        "type": "text-embedder",
        "input_size": 8_191,
        "embedding_size": 1_024,
        "embedding_sizes": range(1, 1_024),
        "provider": "openai",
    },
    "gpt2": {
        "type": "text-generator",
        "input_size": 1_024,
        "hugging_face_task": "text-generation",
        "hugging_face_id": "gpt2" ,
        "provider": "huggingface",
    },
    # @TODO rewrite GPT4AllEmbeddings class to be able to make use of this
    #   remember to normalize: https://platform.openai.com/docs/guides/embeddings/use-cases
    "nomic-embed-text-v1.5": {
        "type": "text-embedder",
        "input_size": 2_048,
        "embedding_size": 768,
        "embedding_sizes": range(64, 1+768),
        "hugging_face_id": "nomic-ai/nomic-embed-text-v1.5",
        "provider": "huggingface",
    },
    "all-minilm-l6-v2": {
        "type": "text-embedder",
        "input_size": 512,
        "embedding_size": 384,
        # "hugging_face_id": "nreimers/MiniLM-L6-H384-uncased",
        "hugging_face_id": "sentence-transformers/all-MiniLM-L6-v2",
        "provider": "huggingface",
    },
}


class HuggingFaceModels:
    DEFAULT_TEXT_GENERATOR = "gpt2"
    DEFAULT_TEXT_EMBEDDER = "all-minilm-l6-v2"

    def get_text_generator(self, model_name=None, temperature=0, **pipeline_kwargs):
        model_name = model_name or self.DEFAULT_TEXT_GENERATOR
        model_info = models.get(model_name)
        
        if not model_info or model_info["type"] != "text-generator":
            raise NotImplementedError(f"Model {model_name!r} is not a valid text generator.")
        
        model = HuggingFacePipeline.from_model_id(
            model_id=model_info["hugging_face_id"],
            task=model_info["hugging_face_task"],
            device=-1,
            pipeline_kwargs=pipeline_kwargs | {
                "clean_up_tokenization_spaces": True,
            },
        )
        return model

    def get_text_embedder(self, model_name=None, size=None, return_size=False):
        model_name = model_name or self.DEFAULT_TEXT_EMBEDDER
        model_info = models.get(model_name)
        
        if not model_info or model_info["type"] != "text-embedder":
            raise NotImplementedError(f"Model {model_name!r} is not a valid text embedder.")

        # @TODO implement size for matrioska models
        model = HuggingFaceEmbeddings(
            model_name=model_info["hugging_face_id"],
            model_kwargs={"device": "cpu"},
        )
        if return_size:
            size = size or model_info.get("")
            return model, size

        return model


class OpenAIModels:
    DEFAULT_TEXT_GENERATOR = "gpt-4o-mini"
    DEFAULT_TEXT_EMBEDDER = "text-embedding-3-small"
    DEFAULT_IMAGE_GENERATOR = "dall-e-3"

    def __init__(self, api_key=None, base_url=None, client_ca_path=None, max_retries=3, timeout=None):
        credentials = get_credentials("openai")
        self.api_key = api_key or credentials["key"]
        self.base_url = base_url or credentials.get("base_url")
        self.timeout = timeout
        self.max_retries = max_retries
        if client_ca_path:
            self.http_client = httpx.Client(verify=client_ca_path)
        else:
            self.http_client = None
        self.headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
        }

    def get_client(self):
        return openai.OpenAI(
            api_key=self.api_key,
            **({"base_url": self.base_url} if self.base_url else {}),
            **({"default_headers": self.headers} if self.headers else {}),
            **({"http_client": self.http_client} if self.http_client else {}),
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    def get_text_generator(self, model_name=None, temperature=0, n=1, streaming=True, verbose=True):
        return ChatOpenAI(
            model=model_name or self.DEFAULT_TEXT_GENERATOR,
            temperature=temperature,
            n=n,
            streaming=streaming,
            verbose=verbose,
            openai_api_key=self.api_key,
            **({"base_url": self.base_url} if self.base_url else {}),
            **({"default_headers": self.headers} if self.headers else {}),
            **({"http_client": self.http_client} if self.http_client else {}),
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    def get_text_embedder(self, model_name=None, size=None, return_size=False):
        embedder = OpenAIEmbeddings(
            model=model_name or self.DEFAULT_TEXT_EMBEDDER,
            dimensions=size,
            openai_api_key=self.api_key,
            **({"base_url": self.base_url} if self.base_url else {}),
            **({"default_headers": self.headers} if self.headers else {}),
            **({"http_client": self.http_client} if self.http_client else {}),
            max_retries=self.max_retries,
            timeout=self.timeout,
        )
        if return_size:
            size = MODELS[model].get("embedding_size")
            return embedder, size
        return embedder

    def generate_image(self, prompt, model_name=None, size=None, quality=None, n=1):
        """
        Generates images based on the given prompt.

        Args:
            prompt (str): Text prompt to generate images.
            model_name (str, optional): Image model to use. Options:
                - "dall-e-2" (default size: 512x512)
                - "dall-e-3" (default size: 1024x1024)
            size (str, optional): Image size. 
                - For "dall-e-2": Must be one of "256x256", "512x512", or "1024x1024" (default: "512x512").
                - For "dall-e-3": Must be one of "1024x1024", "1792x1024", or "1024x1792" (default: "1024x1024").
            quality (str, optional): Image quality, can be "standard" or "hd" (default: "standard").
            n (int, optional): Number of images to generate (default: 1).

        Returns:
            Response object with generated image(s).
        """

        model_name = model_name or self.DEFAULT_IMAGE_GENERATOR
        quality = quality or "standard"

        client = self.get_client()

        if model_name == "dall-e-2":
            size = size or "512x512"
        if model_name == "dall-e-3":
            size = size or "1024x1024"

        response = client.images.generate(
            prompt=prompt,
            model=model_name,
            n=n,
            size=size,
            quality=quality,
        )
        return response


def get_text_generator(model_name=None, kind=None, temperature=0):
    if not model_name:
        if not kind:
            model_name = DEFAULT_TEXT_GENERATOR
        elif kind in ["fast", "cheap"]:
            model_name = CHEAP_TEXT_GENERATOR
        elif kind in ["smart", "expensive"]:
            model_name = SMART_TEXT_GENERATOR
        elif kind in ["local"]:
            model_name = LOCAL_TEXT_GENERATOR
        else:
            raise ValueError(f"kind not supported: {kind!r}")
    
    model_info = models.get(model_name)
    
    if model_info is None:
        raise NotImplementedError(
            f"Model {model_name} not found in the available models list."
        )
    
    if model_info["provider"] == "openai" and model_info["type"] == "text-generator":
        openai_models = OpenAIModels()
        return openai_models.get_text_generator(model_name=model_name, temperature=temperature)
    elif model_info["provider"] == "gpt4all" and model_info["type"] == "text-generator":
        gpt4all_models = GPT4AllModels()
        return gpt4all_models.get_text_generator(model_name=model_name, temperature=temperature)
    elif model_info["provider"] == "huggingface" and model_info["type"] == "text-generator":
        hf_models = HuggingFaceModels()
        return hf_models.get_text_generator(model_name=model_name, temperature=temperature)

    raise ValueError(f"Model {model_name} is not a valid text generator model.")


def get_text_embedder(model_name=None, return_size=False, size=None):
    model_name = model_name or DEFAULT_TEXT_EMBEDDER
    model_info = models.get(model_name)
    
    if model_info is None:
        raise NotImplementedError(
            f"Model {model_name} not found in the available models list."
        )
    
    if model_info["provider"] == "openai" and model_info["type"] == "text-embedder":
        openai_models = OpenAIModels()
        return openai_models.get_text_embedder(model_name=model_name, size=size, return_size=return_size)
    
    elif model_info["provider"] == "gpt4all" and model_info["type"] == "text-embedder":
        gpt4all_models = GPT4AllModels()
        return gpt4all_models.get_text_embedder(model_name=model_name, size=size, return_size=return_size)

    elif model_info["provider"] == "huggingface" and model_info["type"] == "text-embedder":
        hf_models = HuggingFaceModels()
        return hf_models.get_text_embedder(model_name=model_name, size=size, return_size=return_size)

    raise ValueError(f"Model {model_name} is not a valid text embedder model.")
