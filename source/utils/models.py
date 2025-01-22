import os
import openai
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings
from utils.credentials import get_credentials
from functools import partial
from gpt4all import Embed4All
from utils.paths import GENERATOR_MODELS_DIR, EMBEDDER_MODELS_DIR
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def try_import_attr(module, name, default=None):
    try:
        return getattr(__import__(module, fromlist=[name]), name)
    except ImportError:
        return default


CLIENT_CERTIFICATE_PATH = try_import_attr("paths", "CLIENT_CERTIFICATE_PATH")
TIKTOKEN_CACHE_DIR = try_import_attr("paths", "TIKTOKEN_CACHE_DIR")
ALL_MINILM_L6_V2_DIR = try_import_attr("paths", "ALL_MINILM_L6_V2_DIR")

if TIKTOKEN_CACHE_DIR:
    os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_CACHE_DIR)


CHEAP_TEXT_GENERATOR = "gpt-4o-mini"
SMART_TEXT_GENERATOR = "gpt-4o"
# LOCAL_TEXT_GENERATOR = "phi-3-mini-4k-instruct"
LOCAL_TEXT_GENERATOR = "gpt2"
LOCAL_TEXT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TEXT_GENERATOR = CHEAP_TEXT_GENERATOR
DEFAULT_TEXT_EMBEDDER = LOCAL_TEXT_EMBEDDER
DEFAULT_TEXT_RERANKER = None


# TODO: treat different providers

MODELS = {
    "gpt-4": {
        "aliases": ["gpt4"],
        "type": "generator",
        "modalities": ["text"],
        "providers": ["openai", "azure"],
        "input_size": 128_000,
    },
    "gpt-4o": {
        "aliases": ["4o"],
        "type": "generator",
        "modalities": ["text"],
        "providers": ["openai", "azure"],
        "input_size": 128_000,
    },
    "gpt-4o-mini": {
        "aliases": ["4om"],
        "type": "generator",
        "modalities": ["text"],
        "providers": ["openai", "azure"],
        "input_size": 128_000,
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "aliases": ["minilm"],
        "type": "embedder",
        "modalities": ["text"],
        "input_size": 512,
        "embedding_sizes": [384],  # not used by now
        "providers": ["hugging_face"],
    },
    "text-embedding-ada-002": {
        "aliases": ["ada"],
        "type": "embedder",
        "modalities": ["text"],
        "input_size": 8_191,
        "embedding_size": 512,
        "embedding_sizes": [512],  # not used by now
        "providers": ["openai", "azure"],
    },
    "text-embedding-3-small": {
        "aliases": ["te3s"],
        "type": "embedder",
        "modalities": ["text"],
        "input_size": 8_191,
        "embedding_size": 512,
        "embedding_sizes": range(1, 512),  # not used by now
        "providers": ["openai", "azure"],
    },
    "text-embedding-3-large": {
        "aliases": ["te3l"],
        "type": "embedder",
        "modalities": ["text"],
        "input_size": 8_191,
        "embedding_size": 1_024,
        "embedding_sizes": range(1, 1_024),  # not used by now
        "providers": ["openai", "azure"],
    },
    # "gpt2": {
    #     "type": "text-generator",
    #     "input_size": 1_024,
    #     "hugging_face_task": "text-generation",
    #     "hugging_face_id": "gpt2" ,
    #     "provider": "huggingface",
    # },
    # @TODO rewrite GPT4AllEmbeddings class to be able to make use of this
    #   remember to normalize: https://platform.openai.com/docs/guides/embeddings/use-cases
    # "nomic-embed-text-v1.5": {
    #     "type": "text-embedder",
    #     "source": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/tree/main/1_Pooling",  # provider specific logic!
    #     "input_size": 2_048,
    #     "embedding_size": 768,
    #     "embedding_sizes": range(64, 1+768),
    #     "hugging_face_id": "nomic-ai/nomic-embed-text-v1.5",
    #     "provider": "huggingface",
    # },
}

for name, info in MODELS.items():
    info["name"] = name

MODEL_ALIASES = {}
for name, info in MODELS.items():
    aliases = info["aliases"]
    MODEL_ALIASES[name] = name
    for alias in aliases:
        MODEL_ALIASES[alias] = name

# TODO list_models()

def get_model_info(alias):
    name = MODEL_ALIASES.get(alias)
    return MODELS.get(name, {})


# TODO: get embedding model (hugging face)
# TODO: get embedding model (openai)
# TODO: get generator model (openai)
# TODO: get generator model (hugging face)
# TODO: local model which can embed images too


class HuggingFaceModels:
    DEFAULT_TEXT_GENERATOR = "gpt2"
    DEFAULT_TEXT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
    MODELS = {
        "sentence-transformers/all-MiniLM-L6-v2": {
            "id": "sentence-transformers/all-MiniLM-L6-v2",
            "path": ALL_MINILM_L6_V2_DIR,
        },
    }

    # def get_text_generator(self, model_name=None, temperature=0, **pipeline_kwargs):
    #     model_name = model_name or self.DEFAULT_TEXT_GENERATOR
    #     model_info = models.get(model_name)
        
    #     if not model_info or model_info["type"] != "text-generator":
    #         raise NotImplementedError(f"Model {model_name!r} is not a valid text generator.")
        
    #     model = HuggingFacePipeline.from_model_id(
    #         model_id=model_info["hugging_face_id"],
    #         task=model_info["hugging_face_task"],
    #         device=-1,
    #         pipeline_kwargs=pipeline_kwargs | {
    #             "clean_up_tokenization_spaces": True,
    #         },
    #     )
    #     return model

    def get_text_embedder(self, model_name=None, size=None, return_size=False, device="cpu"):
        model_name = model_name or self.DEFAULT_TEXT_EMBEDDER

        model_info = get_model_info(model_name)
        model_name = model_info.get("name", model_name)

        if model_info and model_info["type"] != "embedder":
            raise NotImplementedError(f"Model {model_name!r} is not a valid text embedder.")

        provider_model_info = self.MODELS.get(model_name, {})
        model_localizer = provider_model_info.get("path") or model_info.get("id")

        model = HuggingFaceEmbeddings(
            model_name=str(model_localizer),
            model_kwargs={
                "device": device,
                "trust_remote_code": False,
            },
            encode_kwargs={
                # do not norm outputs in order to get more complete data...
                # TODO Review this after adding matrioska models
                "normalize_embeddings": False,
                "clean_up_tokenization_spaces": True,
            },
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

        client_ca_path = client_ca_path or CLIENT_CERTIFICATE_PATH
        if client_ca_path and client_ca_path.exists():
            self.http_client = httpx.Client(verify=client_ca_path)
            self.http_async_client = httpx.AsyncClient(verify=client_ca_path)
        else:
            self.http_client = None
            self.http_async_client = None
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
            **({"http_async_client": self.http_async_client} if self.http_async_client else {}),
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    def get_text_generator(self, model_name=None, temperature=0, n=1, streaming=True, verbose=True):
        model_name = model_name or self.DEFAULT_TEXT_GENERATOR
        model_info = get_model_info(model_name)
        model_name = model_info.get("name", model_name)
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            n=n,
            streaming=streaming,
            verbose=verbose,
            openai_api_key=self.api_key,
            **({"base_url": self.base_url} if self.base_url else {}),
            **({"default_headers": self.headers} if self.headers else {}),
            **({"http_client": self.http_client} if self.http_client else {}),
            **({"http_async_client": self.http_async_client} if self.http_async_client else {}),
            max_retries=self.max_retries,
            timeout=self.timeout,
        )

    def get_text_embedder(self, model_name=None, size=None, return_size=False):
        model_name = model_name or self.DEFAULT_TEXT_GENERATOR
        model_info = get_model_info(model_name)
        model_name = model_info.get("name", model_name)
        embedder = OpenAIEmbeddings(
            model=model_name,
            dimensions=size,
            openai_api_key=self.api_key,
            **({"base_url": self.base_url} if self.base_url else {}),
            **({"default_headers": self.headers} if self.headers else {}),
            **({"http_client": self.http_client} if self.http_client else {}),
            **({"http_async_client": self.http_async_client} if self.http_async_client else {}),
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

class AzureOpenAIModels:
    DEFAULT_TEXT_COMPLETER = "gpt-4o"
    DEFAULT_TEXT_EMBEDDER = "text-embedding-ada-002"
    # TODO move azure aliases to credentials
    MODEL_ALIASES = {
    }

    def __init__(self, api_version=None, api_key=None, client_ca_path=None, max_retries=3, timeout=None):
        credentials = get_credentials("azure_openai")
        self.api_key = api_key or credentials["key"]
        self.base_url = credentials["base_url"]
        self.api_version = api_version or credentials.get("version", "2024-02-01")
        self.timeout = timeout
        self.timeout = timeout
        self.max_retries = max_retries

        client_ca_path = client_ca_path or CLIENT_CERTIFICATE_PATH
        if client_ca_path and client_ca_path.exists():
            self.http_client = httpx.Client(verify=client_ca_path)
            self.http_async_client = httpx.AsyncClient(verify=client_ca_path)
        else:
            self.http_client = None
            self.http_async_client = None
        self.headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
        }

    def get_provider_model_alias(self, name):
        return self.MODEL_ALIASES.get(name, name)

    def get_text_generator(self, model_name=None, temperature=0, n=1, streaming=True, verbose=True):
        model_name = model_name or self.DEFAULT_TEXT_GENERATOR
        model_info = get_model_info(model_name)
        model_name = model_info.get("name", model_name)
        model_alias = self.get_provider_model_alias(model_name)
        return AzureChatOpenAI(
            model_name=model_alias,
            temperature=temperature,
            n=n,
            streaming=streaming,
            verbose=verbose,
            max_retries=self.max_retries,
            timeout=self.timeout,
            **({"base_url": self.base_url} if self.base_url else {}),
            **({"default_headers": self.headers} if self.headers else {}),
            **({"http_client": self.http_client} if self.http_client else {}),
            **({"http_async_client": self.http_async_client} if self.http_async_client else {}),
            openai_api_key=self.api_key,
            openai_api_version=self.api_version,
        )

    def get_text_embedder(self, model_name=None, size=None, return_size=False):
        model_name = model_name or self.DEFAULT_TEXT_GENERATOR
        model_info = get_model_info(model_name)
        model_name = model_info.get("name", model_name)
        model_alias = self.get_provider_model_alias(model_name)
        embedder = AzureOpenAIEmbeddings(
            model=model_alias,
            # chunk_size=chunk_size,
            max_retries=self.max_retries,
            timeout=self.timeout,
            **({"base_url": self.base_url} if self.base_url else {}),
            **({"default_headers": self.headers} if self.headers else {}),
            **({"http_client": self.http_client} if self.http_client else {}),
            **({"http_async_client": self.http_async_client} if self.http_async_client else {}),
            openai_api_key=self.api_key,
            openai_api_version=self.api_version,
        )
        if return_size:
            size = MODELS[model_name].get("size")
            return embedder, size
        return embedder

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
    
    model_info = get_model_info(model_name)
    if not model_info:
        raise NotImplementedError(
            f"Model {model_name} not found in the available models list."
        )

    providers = model_info["providers"]
    provider = providers[0]

    if provider == "openai":
        api = OpenAIModels()
    elif provider == "azure":
        api = AzureOpenAIModels()
    elif provider == "hugging_face":
        api = HuggingFaceModels()
    else:
        raise ValueError(f"Model {model_name} is not a valid text generator model.")

    model = api.get_text_generator(model_name, temperature=temperature)
    return model


def get_text_embedder(model_name=None, return_size=False, size=None):
    model_name = model_name or DEFAULT_TEXT_EMBEDDER

    model_info = get_model_info(model_name)
    if model_info is None:
        raise NotImplementedError(
            f"Model {model_name} not found in the available models list."
        )
    
    providers = model_info["providers"]
    provider = providers[0]
    
    if provider == "openai":
        api = OpenAIModels()
    elif provider == "azure":
        api = AzureOpenAIModels()
    elif provider == "hugging_face":
        api = HuggingFaceModels()
    else:
        raise ValueError(f"Model {model_name} is not a valid text embedder model.")
    
    model = api.get_text_embedder(model_name=model_name, size=size, return_size=return_size)
    return model
