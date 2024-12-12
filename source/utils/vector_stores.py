from typing import Optional
from utils.paths import CHROMA_DB_DIR
from utils.models import get_text_embedder
from langchain_chroma import Chroma


def get_chroma_vector_store(
    persist_name: Optional[str] = None,
    embedder_name: Optional[str] = None,
    embedding_size: Optional[int] = None,
    collection: Optional[str] = None,
    create: bool = False,
) -> Chroma:
    embedder, embedding_size_gotten = get_text_embedder(embedder_name, return_size=True)
    embedding_size = embedding_size or embedding_size_gotten
    collection = collection or "default"
    temporary = persist_name is None
    persist_directory = None if temporary else CHROMA_DB_DIR / persist_name

    vector_store = Chroma(
        collection_name=collection,
        embedding_function=embedder,
        persist_directory=str(persist_directory),
        create_collection_if_not_exists=create or temporary,
    )

    return vector_store
