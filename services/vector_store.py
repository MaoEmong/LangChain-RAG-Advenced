# services/vector_store.py

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def create_vector_store(
    api_key,
    embed_model,
    persist_dir,
    collection_name,
):
    embeddings = OpenAIEmbeddings(
        model=embed_model,
        api_key=api_key,
    )

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
