import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "furniture_delivery_assistant"


def main():
    document_paths = sorted(DATA_DIR.glob("*.txt"))
    if not document_paths:
        raise FileNotFoundError(f"No .txt knowledge documents found in {DATA_DIR}")

    documents = []
    for path in document_paths:
        loaded = TextLoader(str(path), encoding="utf-8").load()
        for document in loaded:
            document.metadata["source"] = path.name
        documents.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    embedding_model = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # Rebuild the local collection so repeated ingestion doesn't duplicate chunks.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    print(f"Ingestion complete: {len(chunks)} chunks stored from {len(document_paths)} documents.")


if __name__ == "__main__":
    main()
