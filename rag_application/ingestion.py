import os

from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# Run only if you want to insert vector data in Vector DB, else leave it.
if __name__ == "__main__":
    print("Ingesting...")
    loader = UnstructuredLoader(
        file_path=os.getenv("FILE_PATH"),
        chunking_strategy="basic", max_characters=1000000
    )
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    )
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks of text from the document.")

    print("Embedding...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.getenv("INDEX_NAME")
    )

    print("Finish!")



# class RagApplicationIngestion:
#
#     def __init__(self):
#         pass
#
#     def ingest_rag_application(self):
#         print("Ingesting...")
#         loader = UnstructuredLoader(
#             file_path=os.getenv("FILE_PATH"),
#             chunking_strategy="basic", max_characters=1000000
#         )
#         document = loader.load()
#
#         print("Splitting...")
#         text_splitter = CharacterTextSplitter(
#             chunk_size=1000, chunk_overlap=0
#         )
#         texts = text_splitter.split_documents(document)
#         print(f"Created {len(texts)} chunks of text from the document.")
#
#         print("Embedding...")
#         embeddings = OllamaEmbeddings(model="qwen3:8b")
#         vector_store = PineconeVectorStore.from_documents(
#             texts, embeddings, index_name=os.getenv("INDEX_NAME")
#         )
#
#         print("Finish!")
