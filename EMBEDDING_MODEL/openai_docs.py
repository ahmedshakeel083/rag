from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()

documents = [
    "The quick brown fox jumps over the lazy dog.",
    "The rain in Spain stays mainly in the plain.",
    "There is a snake in the park."
]

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    dimensions=os.getenv("EMBEDDING_DIMENSIONS")
)

embedding_vectors = embeddings.embed_documents(documents)

print(str(embedding_vectors))