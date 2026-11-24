import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # openai_api_key=os.getenv("OPENAI_API_KEY"),
    dimensions=1536
)

embedding_vector = embedding.embed_query("How to master AI archictecture and design?")

print("Embedding vector:", str(embedding_vector))


