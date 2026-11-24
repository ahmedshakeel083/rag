from langchain_huggingface import HuggingFaceEmbeddings

documents = [
    "The quick brown fox jumps over the lazy dog.",
    "There is an empty room on my floor",
    "We will go to Falestine one day"
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

vectors = embeddings.embed_documents(documents)

print(str(vectors))
