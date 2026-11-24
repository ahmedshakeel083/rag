from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

faq_test_questions = [
    "It is enabled by default for all verified users. If missing, go to Settings → Rewards → Enable PayRewards and complete the one-time KYC check.",
    "Visa, Mastercard, RuPay, and American Express, issued by most major banks.",
    "2 points for every ₹100 paid, rounded down to the nearest ₹100.",
    "You can redeem your points for cashback, vouchers, or statement credits at participating stores.",
    "A feature that lets users earn reward points when they pay credit card bills through the app; points can be redeemed for cashback, vouchers, or statement credits.",
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

document_vectors = embeddings.embed_documents(faq_test_questions)

Query = "How can I earn reward points when paying credit card bills through the app?"
query_vectors = embeddings.embed_query(Query)

similarities = cosine_similarity([query_vectors], document_vectors)[0]

index, value = sorted(list(enumerate(similarities)), key=lambda x: x[1])[-1]

print(f"Query: {Query}")
print(f"Answer: {faq_test_questions[index]}")
print(f"Similarity Score: {value}")