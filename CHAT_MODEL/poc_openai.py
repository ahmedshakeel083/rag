from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

model = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4",
    temperature=0,
    max_tokens=256
)

result = model.invoke("Which AI field will be the most financially successful in the next 5 years?")

print(result)