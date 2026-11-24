from langchain-openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(   
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=256)

result = llm.invoke("Which AI field will be the most financially successful in the next 5 years?")

print(result)