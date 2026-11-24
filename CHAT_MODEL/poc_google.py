from langchain_google_genal import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

model = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-1.5-pro",
    temperature=0
)

result = model.invoke("What will be the future of STEM education in the next 5 years?")

print(result.content)