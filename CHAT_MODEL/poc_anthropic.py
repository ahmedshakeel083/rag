from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

model = ChatAnthropic(
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-3-5-sonnet-20241022",
    temperature=0,
    max_tokens=256
)

result = model.invoke("What will be the future of STEM education in the next 5 years?")

print(result.content)