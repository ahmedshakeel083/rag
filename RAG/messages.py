from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_length": 512,
        "temperature": 0.7,
    },
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a solar panel specialist. You will answer questions about solar panels and their installation."),
    HumanMessage(content="What is the best type of solar panel for residential use?"),
]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)

