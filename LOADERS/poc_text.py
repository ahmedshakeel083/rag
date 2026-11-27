from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation", 
    pipeline_kwargs={
        "max_length": 2048,
        "temperature": 0.7,
    },
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a summary report about {topic}.",
    input_variables = ["topic"]
)

chain = prompt | model | parser

loader = TextLoader("plastic_waste.txt")
documents = loader.load()
result =chain.invoke({"topic": documents[0].page_content })

print(result)