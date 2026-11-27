from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation", 
    pipeline_kwargs={
        "max_length": 4096,
        "temperature": 0.7,
    },
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Figure out the product name from the {topic}",
    input_variables = ["topic"]
)

loader = WebBaseLoader("https://www.flipkart.com/nothing-phone-4b-black-128-gb/p/itm939608b40adbb")

docs = loader.load()

chain = prompt | model | parser

print (chain.invoke({'topic': docs}))