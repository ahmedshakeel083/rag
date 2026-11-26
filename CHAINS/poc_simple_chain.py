from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
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

template = PromptTemplate(
    template="Give 5 facts about {topic}.",
    input_variables = ["topic"]
)

parser = StrOutputParser()

chain = template | model | parser

output = chain.invoke({ 'topic': 'Earth Orbit Debris' })

print(output)
