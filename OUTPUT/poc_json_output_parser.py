from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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

parser = JsonOutputParser()

data = PromptTemplate(
    template = "Give me the name, age and city of a fictional person {format_instructions}",
    input_variables=[],
    partial_variables = { "format_instructions": parser.get_format_instructions() }
)

# prompt = data.invoke({})

# result = model.invoke(prompt)

chain = data | model | parser

result = chain.invoke({})

print(result)