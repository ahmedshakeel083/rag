from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
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

class User(BaseModel):
    name: str = Field(description="Name of the user")
    city: str = Field(description="City of the user")
    age: int = Field(description="Age of the user")

parser = PydanticOutputParser(pydantic_object=User)

data = PromptTemplate(
  template = "Give me the name, city and age of a fictional {nationality} person \n {format_instructions}",
  input_variables=["nationality"],
  partial_variables = { "format_instructions": parser.get_format_instructions() }  
)

# prompt = data.invoke({"nationality": "Indian"})

# result = model.invoke(prompt)

# output = parser.parse(result.content)


chain = data | model | parser

output = chain.invoke({"nationality": "Indian"})

print(output)