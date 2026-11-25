from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
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

initial_template = PromptTemplate(
    template="You are a researcher specialist. You will give detailed information about {topic}.",
    input_variables = ["topic"]
)

final_template = PromptTemplate(
    template="Summarize the information given below \n {text}.",
    input_variables = ["text"]
)

initial_prompt = initial_template.invoke({ "topic" : "Black Hole" })

final_input = model.invoke(initial_prompt)

final_prompt = final_template.invoke({ "text" : final_input.content })

result = model.invoke(final_prompt)

print(result.content)