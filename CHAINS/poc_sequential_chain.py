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

prompt_one = PromptTemplate(
    template="Write a detailed report about {topic}.",
    input_variables = ["topic"]
)

prompt_two = PromptTemplate(
    template=(
        "Write a 5 pointer summary about below report."
        "Report: {report}"
    ),
    input_variables = ["report"]
)

parser = StrOutputParser()

chain = prompt_one | model | parser | prompt_two | model | parser

output = chain.invoke({ 'topic': 'Businesses independent of the technology' })

print(output)

chain.get_graph().print_ascii()