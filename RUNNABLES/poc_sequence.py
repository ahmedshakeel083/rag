from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatHuggingFace.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256000,
        "temperature": 0.7,
    },
)

prompt_one = PromptTemplate(
    template="How to optimize recycling of {topic}?",
    input_variables = ['topic']
)

prompt_two = PromptTemplate(
    template="Summarize the information given below \n {text}.",
    input_variables = ["text"]
)

parser = StrOutputParser()

chain = RunnableSequence(prompt_one, model, parser, prompt_two, model, parser)

result = chain.invoke({"topic": "Plastic Waste"})

print(result)