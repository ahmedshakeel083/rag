from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
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
    template="Create a tweet about {topic}",
    input_variables = ["topic"]
)

prompt_two = PromptTemplate(
    template="Create a linkedin post about {topic}",
    input_variables = ["topic"]
)

parser = StrOutputParser()

chain = RunnableParallel({
    'tweet': RunnableSequence(prompt_one, model, parser),
    'linkedin': RunnableSequence(prompt_two, model, parser)
})

result = chain.invoke({'topic': 'Quantum Computing'}) 

print(result)