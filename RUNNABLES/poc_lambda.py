from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableLambda
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

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables = ["topic"]
)

def word_count(text):
    return len(text.split())

joke_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    "joke": joke_chain,
    "word_count": RunnableLambda(lambda text: word_count(text))
})

final_chain = RunnableSequence(joke_chain, parallel_chain)

result = final_chain.invoke({"topic": "wife"})

print(result)