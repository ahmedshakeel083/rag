from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch
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
    template="Write a detailed report about {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template="Write the summary about {text}",    
    input_variables = ["text"]
)

report_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough(),
)

final_chain = RunnableSequence(report_chain, branch_chain)

result = final_chain.invoke({"topic": "AI and its impact in present tech business"})

print(result) 