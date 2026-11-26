from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

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

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="positive or negative")

feedback_output_parser = PydanticOutputParser(pydantic_object=Feedback)

parser = StrOutputParser()

feedback_prompt = PromptTemplate(
    template = "Classify the sentiment of the following feedback text into positive or negative \n {comment} \n  {format_instructions} ",
    input_variables = ["comment"],
    partial_variables = { "format_instructions": feedback_output_parser.get_format_instructions() }
)

feedback_chain = feedback_prompt | model | parser

sentiment_positive_prompt = PromptTemplate(
    template = "Write an appropriate response to this positive feedback \n {sentiment}",
    input_variables = ["sentiment"],
)

sentiment_negative_prompt = PromptTemplate(
    template = "Write an appropriate response to this negative feedback \n {sentiment}",
    input_variables = ["sentiment"],
)

# Assuming the output of feedback_chain is { "sentiment": "positive" || "negative" }

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", sentiment_positive_prompt | model | parser),
    (lambda x: x.sentiment == "negative", sentiment_negative_prompt | model | parser),
    RunnableLambda(lambda x: "Could not find setiment")
)

chain = feedback_chain | branch_chain

result = chain.invoke({ 'comment': 'This mobile phone is terrible' })

print(result)




