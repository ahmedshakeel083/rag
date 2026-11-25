from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from typing import Literal, TypedDict, Annotated, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation", 
    pipeline_kwargs={
        "max_length": 512,
        "temperature": 0.7,
    },
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    key_theme: Annotated[list[str], "Write all the key themes discussed in the review"]
    summary: Annotated[str, "few lines about the context"]
    sentiment: Annotated[Literal["pos", "neg"], "Review the full text and decide if it is positive or negative or neutral"]
    pros: Annotated[Optional[list[str]], "good things about the product"]
    cons: Annotated[Optional[list[str]], "bad things about the product"]

# class Review(BaseModel):
#     key_theme: list[str] = Field(description="Write all the key themes discussed in the review")
#     summary: str = Field(description="few lines about the context")
#     sentiment: Literal["pos", "neg"] = Field(description="Review the full text and decide if it is positive or negative or neutral")
#     pros: Optional[list[str]] = Field(default=None, description="good things about the product")
#     cons: Optional[list[str]] = Field(default=None, description="bad things about the product")
#     rating: int = Field(gt=0, lt=6, default=5, description="Rating should be between 1 to 5")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Solar Panels are very bad power generation prodcuts. They are expensive and take up a lot of space. I would not recommend them to anyone.""")

print(result)