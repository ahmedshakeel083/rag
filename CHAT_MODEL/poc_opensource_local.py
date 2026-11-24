
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# 1. Load a lightweight model locally (no API key required for ungated models)
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256000,
        "temperature": 0.7,
    },
)

# 2. Wrap in ChatHuggingFace to automatically apply the model's chat template
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the scope of solar energy and wind turbines in the next 10 years? and Which one is better for India and why?")

print(result.content)
