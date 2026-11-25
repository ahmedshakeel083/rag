from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
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

schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

data = PromptTemplate(
    template = "Give me 3 facts about {topic} {format_instructions}",
    input_variables=['topic'],
    partial_variables = { "format_instructions": parser.get_format_instructions() }
)

# prompt = data.invoke({'topic': 'Higgs Boson'})

# result = model.invoke(prompt)

# output = parser.parse(result)

chain = data | model | parser

output = chain.invoke({ 'topic': 'Higgs Boson' })

print(output)

# chain = data | model | parser

