from langchain_core.prompts import ChatPromptTemplate

templates = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert, providing detailed and accurate information.'),
    ('human', 'Explain in detail about the following topic: {topic}. Please provide a comprehensive explanation, including relevant examples and references.'),
])

prompt = templates.invoke({
    'domain': 'solar energy',
    'topic': 'What is the scope of solar energy and wind turbines in the next 10 years? and Which one is better for India and why?',
})

print(prompt)