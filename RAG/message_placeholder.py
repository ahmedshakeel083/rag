from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

templates = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert, providing detailed and accurate information.'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', 'Explain in detail about the following topic: {topic}. Please provide a comprehensive explanation, including relevant examples and references.'),
])

chat_history_content= []

with open('chat_history.txt', 'r') as file:
    chat_history_content.extend(file.readlines())

prompt = templates.invoke({
    'domain': 'solar energy',
    'topic': 'What is the scope of solar energy and wind turbines in the next 10 years? and Which one is better for India and why?',
    'chat_history': chat_history_content
})

print('\n', prompt)