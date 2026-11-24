from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()
import streamlit as st 
from langchain_core.prompts import load_prompt


llm = HuggingFacePipeline.from_model_id(
    # model_id="Qwen/Qwen2.5-1.5B-Instruct",
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 1000,
        "temperature": 0.7,
    },
)

model = ChatHuggingFace(llm=llm)

st.header("Dynamic Prompting with Reasearch Tools")

# user_input = st.text_input("Enter your question here:")

paper_name = st.selectbox("Select a research paper:", ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "RoBERTa: A Robustly Optimized BERT Pretraining Approach", "T5: Text-to-Text Transfer Transformer", "GPT-3: Language Models are Few-Shot Learners"])

style = st.selectbox("Select Explanation style:", ["Beginner Friendly", "Technical", "Code Oriented", "Mathematical", "Business Oriented"])

depth = st.selectbox("Select Explanation depth:", ["High Level Overview", "Detailed Explanation", "In-depth Analysis"])

query = load_prompt("dynamic_prompt_template.json")

prompt = query.invoke({
    'paper_name': paper_name,
    'style': style,
    'depth': depth
})


if st.button("Summarize"):
    # result = model.invoke(user_input)
    result = model.invoke(prompt)
    print(result.content)
    st.write(result.content)
