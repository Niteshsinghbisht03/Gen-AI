from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

st.header('Research Tool')

paper_input = st.selectbox('Select Research Paper NAme',['Select...','Attention Is All You Need','BERT: Pre-training of Deep Bidirectional Trasnformers','GPT-3 : Language Models are few-shot Learner','Diffusion Models Beat GAN on Image Synthesis'])

style_input = st.selectbox('Select Explanation Style',['Beginner-Friendly','Technical','Code-Oriented','Mathematial'])

length_input = st.selectbox('Select Explanation Length',['Short(1-2 paragraphs)','Medium(3-5 paragraphs)','Long(detailed Explanation)'])


template = load_prompt('Prompt/template.json')


if st.button('summarizer'):
    chain = template | model
    result = chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input,
})
    
    st.text(result.content)