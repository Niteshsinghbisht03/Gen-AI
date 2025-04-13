from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import openai

load_dotenv()

model = ChatOpenAI(model = 'gpt-4',temperature=1.8,max_completion_tokens=50)
result=model.invoke('write a joke.')

print(result)



