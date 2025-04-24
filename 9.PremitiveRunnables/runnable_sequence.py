from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt1 =  PromptTemplate(
    template  = 'write a joke on the \n{topic}',
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template = 'explain the joke \n {joke} ',
    input_variables=['joke']
)

model= GoogleGenerativeAI(model = 'gemini-1.5-pro',temperature=0.7)

parser = StrOutputParser()

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)
result = chain.invoke({'topic':'AI'})

print(result)