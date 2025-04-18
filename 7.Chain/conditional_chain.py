from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch ,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel ,Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model='gemini-1.5-pro')

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['Positive','Negative'] = Field(description='Provide the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = 'classify the sentiment of the following feedback text into positive or negative \n{feedback} \n{format_instruction}',
    input_variables= ['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='write a appropiate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='write a appropiate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'Positive',prompt2 | model | parser),#condition 1
    (lambda x : x.sentiment == 'Negative',prompt3 | model | parser),#condition 2
    RunnableLambda(lambda x:"could not find sentiment")#default condition
    )

chain = classifier_chain | branch_chain 

result = chain.invoke({'feedback':'This smartphone is wonderful'})
    
print(result)

chain.get_graph().print_ascii()