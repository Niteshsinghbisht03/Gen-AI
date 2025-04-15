from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from langchain_google_genai import GoogleGenerativeAI
load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task='text-generation'
# )

# model = ChatHuggingFace(llm=llm)

model = GoogleGenerativeAI(model='gemini-1.5-pro')

class Person(BaseModel):
    
    name:str = Field(description="name of the person")
    age :int = Field(gt=18,description="age of the person")
    city:str = Field(description="name of the city which the person belongs to")
    
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "generate the name ,place,city of a fictional {place} person \n{format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place':'arabic'})

print(result)