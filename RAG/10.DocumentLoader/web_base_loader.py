from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro')

url  = 'https://medium.com/munchy-bytes/exploring-langchain-ff13fff63340'

loader  = WebBaseLoader(url)

prompt = PromptTemplate(
    template='answer the folowing question \n {question} from the following text -\n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

docs=loader.load()

chain = prompt | model | parser

result = chain.invoke({"question":'When is the article is written?',"text":docs[0].page_content})
    
print(result)