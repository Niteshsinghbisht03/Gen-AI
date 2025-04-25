from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro')

prompt = PromptTemplate(
    template='what is poem using the given /n{text}',
    input_variables=['text']
)

parser = StrOutputParser()
loader = TextLoader("cricket_500_lines.txt", encoding='utf-8')

docs = loader.load()

# print(docs)
# print(type(docs))

# print(docs[0].metadata)
# print(docs[0].page_content)


chain = prompt | model | parser

print(chain.invoke({'text':docs[0].page_content}))