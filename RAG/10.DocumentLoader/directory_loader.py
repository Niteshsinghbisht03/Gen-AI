from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


loader = DirectoryLoader(
    path = 'D:',
    glob= '*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()

# print((docs[5].page_content))
# print((docs[5].metadata))

#lazyload

for document in docs:
    print(document.metadata)