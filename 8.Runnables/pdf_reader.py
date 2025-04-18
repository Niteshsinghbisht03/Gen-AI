from langchain.document_loaders import TextLoader ,PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_google_vertexai import VertexAIEmbeddings

loader = TextLoader('../example.txt')
documents = loader.load()


text_splitter  = RecursiveCharacterTextSplitter(chunk_size=50,chunk_overlap=5)
docs = text_splitter.split_documents(documents)


vectorstore = FAISS.from_documents(docs,VertexAIEmbeddings())
retriever = vectorstore.as_retriver()



llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, verbose=True)


qa_chain = RetrievalQA.from_chain_type(llm=llm,retriever=retriever)

query = 'what is acne?'
answer = qa_chain.run(query)

print('answer'.answer)