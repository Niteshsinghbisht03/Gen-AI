from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2')

# text = 'delhi is the capital of india'
documents = ['delhi is the capital of india','all indian are my brother and sister','hello how are youth']
# vector = embedding.embed_query(text)
doc_vector = embedding.embed_documents(documents)

print(str(doc_vector))