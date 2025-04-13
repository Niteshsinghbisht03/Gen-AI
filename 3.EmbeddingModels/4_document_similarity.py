from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)
embedding = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2')
documents = [
    "Python is a versatile and beginner-friendly programming language that is widely used in web development, data analysis, artificial intelligence, and scientific computing.",
    "With the rapid advancement of machine learning technologies, industries across the globe are leveraging data-driven models to enhance decision-making and automate complex tasks.",
    "Chatbots, powered by natural language processing and AI, are increasingly being used by businesses to provide 24/7 customer support and streamline user interactions.",
    "Data science is a multidisciplinary field that combines statistical analysis, computer science, and domain knowledge to extract meaningful insights from vast amounts of structured and unstructured data.",
    "Artificial intelligence is revolutionizing healthcare by enabling early disease detection, personalized treatment recommendations, and the automation of administrative processes."
]

query = 'tell me about AI.'

doc_embeddings = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embeddings)[0]

index, score =sorted(list(enumerate(scores)),key= lambda x:x[1])[-1]

print(documents[index])
print(score)