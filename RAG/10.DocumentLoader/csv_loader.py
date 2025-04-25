from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Student_Performance.csv')

docs= loader.load()

print(docs[1])