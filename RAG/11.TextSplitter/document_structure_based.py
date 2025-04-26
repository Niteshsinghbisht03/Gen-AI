from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

code = '''
# Class definition and object instantiation
class Dog:
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed
    
    def bark(self):
        print("Woof!")

# Creating an instance of the Dog class
my_dog = Dog("Buddy", "Golden Retriever")

# Accessing attributes and calling methods
print(f"{my_dog.name} is a {my_dog.breed}")
my_dog.bark()
'''

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0,
)

chunks = splitter.split_text(code)

print(len(chunks))
print(chunks[0])