from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel ,RunnableSequence,RunnablePassthrough ,RunnableLambda

load_dotenv()

prompt1 =  PromptTemplate(
    template  = 'write a joke on the \n{topic}',
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template = 'explain the joke \n {joke} ',
    input_variables=['joke']
)

def word_counter(text):
    return len(text.split())

model= GoogleGenerativeAI(model = 'gemini-1.5-pro',temperature=0.7)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    # 'length':RunnableLambda(word_counter) <--- can be use like this
    'length':RunnableLambda(lambda x : len(x.split())),
    'explanation':RunnableSequence(prompt2,model,parser)
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)
result = final_chain.invoke({'topic':'football'})

print(result['joke'] , '\n' , result['length'] , '\n' , result['explanation'])