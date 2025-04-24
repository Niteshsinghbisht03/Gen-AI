from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel ,RunnableSequence

load_dotenv()

prompt1= PromptTemplate(
    template='Genrate only a tweet about {topic}',
    input_variables=['topics']
)

prompt2 = PromptTemplate(
    template='Generate only a linkedin post about {topic}',
    input_variables = ['topic']
)

model = GoogleGenerativeAI(model = 'gemini-1.5-pro')

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1,model,parser),
    'linkedin': RunnableSequence(prompt2,model,parser)
})

result = parallel_chain.invoke({'topic':'AI'})

print(result['tweet'] ,'\n')
print(result['linkedin'])