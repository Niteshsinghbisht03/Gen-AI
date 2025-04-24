from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableBranch ,RunnableSequence ,RunnablePassthrough

load_dotenv()

prompt1 =  PromptTemplate(
    template  = 'write a report on the \n {topic}',
    input_variables=['topic']
)

prompt2  = PromptTemplate(
    template = 'write the summary for the  given text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()
model = GoogleGenerativeAI(model = 'gemini-1.5-pro',temperature=0.7)

report_gen_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split())>50 , RunnableSequence(prompt2,model,parser)),
     RunnablePassthrough()
) 

final_chain = RunnableSequence(report_gen_chain,branch_chain)

result = final_chain.invoke({'topic':'Russia vs Ukrain'})

print(result)