from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser , ResponseSchema
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name = 'fact1',description='fact 1 about topic'),
    ResponseSchema(name = 'fact2',description='fact 2 about topic'),
    ResponseSchema(name = 'fact3',description='fact 3 about topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template  = PromptTemplate(
    template='give 3 fact about {topic} \n{format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain  = template | model | parser 
final_result =chain.invoke({'topic':'black hole'})


print(final_result)