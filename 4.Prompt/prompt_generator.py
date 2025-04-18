from langchain.prompts import PromptTemplate

template = PromptTemplate(
    template='''
    Please Summarize the research Paper titled "{paper_input}"with the following Specification:
    Explanation Style :{style_input}
    Explanation Length : {length_input}
    1. Mathematical Details:
        -include relevant mathematical equation if present in the paper.
        -Explain the mathematical concept using simple,intutive code snippets where applicable.
    2.Analogies:
        -Use relatable analogies to simplify complex ideas.
    if certain information is not available in the paper,respond with :'Insufficient information availbale' insted of guessing.
    Ensure the summary is clear , accurate and aligned with the provided style and length
    ''',
    input_variables=['paper_input','style_input','length_input'],
    validate_template=True
)

template.save('template.json')