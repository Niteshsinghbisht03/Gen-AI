from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = GoogleGenerativeAI(model='gemini-1.5-pro')

model2 = ChatOpenAI(model='ChatGooglePalm')

# model2 = ChatAnthropic(model_name= 'claude-3-7-sonnet-20250219')

prompt1 = PromptTemplate(
    template= "Generate Short and Simple text from given Text \n{text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template= "Generate 5 short separate question answer from the following given Text \n{text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template= "merge the provided quiz and notes into a single document \n notes->{notes} and quiz->{quiz}",
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

#parallel chain
parallel_chain = RunnableParallel({
    'notes':prompt1 | model1 | parser ,
    'quiz': prompt2 | model1 | parser
})

#merging
merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
What is Linear Regression?
Linear Regression is a statistical supervised learning technique to predict the quantitative variable by forming a linear relationship with one or more independent features.
It helps determine:
→ If a independent variable does a good job in predicting the dependent variable.
→ Which independent variable plays a significant role in predicting the dependent variable.

Now,As you know most of the algorithm works with some kind of assumptions.So, Before moving on here is the list of assumptions of the Linear Regression.
These assumptions should be kept in mind when performing Linear Regression analysis so that the model performs it’s best.

Assumptions of Linear Regression:
The Independent variables should be linearly related to the dependent variables.
This can be examined with the help of several visualization techniques like: Scatter plot or maybe you can use Heatmap or pairplot(to visualize every features in the data in one particular plot).
Every feature in the data is Normally Distributed.
This again can be checked with the help of different visualization Techniques,such as Q-Q plot,histogram and much more.
There should be little or no multi-collinearity in the data.
The best way to check the prescence of multi-collinearity is to perform VIF(Variance Inflation Factor).
The mean of the residual is zero.
A residual is the difference between the observed y-value and the predicted y-value.However, Having residuals closer to zero means the model is doing great.
Residuals obatined should be normally distributed.
This can be verified using the Q-Q Plot on the residuals.
Variance of the residual throughout the data should be same.This is known as homoscedasticity.
This can be checked with the help of residual vs fitted plot.
There should be little or no Auto-Correlation is the data.
Auto-Correlation Occurs when the residuals are not independent of each other.This usally takes place in time series analysis.
You can perform Durbin-Watson test or plot ACF plot to check for the autocorrelation.If the value of Durbin-Watson test is 2 then that means no autocorrelation,If value < 2 then there is positive correlation and if the value is between >2 to 4 then there is negative autocorrelation.
If the features in the dataset are not normally distributed try out different transformation techniques to transform the distribution of the features present in the data.

Can you say,why these assumptions are needed?
The Gauss-Markov theorem states that if your linear regression model satisfies the first six classical assumptions, then ordinary least squares (OLS) regression produces unbiased estimates that have the smallest variance of all possible linear estimators.
"""


result = chain.invoke({"text":text})

print(result)

chain.get_graph().print_ascii()