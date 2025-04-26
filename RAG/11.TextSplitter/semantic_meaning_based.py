from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    GoogleGenerativeAIEmbeddings(),
    breakpoint_treshold_tyep = 'standard_deviation',
    breakpoint_treshold_amount =1
)

sample = '''Firstly, some samples of noise were generated from a uniform distribution and fed into the generator to produce fake data samples. . The generator was then disconnected, and the fake images were fed into the discriminator alongside real images sampled from the real data distribution. The weights and biases of the discriminator are then updated using gradient ascent by finding the global maximum.Secondly, the discriminator was disconnected, and the generator was trained by generating fake images and passing them into the discriminator. The generator’s parameters (weights and biases) were then updated using gradient descent. Both the discriminator and the generator were updated once per training cycle.I was not surprised to see this sort of thing coming from Google — with their deep-seated hatred for local desktop apps.Loading your projects from GitHub and then install dependencies instantly without any local downloading.It’s a serious game changer if your local-based IDE is hoarding all your resources in your normie PC — like VS Code does a lot.Tari for the last time, VS Code is a code editor and not an IDE!! Learn the difference for goodness sake!!!Ah yes, a code editor that eats up several gigabytes of RAM and gobbles up all your battery life that your OS itself starts complaining bitterly.'''

docs = text_splitter.create_document([sample])

print(len(docs))
print(docs)