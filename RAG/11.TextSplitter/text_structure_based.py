from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


text= """
Hello and welcome back to my blog😇. This post will be about a unique deep learning framework known as Generative Adversarial Networks (GANs), which I’ve found quite intriguing, not only because of how they work but also because of how they’re revolutionizing the way research problems in fluid dynamics, especially in reduced-order modeling and dynamical systems, are being approached.
Before GANs, most machine learning models were discriminative, meaning they were mainly used for classification or regression tasks. However I believe that GANs actually marked the beginning of the creative era in machine and deep learning. Yann LeCun, the Chief Scientist at Meta AI, once described GANs as “the most interesting idea in the last 10 years in Machine Learning” — and I really couldn’t agree more.


Yann LeCun on GANs
In as much as this framework has only been around since 2014, GANs have already made a solid mark in the fluid dynamics community. They’ve been used to create realistic flow field snapshots by learning the data distribution of actual fluid flows — which is a big deal. This means we can now produce fluid flow data without running full CFD simulations, which are often expensive. This is especially useful when you want to build an ML model but only have a limited amount of data to work with. GANs have also been used to generate new, plausible boundary condition profiles for simulations based on prior data — and that’s just one of many cool applications.

Now, if you’re not into fluid dynamics like I am 😎, GANs are still everywhere. They’ve been used to create hyper-realistic images of people that don’t actually exist haha. You can check out this website. Every time you refresh, a totally fake (but very realistic) human face pops up. Beyond that, GANs have been applied in generative design that enable you create cool 3d furniture for your home. Companies like Adobe are using GANs to build next-generation Photoshop tools, Google uses them for text generation, IBM uses them for data augmentation, and platforms like Snapchat and Tiktok have been using GANs to create image filters for quite a while now.
"""



spiltter  = RecursiveCharacterTextSplitter(
    chunk_size= 100,
    chunk_overlap = 0,
)

chunks = spiltter.split_text(text)

print(len(chunks))

print(chunks[0])