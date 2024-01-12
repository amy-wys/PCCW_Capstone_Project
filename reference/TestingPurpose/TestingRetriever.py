# IMPORTING LIBRARIES
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
import os
from dotenv import load_dotenv

# GETTING API KEY FROM .env
_ = load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# SETTING UP EMBEDDINGS AND LANGUAGE MODELS
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-base-en-v1.5",
                                                      model_kwargs={"device": "cuda"})
llm=HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    model_kwargs={"temperature":0.2, "max_length":256},
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

persist_directory = 'db_QA_testing_1000_200'
embedding = HuggingFaceEmbeddings()
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

retriever = vectordb.as_retriever()
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# MAKE A CHAIN TO ANSWER QUERIES
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

def process_llm_response(llm_response):
    print(llm_response['result'])
    print(llm_response['source_documents'][0].metadata)

while True:
    query = input("What is your query? (Type QUIT to stop)")
    llm_response = qa_chain(query)
    process_llm_response(llm_response)