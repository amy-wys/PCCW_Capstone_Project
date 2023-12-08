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

# LOAD DOCUMENTS AND SPLIT TEXT
loader = DirectoryLoader('./data/', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# CREATE VECTOR DB
persist_directory = 'db_1000_500_test'
embedding = HuggingFaceEmbeddings()
vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)
vectordb.persist()
vectordb = None
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)
