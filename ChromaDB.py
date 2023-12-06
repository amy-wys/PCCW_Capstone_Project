from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from dotenv import load_dotenv
import os

class DocumentProcessor:
    def __init__(self, directory):
        self.directory = directory
    
    def load_and_process_documents(self):
        loader = DirectoryLoader(self.directory, glob="./*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        return texts

class VectorStore:
    def __init__(self, embedding, persist_directory):
        self.embedding = embedding
        self.persist_directory = persist_directory
    
    def create_and_persist(self, texts):
        vectordb = Chroma.from_documents(documents=texts, embedding=self.embedding, persist_directory=self.persist_directory)
        vectordb.persist()
        return vectordb
    
    def load_from_disk(self):
        vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding)
        return vectordb

class QASystem:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    
    def process_llm_response(self, llm_response):
        print(llm_response['result'])
        print('\n\nSources:')
        for source in llm_response["source_documents"]:
            print(source.metadata['source'])
    
    def answer_question(self, user_input):
        llm_response = self.qa_chain(user_input)
        self.process_llm_response(llm_response)

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve the Hugging Face Hub API token
    HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

    # Initialize HuggingFaceHub LLM
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={"temperature": 0.2, "max_length": 256},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    # Process and load the documents
    document_processor = DocumentProcessor('./data/')
    texts = document_processor.load_and_process_documents()

    # Embed and store the texts
    embedding = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-large-en-v1.5", model_kwargs={'device': 'cpu'})
    vector_store = VectorStore(embedding, 'db')
    vectordb = vector_store.create_and_persist(texts)

    # Load the persisted database from disk
    vectordb = vector_store.load_from_disk()

    # Create the retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    # Create the question-answering system
    qa_system = QASystem(llm, retriever)

    while True:
        user_input = input("User: ")
        if user_input:
            qa_system.answer_question(user_input)
            break
        else:
            print("Invalid input.")
            break