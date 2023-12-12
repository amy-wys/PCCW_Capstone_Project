from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

class DocumentProcessor:
    def __init__(self, directory):
        self.directory = directory
    
    def load_and_process_documents(self):
        loader = DirectoryLoader(self.directory, glob="./*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        return texts