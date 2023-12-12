import os
import shutil
from langchain.vectorstores import Chroma
from local_embedding import bge_base_embeddings


class DB:
    def __init__(self, persist_directory):
        self.persist_directory = persist_directory
        
    def delete_vectordb(self):
        if os.path.exists(self.persist_directory):
            # Delete the existing database directory
            shutil.rmtree(self.persist_directory)
    
    def create_vectordb(self, texts):
        self.delete_vectordb()
        
        vectordb = Chroma.from_documents(documents=texts, embedding=bge_base_embeddings, persist_directory=self.persist_directory)
        # Persist the database to disk
        vectordb.persist()
        vectordb = None
        # Load the persisted database from disk
        vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=bge_base_embeddings)
        return vectordb