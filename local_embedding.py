from langchain.embeddings import HuggingFaceInstructEmbeddings
from chromadb import Documents, EmbeddingFunction, Embeddings

model_kwargs={'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

bge_base_embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-large-en-v1.5", 
                                          model_kwargs=model_kwargs, 
                                          encode_kwargs = encode_kwargs)

class BGE_embeddings(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = bge_base_embeddings.embed_documents(input)
        return embeddings