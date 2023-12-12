from langchain.embeddings import HuggingFaceInstructEmbeddings

model_kwargs={'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

bge_base_embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-large-en-v1.5", 
                                          model_kwargs=model_kwargs, 
                                          encode_kwargs = encode_kwargs)

