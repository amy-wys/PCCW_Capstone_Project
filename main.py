from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from prompts import answer_template_from_contexts
from process_doc import DocumentProcessor
import os
from dotenv import load_dotenv

        
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
    persist_directory = 'db'
    model_kwargs={'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    embedding = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-large-en-v1.5", 
                                            model_kwargs=model_kwargs, 
                                            encode_kwargs = encode_kwargs)
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

    # Persist the database to disk
    vectordb.persist()
    vectordb = None

    # Load the persisted database from disk
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    prompt = PromptTemplate(template=answer_template_from_contexts, 
	 	input_variables=['context', 'question'])

    # Create the retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    # Create the question-answering chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, 
                                           chain_type_kwargs={"prompt": prompt}, return_source_documents=True)

    def process_llm_response(llm_response):
        print(llm_response['result'])
        print('\n\nSources:')
        for source in llm_response["source_documents"]:
            print(source.metadata['source'])

    while True:
        user_input = input("User: ")
        if user_input:
            llm_response = qa_chain(user_input)
            process_llm_response(llm_response)
            break
        else:
            print("Invalid input.")
            break