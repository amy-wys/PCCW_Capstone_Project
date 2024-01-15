import os
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from process_doc import DocumentProcessor
from create_db import DB
from get_ans import create_qa_chain, process_llm_response


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
    db = DB(persist_directory)
    vectordb = db.create_vectordb(texts)

    # Create the retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    
    # Create the question-answering chain
    qa_chain = create_qa_chain(llm, retriever)


    while True:
        user_input = input("\nUser: (Press 'Enter' to exit)")
        if user_input:
            llm_response = qa_chain(user_input)
            process_llm_response(llm_response)
        else:
            print("See you.")
            break