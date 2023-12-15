from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from prompts import answer_template_from_contexts

def create_qa_chain(llm, retriever):
    prompt = PromptTemplate(template=answer_template_from_contexts, 
                            input_variables=['context', 'question'])
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, 
                                       chain_type_kwargs={"prompt": prompt}, return_source_documents=True)


def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])