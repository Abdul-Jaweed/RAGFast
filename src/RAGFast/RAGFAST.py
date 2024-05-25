import os
import requests
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser



def OpenAIRAGFast(
    txt=None,
    pdf=None,
    directory_pdf=None,
    chunk_size=1000,
    chunk_overlap=200,
    vectorstore_db=FAISS,
    openai_api_key=None,
    template=None,
    openai_model_name=None,
    
):
    if txt:
        loader = TextLoader(txt, encoding = 'UTF-8')
    if pdf:
        loader = PyPDFLoader(pdf)
    if directory_pdf:
        loader = PyPDFDirectoryLoader(directory_pdf)
    
    document = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    text_chunks = text_splitter.split_documents(document)
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key
    )
    
    vectorstore = vectorstore_db.from_documents(text_chunks, embeddings)
    
    retriever = vectorstore.as_retriever()
    
    template = template
    
    prompt = ChatPromptTemplate.from_template(template)
    
    output_parser = StrOutputParser()
    
    llm_model = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=openai_model_name
    )
    
    rag_chain = (
        {"context": retriever, "question":RunnablePassthrough()}
        | prompt
        | llm_model
        | output_parser
    )
    
    return rag_chain
