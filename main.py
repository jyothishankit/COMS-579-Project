import sys
import weaviate
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank


def upload(pdf_file_1,pdf_file_2,pdf_file_3,question):
    loader = PyPDFLoader(pdf_file_1)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    split_docs_1 = text_splitter.split_documents(pages)
    print(str(len(split_docs_1))+" is the length of split docs for pdf 1")

    loader = PyPDFLoader(pdf_file_2)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    split_docs_2 = text_splitter.split_documents(pages)
    print(str(len(split_docs_2))+" is the length of split docs for pdf 2")

    loader = PyPDFLoader(pdf_file_3)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    split_docs_3 = text_splitter.split_documents(pages)
    print(str(len(split_docs_3))+" is the length of split docs for pdf 3")


    embedding_model_name = "google-bert/bert-base-uncased"

    WEAVIATE_URL="http://localhost:8080"

    client = weaviate.Client(
        url=WEAVIATE_URL
    )

    embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name
    )
    split_docs = split_docs_1 + split_docs_2 + split_docs_3
    vector_store = Weaviate.from_documents(documents=split_docs, embedding=embeddings, client=client)

    print("Reached here after storing documents in vector store")
    print("\n\n")
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    print("Question is "+question)
    retrieved_docs = retriever.invoke(question)
    print("\n\n")
    print("Full output of retriever")
    print(retrieved_docs)
    print("\n")
    print("\n")
    print("Answer to the question:\n")
    print(retrieved_docs[0].page_content)

    print("\n\n")
    print("Using reranking for fetching based on context")
    retriever_contextual_with_reranking = ContextualCompressionRetriever(base_retriever=retriever,base_compressor=FlashrankRerank(),query=question, search_kwargs={"k": 5})

    reranked_docs = retriever_contextual_with_reranking.get_relevant_documents(question)

    print("\n\n")
    print("Output of reranked documents with relevance scores")
    print(reranked_docs)

    print("\n\n")
    print("Top answer of reranked vectors")
    print(reranked_docs[0].page_content)
    final_answer = reranked_docs[0].page_content
    return final_answer
