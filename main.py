import sys
import weaviate
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

if len(sys.argv) < 2:
  print("Atleast one of --pdf_file or --question must be sent. Usage: python main.py --pdf_file=a.pdf --question=\"What is the capital of USA?\"")
  exit()

pdf_file = None
question = None
for arg in sys.argv[1:]:
  if arg.startswith("--pdf_file="):
    pdf_file = arg.split("=")[1]
  elif arg.startswith("--question="):
    question = arg.split("=")[1]
  else:
    print(f"Invalid argument: {arg}")
    exit()

if not pdf_file or not question:
  print("Error: Missing required arguments!")
  exit()

if(len(sys.argv)<2):
    print("No pdf document passed")
    exit(0)

loader = PyPDFLoader(pdf_file)
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
split_docs = text_splitter.split_documents(pages)

#Can print initial file contents
#print(split_docs[0].page_content)
embedding_model_name = "google-bert/bert-base-uncased"

WEAVIATE_URL="http://localhost:8080"
# class_object_document = {
#     "class": "Document",
#     "vectorizer": "text2vec-huggingface",
#     "moduleConfig": {
#         "text2vec-huggingface": {
#             "vectorizeClassName": True
#         }
#     }
# }

client = weaviate.Client(
    url=WEAVIATE_URL
)

#client.schema.create_class(class_object_document)

embeddings = HuggingFaceEmbeddings(
  model_name=embedding_model_name
)
#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

#split_docs=split_docs[:5]
print(str(len(split_docs))+" is the length of split docs")
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

