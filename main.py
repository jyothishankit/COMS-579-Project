import sys
import weaviate
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_text_splitters import RecursiveCharacterTextSplitter
if(len(sys.argv)<2):
    print("No pdf document passed")
    exit(0)

loader = PyPDFLoader(sys.argv[1])
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
split_docs = text_splitter.split_documents(pages)

print(split_docs[0].page_content)
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

split_docs=split_docs[:5]
print(str(len(split_docs))+" is the length of split docs")
vector_store = Weaviate.from_documents(documents=split_docs, embedding=embeddings, client=client)

print("Reached here after storing documents in vector store")
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke("What is the relation between somatic mutations with age?")

len(retrieved_docs)
print(retrieved_docs[0].page_content)
