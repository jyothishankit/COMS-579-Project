from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("LLMbasedTesting.pdf")
pages = loader.load_and_split()

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

split_docs = text_splitter.split_documents(pages)

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(
  model_name=embedding_model_name
)

WEAVIATE_URL="http://localhost:8080"

import weaviate
import json

client = weaviate.Client(
    url=WEAVIATE_URL
)

print("Reached here 0 $$$")
# Ingest the documents into Weaviate

print("Reached here 1 $$$")
uuid = client.data_object.create({
    'hello': 'World!'
}, 'MyClass')

print("Reached here 2 $$$")

obj = client.data_object.get_by_id(uuid, class_name='MyClass')

print("Reached here 3 $$$")
print(json.dumps(obj, indent=2))

print("Reached here 4 $$$")


# client.schema.create_class({
#     'class': 'Wine'
# })

client.data_object.create({
    'name': 'Chardonnay',
    'review': 'Goes well with fish!',
}, 'Wine')

print("Reached here 5 $$$")
response = (
    client.query
    .get('Wine', ['name', 'review'])
    .with_near_text({
        'concepts': ['great for seafood']
    })
    .do()
)

print("Reached here 6 $$$")
#assert response['data']['Get']['Wine'][0]['review'] == 'Goes well with fish!'

print("Reached here 7 $$$")
print(response)