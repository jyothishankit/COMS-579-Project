from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("LLMbasedTesting.pdf")
pages = loader.load_and_split()

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
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

# Ingest the documents into Weaviate
vector_db = Weaviate.from_documents(
    split_docs, embeddings, client=client, by_text=False
)


uuid = client.data_object.create({
    'hello': 'World!'
}, 'MyClass')

obj = client.data_object.get_by_id(uuid, class_name='MyClass')

print(json.dumps(obj, indent=2))



client.schema.create_class({
    'class': 'Wine'
})

client.data_object.create({
    'name': 'Chardonnay',
    'review': 'Goes well with fish!',
}, 'Wine')

response = (
    client.query
    .get('Wine', ['name', 'review'])
    .with_near_text({
        'concepts': ['great for seafood']
    })
    .do()
)

assert response['data']['Get']['Wine'][0]['review'] == 'Goes well with fish!'

print(response)