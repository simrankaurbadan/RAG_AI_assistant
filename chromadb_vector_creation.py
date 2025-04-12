# This is a sample code to create a chromadb vector database using persistent client.
# The total data received had 945278 dictionaries with input and output sentences. 
# Now for explaining how the creation happens, I have created and attached a db_sample which has the embedded data. The user can use this
# data directly. 
# For example purpose, I am creating embedding in a batch_size of 100. collection_name i have given is sample_data. 
# Length for which i am doing embedding is for only 10000 data points here for code explanation. The user can change this number to the 
# actual length of the data. Since the db_sample would have become bulky (around 6-7GB for full data), I am taken only a sample for 
# uploading perspective.

import json
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import uuid
from tqdm import tqdm

from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv('OPEN_AI_API__KEY')

DB_DIR = os.path.join(os.getcwd(), "db_sample")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=DB_DIR)

# OpenAI Embedding Function
embedding_function = OpenAIEmbeddingFunction(
    model_name='text-embedding-3-small',
    api_key=  openai_api_key # Keep your key secure
)

collection_name = 'sample_data' # Mention the collection_name you want to keep

# Create or Get Collection
if collection_name in [c.name for c in client.list_collections()]:
    collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
else:
    collection = client.create_collection(name=collection_name, embedding_function=embedding_function)

# Load Data
with open('preprocessed_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total records loaded: {len(data)}")

# Insert in Batches
batch_size = 100
valid_records = []

for record in data:
    if 'input' in record and isinstance(record['input'], str) and record['input'].strip() != '':
        valid_records.append(record)

print(f"Total valid records for insertion: {len(valid_records)}")

for i in tqdm(range(0, 10000, batch_size)):
    batch = valid_records[i:i + batch_size]

    documents = [r['input'] for r in batch]
    metadatas = [{'output': r['output']} for r in batch]
    ids = [str(uuid.uuid4()) for _ in batch]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

print("Data Insertion Completed Successfully!")
