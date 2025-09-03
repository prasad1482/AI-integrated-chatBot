from dotenv import load_dotenv
import os
# Corrected imports from langchain_community to fix deprecation warnings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# We need to define the helper functions here since they are part of your local project.
def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def filter_to_minimal_doc(documents):
    # Your filter logic
    return documents

def text_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Changed the embedding model to the one we are using in the main server
def downloading_embedings():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

print("Starting data processing and indexing...")
extracted_docs = load_pdf_files("data")
minimal_docs = filter_to_minimal_doc(extracted_docs)
text_chunks = text_split(minimal_docs)
embeddings = downloading_embedings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = 'medical-index'
if pc.has_index(index_name):
    pc.delete_index(index_name)
pc.create_index(
    name=index_name,
    dimension=768,  # Match your embedding model's output
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)

# Corrected the attribute name from .index to .Index
index = pc.Index(index_name)

# Here you would upsert documents to the index
# Note: PineconeVectorStore.from_documents automatically handles upserting.
# Corrected the PineconeVectorStore.from_documents() call to use batching
batch_size = 100
for i in range(0, len(text_chunks), batch_size):
    batch = text_chunks[i:i + batch_size]
    if i == 0:
        # For the first batch, create the PineconeVectorStore instance
        docsearch = PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=index_name
        )
    else:
        # For subsequent batches, add documents to the existing instance
        docsearch.add_documents(batch)

print("Pinecone index has been successfully created and populated.")
print(f"Total documents indexed: {len(text_chunks)}")
