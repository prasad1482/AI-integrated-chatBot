
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceBgeEmbeddings
from typing import List
from langchain.schema import Document 

# Extract text from PDF files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

# Minimal
def filter_to_minimal_doc(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        scr = doc.metadata.get('source')
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    'source': scr
                }
            )
        )
    return minimal_docs

## split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        
    )
    text_chunks= text_splitter.split_documents(minimal_docs)
    return text_chunks

# download embeddings
def downloading_embedings():
    ''' Download  and return huggingface embeddings '''
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        
    )
    return embeddings
