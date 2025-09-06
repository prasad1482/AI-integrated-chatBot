from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.chains import create_history_aware_retriever
from scr.helper import load_pdf_files, filter_to_minimal_doc, text_split

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Flask App setup
app = Flask(__name__)
CORS(app)

# --- RAG SYSTEM SETUP ---

# Initialize embedding model to generate query embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'medical-index'

# Connect to the existing Pinecone index
docsearch = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
)

# To use BM25, we need the text chunks. We will load the full dataset here.
print("Loading medical data for BM25 retriever...")
# We use os.path.dirname(__file__) to get the directory of the current file (app.py)
# and then construct an absolute path to the data directory.
data_path = os.path.join(os.path.dirname(__file__), "data")
if not os.path.exists(data_path):
    print(f"Error: The data directory '{data_path}' was not found.")
    text_chunks = []
else:
    extracted_docs = load_pdf_files(data_path)
    minimal_docs = filter_to_minimal_doc(extracted_docs)
    text_chunks = text_split(minimal_docs)

if not text_chunks:
    print("Warning: No text chunks were loaded. The keyword retriever will be empty.")

keyword_retriever = BM25Retriever.from_documents(text_chunks)
keyword_retriever.k = 3

# Define the semantic retriever
semantic_retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Combine both retrievers into a single hybrid retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, keyword_retriever],
    weights=[0.6, 0.4]
)

# Define the LLM and prompt template
chatModel = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# Contextualize Question Prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", 
         "Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question. "
         "If the user's question is already a standalone question, simply return it without modification. "
         "\n\nFollow-up Question: {input}"
         "\n\nStandalone question:")
    ]
)

# The new history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    chatModel, hybrid_retriever, contextualize_q_prompt
)

# Answer Prompt
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You are MedBot, a helpful and knowledgeable medical assistant. "
         "You are not a doctor, but you can provide general health information, "
         "explain symptoms, suggest possible causes, and guide users toward professional help. "
         "Always remind users to consult a licensed healthcare provider for diagnosis or treatment. "
         "Be empathetic, respectful, and clear in your responses. Avoid giving definitive diagnoses or prescribing medications."
         "\n\nIf a user asks for emergency help, advise them to contact emergency services immediately."
         "\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# Final RAG chain
question_answer_chain = create_stuff_documents_chain(chatModel, answer_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# In-memory session store
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Final chain with message history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# --- API ENDPOINT ---

@app.route("/")
def home():
    """Serves the main HTML file for the chatbot interface."""
    try:
        with open("interface/frontend.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "Error: frontend.html not found.", 404

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message")
    session_id = data.get("session_id", "default_session_id") # Use a default ID or create one

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Invoke the RAG chain
        response = conversational_rag_chain.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}}
        )
        return jsonify({"answer": response["answer"]})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True)

