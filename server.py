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
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever

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

# Define the semantic retriever
semantic_retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# To use BM25, we need the text chunks. Since you're indexing separately,
# we need to load them here to build the BM25 index in memory.
# This is a key step to make hybrid search work.
# For this example, we'll use a placeholder for a small keyword index.
text_chunks = [
    Document(page_content="Acne is a common skin condition that occurs when hair follicles become clogged with oil and dead skin cells."),
    Document(page_content="Influenza is a contagious respiratory illness caused by influenza viruses. It can cause mild to severe illness."),
    Document(page_content="Pneumonia is an infection that inflames air sacs in one or both lungs. The air sacs may fill with fluid or pus.")
]
keyword_retriever = BM25Retriever.from_documents(text_chunks)
keyword_retriever.k = 3

# Combine both retrievers into a single hybrid retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, keyword_retriever],
    weights=[0.6, 0.4]
)

# Define the LLM and prompt template
chatModel = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

prompt = ChatPromptTemplate.from_template(
    "You are MedBot, a helpful and knowledgeable medical assistant. "
    "You are not a doctor, but you can provide general health information, "
    "explain symptoms, suggest possible causes, and guide users toward professional help. "
    "Always remind users to consult a licensed healthcare provider for diagnosis or treatment. "
    "Be empathetic, respectful, and clear in your responses. Avoid giving definitive diagnoses or prescribing medications."
    "\n\nIf a user asks for emergency help, advise them to contact emergency services immediately."
    "\n\nTone: Professional, supportive, and informative."
    "\nKnowledge Base: General medical knowledge, symptoms, wellness tips, and healthcare guidance."
    "\nLimitations: Do not provide medical diagnoses, prescribe treatments, or replace professional medical advice."
    "\n\nContext:\n{context}\n\nQuestion: {input}"
)

# Create the RAG chain
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(hybrid_retriever, question_answer_chain)

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

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Invoke the RAG chain
        response = rag_chain.invoke({"input": user_message})
        return jsonify({"answer": response["answer"]})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True)
