# **AI-integrated Medical Chatbot**

## **Project Description**

This project presents an **AI-integrated medical chatbot** designed to provide preliminary health information, answer medical questions, and offer general advice based on a pre-existing knowledge base. It's intended to be a **first-line resource** for users seeking quick, non-emergency health-related information. The chatbot leverages a powerful tech stack to understand user queries, process natural language, and deliver relevant responses, making it an accessible and intelligent tool for health education and awareness.

The core of the system is a **Retrieval-Augmented Generation (RAG)** pipeline. This approach allows the chatbot to retrieve information from a curated knowledge base (a collection of medical data) and then use a Large Language Model (LLM) to generate a coherent and context-aware response, rather than solely relying on the LLM's pre-trained knowledge.

---

## **Key Features**

* **Symptom-based Queries**: Users can describe symptoms and receive information about potential conditions, related remedies, and general advice.
* **Knowledge-based Responses**: The chatbot retrieves relevant medical information from a dedicated knowledge base, ensuring responses are grounded in specific, trusted data.
* **Natural Language Understanding**: Uses advanced NLP techniques to interpret user questions accurately, even with complex or conversational language.
* **User-friendly Interface**: The Flask-based web application provides a simple and intuitive interface for seamless interaction.
* **Scalable and Extensible**: The architecture is designed to be easily updated with new medical data and improved AI models.

---

## **Technologies Used**

This project is built using a modern and powerful tech stack for building LLM applications.

* **Python**: The primary programming language for all backend logic, scripting, and application development.
* **LangChain**: A powerful framework used to orchestrate the entire RAG pipeline. LangChain handles the seamless integration of the Large Language Model, the vector database, and the data processing steps.
* **Flask**: A lightweight Python web framework used to create the web server and handle API endpoints for the chatbot interface.
* **GPT (Generative Pre-trained Transformer)**: The core Large Language Model (LLM) used for generating human-like text. It takes the context retrieved from the knowledge base and crafts a natural, conversational response.
* **Pinecone**: A high-performance vector database used to store and index the vectorized embeddings of the medical knowledge base. Pinecone allows for incredibly fast and accurate similarity searches, which is crucial for the RAG pipeline to retrieve the most relevant information for a user's query.

---

## **Getting Started**

### **Prerequisites**

* Python 3.x
* Pip (Python package installer)
* Access to API keys for your GPT model and Pinecone.

### **Installation**

1.  Clone the repository:
    ```bash
    git clone [https://github.com/prasad1482/AI_integrated_Medical_chatBot.git](https://github.com/prasad1482/AI_integrated_Medical_chatBot.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd AI_integrated_Medical_chatBot
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Set up your environment variables. Create a `.env` file in the root directory and add your API keys:
    ```
    OPENAI_API_KEY="your_openai_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    PINECONE_ENVIRONMENT="your_pinecone_environment"
    ```

### **Usage**

1.  Run the main application file:
    ```bash
    python [main_file_name.py] # Replace with your main file, e.g., app.py
    ```
2.  Access the chatbot through the provided web interface at `http://localhost:5000` (or the port specified in your Flask app).

---

## **Future Enhancements**

* **Dynamic Data Ingestion**: Create a user interface for easily adding new medical documents to the knowledge base.
* **Multi-turn Conversation Management**: Implement a memory component to allow the chatbot to remember the context of a conversation for more natural and helpful follow-ups.
* **Integration with more LLMs**: Expand the project to support other open-source or commercial LLMs besides GPT.
* **Advanced UI/UX**: Develop a more robust frontend with features like chat history and user profiles.

---

## **Contributing**

We welcome contributions! If you have suggestions for improvements, please feel free to create a pull request or open an issue.

---

