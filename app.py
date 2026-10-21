
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt


# --------------------------------------------------
# 1. Initialize Flask application
# --------------------------------------------------

app = Flask(__name__)

PROJECT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# 2. Load environment variables from .env
# --------------------------------------------------

load_dotenv(PROJECT_DIR / ".env", override=False)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
google_api_key = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if not pinecone_api_key:
    raise RuntimeError(
        "PINECONE_API_KEY is missing. Check your .env file."
    )

if not google_api_key:
    raise RuntimeError(
        "GOOGLE_API_KEY or GEMINI_API_KEY is missing. "
        "Check your .env file."
    )

# Make the keys available to the respective SDKs
os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key


# --------------------------------------------------
# 3. Load Hugging Face embedding model
# --------------------------------------------------

print("Loading embedding model...")

embeddings = download_hugging_face_embeddings()

print("Embedding model loaded successfully.")


# --------------------------------------------------
# 4. Connect to the existing Pinecone index
# --------------------------------------------------

index_name = "medical-chatbot"

print(f"Connecting to Pinecone index: {index_name}")

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

print("Pinecone retriever initialized successfully.")


# --------------------------------------------------
# 5. Initialize Gemini model
# --------------------------------------------------

chatModel = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=google_api_key,
    temperature=0.3
)

print("Gemini model initialized successfully.")


# --------------------------------------------------
# 6. Create prompt template
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# --------------------------------------------------
# 7. Create the RAG chain
# --------------------------------------------------

question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

print("Medical chatbot RAG chain initialized successfully.")


# --------------------------------------------------
# 8. Flask home route
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("chat.html")


# --------------------------------------------------
# 9. Flask chat route
# --------------------------------------------------

@app.route("/get", methods=["GET", "POST"])
def chat():
    user_message = request.values.get("msg", "").strip()

    if not user_message:
        return "Please enter a question.", 400

    try:
        response = rag_chain.invoke({
            "input": user_message
        })

        answer = response.get(
            "answer",
            "Sorry, I could not generate an answer."
        )

        return str(answer)

    except Exception:
        app.logger.exception("Medical chatbot request failed")
        return (
            "Sorry, an error occurred while processing "
            "your question. Please try again."
        ), 500


# --------------------------------------------------
# 10. Run Flask application
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
