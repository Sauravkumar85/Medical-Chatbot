
import os
import threading
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


app = Flask(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=False)

# Render environment variables are configured in the dashboard.
pinecone_api_key = os.getenv("PINECONE_API_KEY")
google_api_key = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if pinecone_api_key:
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

# Initialize the chatbot only when needed.
rag_chain = None
initialization_lock = threading.Lock()


def get_rag_chain():
    global rag_chain

    if rag_chain is not None:
        return rag_chain

    with initialization_lock:
        if rag_chain is not None:
            return rag_chain

        if not pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is missing in Render Environment Variables.")

        if not google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is missing in Render Environment Variables.")

        app.logger.info("Loading embedding model...")
        embeddings = download_hugging_face_embeddings()
        app.logger.info("Embedding model loaded.")

        index_name = "medical-chatbot"
        app.logger.info("Connecting to Pinecone index: %s", index_name)

        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )

        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        chat_model = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=google_api_key,
            temperature=0.3
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(
            chat_model,
            prompt
        )

        rag_chain = create_retrieval_chain(
            retriever,
            question_answer_chain
        )

        app.logger.info("Medical chatbot initialized successfully.")
        return rag_chain


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    user_message = request.values.get("msg", "").strip()

    if not user_message:
        return "Please enter a question.", 400

    try:
        chain = get_rag_chain()
        response = chain.invoke({"input": user_message})

        return str(
            response.get(
                "answer",
                "Sorry, I could not generate an answer."
            )
        )

    except Exception:
        app.logger.exception("Medical chatbot request failed")
        return (
            "Sorry, the chatbot could not process your question. "
            "Please check the Render logs."
        ), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)