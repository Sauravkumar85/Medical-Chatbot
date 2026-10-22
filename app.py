
import os
import time
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt


# --------------------------------------------------
# 1. Flask application and environment variables
# --------------------------------------------------

app = Flask(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=False)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# --------------------------------------------------
# 2. Lazy initialization
# --------------------------------------------------

rag_chain = None
initialization_lock = threading.Lock()


def get_rag_chain():
    global rag_chain

    if rag_chain is not None:
        return rag_chain

    with initialization_lock:
        if rag_chain is not None:
            return rag_chain

        if not PINECONE_API_KEY:
            raise RuntimeError(
                "PINECONE_API_KEY is missing from environment variables."
            )

        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is missing from environment variables."
            )

        start_time = time.monotonic()

        # Load embedding model
        app.logger.info("STEP 1: Loading embedding model...")

        embeddings = download_hugging_face_embeddings()

        app.logger.info(
            "STEP 1 COMPLETE: Embeddings loaded in %.2f seconds.",
            time.monotonic() - start_time,
        )

        # Connect to existing Pinecone index
        app.logger.info("STEP 2: Connecting to Pinecone...")

        docsearch = PineconeVectorStore.from_existing_index(
            index_name="medical-chatbot",
            embedding=embeddings,
        )

        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        app.logger.info("STEP 2 COMPLETE: Pinecone connected.")

        # Configure Gemini
        app.logger.info("STEP 3: Configuring Gemini...")

        chat_model = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            timeout=45,
            max_retries=1,
        )

        # Build RAG chain
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            chat_model,
            prompt,
        )

        rag_chain = create_retrieval_chain(
            retriever,
            question_answer_chain,
        )

        app.logger.info(
            "STEP 4 COMPLETE: RAG chain initialized in %.2f seconds.",
            time.monotonic() - start_time,
        )

        return rag_chain


# --------------------------------------------------
# 3. Home page
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("chat.html")


# --------------------------------------------------
# 4. Chat endpoint
# --------------------------------------------------

@app.route("/get", methods=["GET", "POST"])
def chat():
    user_message = request.values.get("msg", "").strip()

    if not user_message:
        return "Please enter a question.", 400

    request_start = time.monotonic()

    try:
        app.logger.info("CHAT STEP 1: Question received.")

        chain = get_rag_chain()

        app.logger.info("CHAT STEP 2: RAG chain initialized.")
        app.logger.info("CHAT STEP 3: Running retrieval and Gemini.")

        response = chain.invoke({"input": user_message})

        answer = response.get("answer")

        if not answer:
            raise RuntimeError("The chatbot returned an empty answer.")

        elapsed = time.monotonic() - request_start

        app.logger.info(
            "QUESTION ANSWERED in %.2f seconds.",
            elapsed,
        )

        return str(answer), 200

    except Exception:
        app.logger.exception(
            "CHATBOT REQUEST FAILED after %.2f seconds.",
            time.monotonic() - request_start,
        )

        return (
            "The chatbot could not process your question. "
            "Please check the Render logs for the exact error."
        ), 500


# --------------------------------------------------
# 5. Run locally
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )

