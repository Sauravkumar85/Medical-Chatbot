
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)


# 1. Load environment variables
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise RuntimeError(
        "PINECONE_API_KEY is missing. Check your .env file."
    )


# 2. Load PDF documents
data_dir = PROJECT_DIR / "data"
extracted_data = load_pdf_file(str(data_dir))

if not extracted_data:
    raise RuntimeError(
        f"No PDF documents found in {data_dir}"
    )

print(f"Loaded {len(extracted_data)} document pages.")


# 3. Filter documents and split into chunks
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

if not text_chunks:
    raise RuntimeError("No text chunks were created.")

print(f"Created {len(text_chunks)} text chunks.")


# 4. Load the Hugging Face embedding model
embeddings = download_hugging_face_embeddings()


# 5. Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )
    print(f"Created Pinecone index: {index_name}")
else:
    print(f"Using existing Pinecone index: {index_name}")


# 6. Store document chunks in Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("Medical documents stored in Pinecone successfully.")