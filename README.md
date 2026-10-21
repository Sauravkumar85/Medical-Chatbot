# Medical Chatbot Using LLMs, LangChain, Gemini, and Pinecone

An AI-powered medical question-answering chatbot developed using Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), LangChain, Google Gemini, Pinecone, Hugging Face, and Flask.

The application retrieves relevant information from medical PDF documents stored in a vector database and uses Google Gemini to generate concise, context-aware responses through an interactive web interface.

## Features

* **Medical Question Answering:** Provides answers to medical questions using information retrieved from medical documents.
* **Retrieval-Augmented Generation (RAG):** Combines semantic document retrieval with language model generation.
* **PDF Processing:** Extracts text from medical PDF documents and divides it into manageable chunks.
* **Semantic Search:** Uses Hugging Face embeddings and Pinecone for similarity-based document retrieval.
* **Google Gemini Integration:** Generates responses using a Gemini language model.
* **Interactive Web Interface:** Provides a chat interface built with Flask, HTML, CSS, JavaScript, and Bootstrap.
* **Environment Variable Management:** Stores API credentials in environment variables rather than application source code.

## Technology Stack

| Component                | Technology                               |
| ------------------------ | ---------------------------------------- |
| Programming Language     | Python                                   |
| Web Framework            | Flask                                    |
| Large Language Model     | Google Gemini                            |
| LLM Framework            | LangChain                                |
| Vector Database          | Pinecone                                 |
| Embedding Model          | `sentence-transformers/all-MiniLM-L6-v2` |
| PDF Processing           | PyPDF and LangChain document loaders     |
| Frontend                 | HTML, CSS, JavaScript, Bootstrap         |
| Configuration Management | python-dotenv                            |

## Project Structure

```text
Medical_Chatbot/
│
├── app.py
├── store_index.py
├── setup.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── Medical_book.pdf
│
├── research/
│   └── trials.ipynb
│
├── src/
│   ├── __init__.py
│   ├── helper.py
│   └── prompt.py
│
├── templates/
│   └── chat.html
│
└── static/
    └── style.css
```

*Note: The structure represents the intended organization of the project. Only files and directories present in the repository should be included. The `.env` file is excluded from version control.*

## System Architecture and Workflow

The application follows a Retrieval-Augmented Generation (RAG) pipeline.

1. **Document Loading:** Medical PDF documents are loaded from the `data/` directory.
2. **Text Extraction and Splitting:** Extracted text is divided into chunks of 500 characters with a 20-character overlap.
3. **Embedding Generation:** The Hugging Face model converts text chunks into 384-dimensional vector embeddings.
4. **Vector Storage:** Pinecone stores the embeddings for efficient similarity search.
5. **Context Retrieval:** The three most relevant document chunks are retrieved for each user query.
6. **Response Generation:** Google Gemini generates a concise answer using the retrieved context and the configured system prompt.
7. **Response Delivery:** Flask returns the generated answer to the web interface.

## Prerequisites

The following components are required:

* Python 3.13, subject to compatibility with the installed dependency versions.
* Git.
* Google AI API credentials for Gemini.
* Pinecone API credentials.
* Internet connectivity for API requests and initial model downloads.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <GITHUB_REPOSITORY_URL>
cd <PROJECT_FOLDER>
```

Replace the placeholders with the repository URL and project directory name.

### 2. Create a Virtual Environment

For Windows:

```powershell
py -3.13 -m venv .venv
```

### 3. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell execution policies prevent activation, the environment's Python executable can be used directly.

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency file must contain compatible versions of all packages imported by the application.

### 5. Configure Environment Variables

Create a `.env` file in the project root directory containing:

```dotenv
PINECONE_API_KEY=<PINECONE_API_KEY>
GOOGLE_API_KEY=<GOOGLE_AI_API_KEY>
```

The placeholders must be replaced with valid credentials. The `.env` file must not be committed to version control.

### 6. Prepare the Medical Documents

Place the required medical PDF documents inside the `data/` directory.

Example:

```text
data/
└── Medical_book.pdf
```

The document-loading utility processes PDF files available in this directory.

### 7. Create or Populate the Pinecone Index

Execute the following command:

```bash
python store_index.py
```

The script loads the medical documents, generates text chunks and embeddings, and stores the resulting vectors in the Pinecone index named `medical-chatbot`.

The index configuration must match the embedding model and vector dimensions used by the application. Repeated ingestion should be avoided unless the index needs to be repopulated, as duplicate vectors may be created.

This step can be skipped when the existing index has already been populated successfully.

### 8. Run the Flask Application

Start the application using:

```bash
python app.py
```

After successful startup, the chatbot interface is accessible at:

**http://127.0.0.1:8080**

Medical questions can then be submitted through the web interface.

## Configuration Details

| Parameter                  | Configuration                            |
| -------------------------- | ---------------------------------------- |
| Pinecone Index             | `medical-chatbot`                        |
| Embedding Model            | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding Dimension        | 384                                      |
| Retrieval Method           | Similarity Search                        |
| Number of Retrieved Chunks | 3                                        |
| Text Chunk Size            | 500 characters                           |
| Text Chunk Overlap         | 20 characters                            |
| Maximum Response Length    | Three sentences                          |

Consistent configuration between document ingestion and application execution is essential for reliable retrieval.

## Security Considerations

* API credentials are managed through environment variables.
* The `.env` file and virtual environment directories are excluded from version control.
* Private medical documents and sensitive user conversations should not be published.
* Secure deployment settings and appropriate access controls should be implemented.
* API keys and confidential information should not be included in application logs.

## Limitations

* Response quality depends on the accuracy, relevance, and completeness of the medical documents.
* Retrieved information may be outdated or incomplete.
* Generated responses may contain inaccuracies.
* The chatbot is intended for informational and educational purposes and does not replace professional medical advice, diagnosis, or treatment.
* Emergency medical situations require immediate assistance from qualified healthcare professionals.

## Future Enhancements

* Conversation history and multi-turn question answering.
* Source citations and document metadata filtering.
* Medical document upload functionality.
* Enhanced request validation and error handling.
* Automated testing and continuous integration.
* Deployment to a suitable cloud hosting platform.

## License

An appropriate open-source license should be selected before public distribution. Redistribution rights for all included medical documents and third-party resources must be verified.

## Acknowledgements

The project uses open-source technologies and services, including LangChain, Hugging Face, Flask, and Pinecone, along with Google's Gemini API.

