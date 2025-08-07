# RAG-APP

## Overview
RAG-APP is a Retrieval-Augmented Generation (RAG) application that allows you to chat with the contents of a PDF document using LangChain, ChromaDB, and Google Gemini. The app is built with Streamlit for an interactive chat interface.

## Features
- Upload a PDF and ask questions about its content
- Uses vector search for document retrieval
- Supports Google Gemini LLM and OpenAI embeddings
- Session-based chat history

## Setup Instructions

### 1. Clone the Repository
```bash
git clone git@github.com/abnsol/RAG-APP
cd RAG-APP
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the project root with the following content:

```
GOOGLE_API_KEY=your_google_api_key_here
GITHUB_TOKEN=your_github_token_here
```

- `GOOGLE_API_KEY`: Your Google Gemini API key for LLM access
- `GITHUB_TOKEN`: Your GitHub Copilot API token for OpenAI embeddings

### 5. Run the App
```bash
streamlit run app.py
```

## Usage
1. Open the Streamlit app in your browser.
2. Upload a PDF file using the sidebar.
3. Ask questions about the document in the chat interface.


