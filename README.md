# RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg) ![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg) ![Groq](https://img.shields.io/badge/Groq-LLM-orange.svg) ![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-purple.svg) !

Retrieval-Augmented Generation chatbot using FastAPI, React, Groq, and ChromaDB.


## 📚 Overview

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot that allows you to:
- Upload **PDF, DOCX, or TXT** documents
- Store and search them using **ChromaDB**
- Retrieve relevant chunks and pass them to **Groq's LLaMA model**
- Get **streaming AI responses** with **conversation memory**



## 📂 Project Structure

```bash
rag-chatbot/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env
│   └── chroma_db/
├── frontend/
│   ├── public/
│   ├── src/
├── README.md
└── .gitignore

```

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```


### 2️⃣ Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```


Create `.env` file in backend/

```env
GROQ_API_KEY=your_groq_api_key
URL=https://api.groq.com/openai/v1
```


Run backend

```bash
cd ../backend
python main.py
```


### 3️⃣ Frontend Setup

```bash
cd ../frontend
npm install
npm start
```



## 📜 API Endpoints

Method | Endpoint | Description
--- | --- | ---
POST | /upload-document | Upload and process a document
POST | /chat | Send a query and get AI response
GET | /documents | List all stored documents

