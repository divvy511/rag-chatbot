# backend/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import chromadb
import uuid
import json
from datetime import datetime
import os
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import io
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")
groq_client = OpenAI(api_key=os.environ.get("GROQ_API_KEY"), base_url=os.environ.get("URL"))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Memory storage (in production, use Redis or database)
conversation_memory: Dict[str, List[Dict]] = {}


# Models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[str]


class DocumentUpload(BaseModel):
    content: str
    filename: str
    doc_type: str


# Helper functions
def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from different file types"""
    if filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    elif filename.endswith(".txt"):
        return file_content.decode("utf-8")
    else:
        raise ValueError("Unsupported file type")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks - like dividing a book into chapters with some overlap"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Convert text to vector embeddings - like creating a numerical fingerprint for each text"""
    return embedding_model.encode(texts).tolist()


def retrieve_relevant_docs(query: str, k: int = 5) -> List[Dict]:
    """Find most relevant documents - like the librarian finding the most relevant books"""
    query_embedding = embedding_model.encode([query])[0]
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=k)

    documents = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            documents.append(
                {
                    "content": doc,
                    "metadata": results["metadatas"][0][i]
                    if results["metadatas"][0]
                    else {},
                    "distance": results["distances"][0][i]
                    if results["distances"] and results["distances"][0]
                    else 0,
                }
            )

    return documents


def get_conversation_history(session_id: str, limit: int = 5) -> List[Dict]:
    """Get recent conversation history - like remembering recent discussions"""
    if session_id not in conversation_memory:
        return []
    return conversation_memory[session_id][-limit:]


def save_conversation(session_id: str, user_message: str, assistant_response: str):
    """Save conversation to memory - like keeping notes of discussions"""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    conversation_memory[session_id].append(
        {
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "assistant": assistant_response,
        }
    )


async def generate_streaming_response(
    query: str, context_docs: List[Dict], history: List[Dict]
) -> str:
    """Generate streaming response using Groq - like the librarian speaking while thinking"""

    # Build context from retrieved documents
    context = "\n\n".join([doc["content"] for doc in context_docs])

    # Build conversation history
    history_text = ""
    for conv in history:
        history_text += f"User: {conv['user']}\nAssistant: {conv['assistant']}\n\n"

    # Create prompt
    prompt = f"""You are a helpful AI assistant with access to a knowledge base. Use the provided context to answer the user's question accurately and comprehensively.

Previous conversation:
{history_text}

Context from knowledge base:
{context}

User question: {query}

Please provide a helpful and accurate response based on the context provided. If the context doesn't contain relevant information, say so clearly."""

    try:
        # Stream response from Groq
        stream = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=1000,
            temperature=0.1,
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield f"data: {json.dumps({'content': content, 'done': False})}\n\n"

        yield f"data: {json.dumps({'content': '', 'done': True, 'full_response': full_response})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"


# API Routes
@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process documents - like adding new books to the library"""
    try:
        content = await file.read()
        text = extract_text_from_file(content, file.filename)

        # Chunk the text
        chunks = chunk_text(text)

        # Generate embeddings
        embeddings = get_embeddings(chunks)

        # Store in ChromaDB
        doc_ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"filename": file.filename, "chunk_id": i} for i in range(len(chunks))
        ]

        collection.add(
            documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=doc_ids
        )

        return {
            "message": f"Document {file.filename} uploaded successfully",
            "chunks_created": len(chunks),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_endpoint(chat_msg: ChatMessage):
    """Main chat endpoint with streaming response"""
    try:
        # Generate session ID if not provided
        session_id = chat_msg.session_id or str(uuid.uuid4())

        # Retrieve relevant documents
        relevant_docs = retrieve_relevant_docs(chat_msg.message)

        # Get conversation history
        history = get_conversation_history(session_id)

        # Generate streaming response
        async def stream_generator():
            full_response = ""
            async for chunk in generate_streaming_response(
                chat_msg.message, relevant_docs, history
            ):
                if chunk.startswith("data: "):
                    data = json.loads(chunk[6:-2])  # Remove "data: " and "\n\n"
                    if data.get("done") and data.get("full_response"):
                        full_response = data["full_response"]
                        # Save conversation to memory
                        save_conversation(session_id, chat_msg.message, full_response)
                yield chunk

            # Send final metadata
            sources = [
                doc["metadata"].get("filename", "Unknown") for doc in relevant_docs
            ]
            final_data = {
                "session_id": session_id,
                "sources": list(set(sources)),
                "done": True,
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-ID": session_id,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{session_id}")
async def get_conversation_history_endpoint(session_id: str):
    """Get conversation history for a session"""
    history = get_conversation_history(session_id, limit=50)
    return {"session_id": session_id, "history": history}


@app.delete("/conversations/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversation_memory:
        del conversation_memory[session_id]
    return {"message": "Conversation cleared"}


@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        # Get all documents from ChromaDB
        results = collection.get()
        documents = {}

        for metadata in results["metadatas"]:
            filename = metadata.get("filename", "Unknown")
            if filename not in documents:
                documents[filename] = 0
            documents[filename] += 1

        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
