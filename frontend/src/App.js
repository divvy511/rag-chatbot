// frontend/src/App.js
import React, { useState, useRef, useEffect } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [documents, setDocuments] = useState({});
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load documents on component mount
  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents`);
      const data = await response.json();
      setDocuments(data.documents || {});
    } catch (error) {
      console.error('Error loading documents:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/upload-document`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Upload failed');
      }
      
      const result = await response.json();
      setUploadedFiles(prev => [...prev, {
        name: file.name,
        size: file.size,
        chunks: result.chunks_created
      }]);
      
      // Reload documents list
      await loadDocuments();
      
      // Clear file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      
    } catch (error) {
      console.error('Upload error:', error);
      alert('Failed to upload file. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading || isStreaming) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    
    // Add user message to chat
    setMessages(prev => [...prev, {
      type: 'user',
      content: userMessage,
      timestamp: new Date().toLocaleTimeString()
    }]);

    setIsStreaming(true);
    
    // Add empty assistant message that will be filled via streaming
    const assistantMessageIndex = messages.length + 1;
    setMessages(prev => [...prev, {
      type: 'assistant',
      content: '',
      timestamp: new Date().toLocaleTimeString(),
      sources: [],
      isStreaming: true
    }]);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionId
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullResponse = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.content) {
                fullResponse += data.content;
                // Update the streaming message
                setMessages(prev => {
                  const newMessages = [...prev];
                  if (newMessages[assistantMessageIndex]) {
                    newMessages[assistantMessageIndex].content = fullResponse;
                  }
                  return newMessages;
                });
              }
              
              if (data.done) {
                // Update session ID and sources
                if (data.session_id) {
                  setSessionId(data.session_id);
                }
                
                setMessages(prev => {
                  const newMessages = [...prev];
                  if (newMessages[assistantMessageIndex]) {
                    newMessages[assistantMessageIndex].isStreaming = false;
                    newMessages[assistantMessageIndex].sources = data.sources || [];
                  }
                  return newMessages;
                });
              }
            } catch (e) {
              console.error('Error parsing streaming data:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => {
        const newMessages = [...prev];
        if (newMessages[assistantMessageIndex]) {
          newMessages[assistantMessageIndex].content = 'Sorry, there was an error processing your message. Please try again.';
          newMessages[assistantMessageIndex].isStreaming = false;
        }
        return newMessages;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearConversation = async () => {
    if (sessionId) {
      try {
        await fetch(`${API_BASE_URL}/conversations/${sessionId}`, {
          method: 'DELETE'
        });
      } catch (error) {
        console.error('Error clearing conversation:', error);
      }
    }
    setMessages([]);
    setSessionId(null);
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🤖 RAG Chatbot</h1>
        <p>Upload documents and chat with your personal AI assistant</p>
      </header>

      <div className="main-container">
        {/* Sidebar */}
        <div className="sidebar">
          <div className="upload-section">
            <h3>📚 Upload Documents</h3>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".pdf,.txt,.docx"
              style={{ display: 'none' }}
            />
            <button 
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              className="upload-btn"
            >
              {isLoading ? '⏳ Uploading...' : '📁 Choose File'}
            </button>
            <p className="file-info">Supports: PDF, TXT, DOCX</p>
          </div>

          <div className="documents-section">
            <h3>📄 Uploaded Documents</h3>
            {Object.keys(documents).length === 0 ? (
              <p className="no-docs">No documents uploaded yet</p>
            ) : (
              <div className="documents-list">
                {Object.entries(documents).map(([filename, chunks]) => (
                  <div key={filename} className="document-item">
                    <span className="doc-name">{filename}</span>
                    <span className="doc-chunks">{chunks} chunks</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="session-section">
            <h3>💬 Session</h3>
            {sessionId && (
              <p className="session-id">ID: {sessionId.slice(0, 8)}...</p>
            )}
            <button onClick={clearConversation} className="clear-btn">
              🗑️ Clear Chat
            </button>
          </div>
        </div>

        {/* Chat Area */}
        <div className="chat-container">
          <div className="messages-container">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <h2>👋 Welcome to RAG Chatbot!</h2>
                <p>Upload some documents and start asking questions about them.</p>
                <div className="example-questions">
                  <h4>Try asking:</h4>
                  <ul>
                    <li>"What are the main topics in the uploaded documents?"</li>
                    <li>"Can you summarize the key points?"</li>
                    <li>"Tell me about [specific topic from your documents]"</li>
                  </ul>
                </div>
              </div>
            ) : (
              messages.map((message, index) => (
                <div key={index} className={`message ${message.type}`}>
                  <div className="message-content">
                    <div className="message-text">
                      {message.content}
                      {message.isStreaming && <span className="cursor">▋</span>}
                    </div>
                    {message.sources && message.sources.length > 0 && (
                      <div className="message-sources">
                        <strong>📚 Sources:</strong> {message.sources.join(', ')}
                      </div>
                    )}
                  </div>
                  <div className="message-time">{message.timestamp}</div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-container">
            <div className="input-wrapper">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a question about your documents..."
                disabled={isLoading || isStreaming}
                rows="3"
              />
              <button 
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isLoading || isStreaming}
                className="send-btn"
              >
                {isStreaming ? '⏳' : '🚀'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;