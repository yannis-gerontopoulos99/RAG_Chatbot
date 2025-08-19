"""
RAG Chatbot Backend - Working Version
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from dotenv import load_dotenv
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A simple RAG-powered chatbot",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for RAG pipeline
rag_pipeline = None

class ChatQuery(BaseModel):
    question: str
    conversation_history: Optional[List[Dict[str, str]]] = []
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    #sources: List[Dict[str, Any]]  # COMMENTED OUT - sources disabled
    #confidence_score: float # COMMENTED OUT - confidence disabled
    session_id: Optional[str] = None
    timestamp: str = datetime.now().isoformat()

def initialize_pipeline():
    """Initialize the RAG pipeline"""
    global rag_pipeline
    try:
        logger.info("Initializing RAG pipeline...")
        
        # Try to import the RAG pipeline
        try:
            from rag_pipeline import RAGPipeline
        except ImportError as e:
            logger.error(f"Cannot import RAGPipeline: {e}")
            return False  # exit if import fails
        
        # Only instantiate if import succeeds
        rag_pipeline = RAGPipeline()
        rag_pipeline.initialize()
        logger.info("RAG pipeline initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        return False

@app.get("/")
def root():
    return {
        "message": "RAG Chatbot API", 
        "status": "running",
        "docs": "Visit /docs for API documentation",
        "health": "Visit /health for health check"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "pipeline_ready": rag_pipeline is not None,
        "timestamp": datetime.now().isoformat(),
        "message": "Server is running!"
    }

@app.post("/chat", response_model=ChatResponse)
def chat(query: ChatQuery):
    """Process a chat query"""
    if not rag_pipeline:
        # Try to initialize if not already done
        if not initialize_pipeline():
            raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        logger.info(f"Processing query: {query.question[:100]}...")
        
        result = rag_pipeline.process_query(
            question=query.question,
            conversation_history=query.conversation_history
        )
        
        return ChatResponse(
            answer=result["answer"],
            #sources=result["sources"],  # COMMENTED OUT - sources disabled
            #confidence_score=result["confidence_score"], # COMMENTED OUT - confidence disabled
            session_id=query.session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        # Return a friendly error response instead of crashing
        return ChatResponse(
            answer=f"I apologize, but I encountered an error processing your question. Error: {str(e)}",
            #sources=[],  # COMMENTED OUT - sources disabled
            #confidence_score=0.0, # COMMENTED OUT - confidence disabled
            session_id=query.session_id
        )

@app.get("/documents")
def list_documents():
    """List available documents"""
    if not rag_pipeline:
        initialize_pipeline()
    
    try:
        return {"documents": rag_pipeline.get_document_metadata()}
    except Exception as e:
        return {"documents": [], "error": str(e)}

@app.get("/test")
def test_endpoint():
    """Simple test endpoint"""
    return {
        "message": "Test endpoint is working!",
        "timestamp": datetime.now().isoformat(),
        "pipeline_status": "Ready" if rag_pipeline else "Not initialized"
    }

# Initialize pipeline when the module is imported
logger.info("Starting server...")
initialize_pipeline()

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("🚀 Starting RAG Chatbot Backend Server")
    print("=" * 50)
    print("📊 API Documentation: http://localhost:8000/docs")
    print("💚 Health Check: http://localhost:8000/health")
    print("🧪 Test Endpoint: http://localhost:8000/test")
    print("=" * 50)
    
    # Use simple uvicorn run without reload to avoid warnings
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )