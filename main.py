"""
HackRX 6.0 - Intelligent Query-Retrieval System
===============================================

Main FastAPI application providing LLM-powered document processing and question answering.
This system is designed for high-performance, concurrent processing with <30s latency requirement.

Key Features:
- Multi-format document processing (PDF, DOCX, Email)
- Semantic search with vector embeddings (Pinecone)
- LLM integration with Google Gemini/OpenAI
- PostgreSQL for metadata and query logging
- Bearer token authentication
- Concurrent question processing for optimal performance

Architecture:
Document URL → Document Processor → RAG Engine → Vector DB (Pinecone)
                                          ↓
Question → LLM Client (Gemini) ← Context Retrieval ← PostgreSQL (Metadata)
    ↓
Enhanced Response (Answer + Citations + Reasoning)

Author: HackRX 6.0 Team
Version: 1.0.0
"""

import asyncio
import time
import json
from typing import List, Dict, Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import custom modules
from config import config
from doc_processor import DocumentProcessor
from rag_core import RAGEngine
from llm_client import LLMClient
from database import DatabaseManager


# =============================================================================
# FastAPI Application Setup
# =============================================================================

app = FastAPI(
    title="HackRX 6.0 Intelligent Query-Retrieval System",
    description="LLM-Powered document processing with contextual Q&A",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =============================================================================
# Service Initialization (Singleton Pattern)
# =============================================================================

auth_scheme = HTTPBearer()
document_processor = DocumentProcessor()
rag_engine = RAGEngine()
llm_client = LLMClient()
db_manager = DatabaseManager()


# =============================================================================
# Data Models (Pydantic)
# =============================================================================

class HackRxRequest(BaseModel):
    """HackRX 6.0 specification compliant request model."""
    documents: str
    questions: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
                "questions": [
                    "What is the grace period for premium payment?",
                    "What are the waiting periods in this policy?"
                ]
            }
        }


class HackRxResponse(BaseModel):
    """HackRX 6.0 specification compliant response model."""
    answers: List[str]


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    message: str
    services: Dict


# =============================================================================
# Security & Authentication
# =============================================================================

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)) -> bool:
    """
    Verify bearer token for API authentication.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        bool: True if token is valid
        
    Raises:
        HTTPException: 401 if token invalid
    """
    expected_token = config.api_bearer_token
    
    if (not credentials or 
        credentials.scheme != "Bearer" or 
        credentials.credentials != expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint for basic health check."""
    return HealthResponse(
        status="healthy",
        message="HackRX 6.0 Intelligent Query-Retrieval System",
        services={
            "document_processor": "active",
            "rag_engine": "active",
            "llm_client": "active",
            "database": "active"
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check with service status."""
    try:
        # Test database connection
        stats = db_manager.get_document_stats()
        
        return HealthResponse(
            status="healthy",
            message="All services operational",
            services={
                "document_processor": "active",
                "rag_engine": "active", 
                "llm_client": "active",
                "database": f"active - {stats.get('total_processed_documents', 0)} docs",
                "vector_db": "active"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_evaluation_endpoint(
    request: HackRxRequest, 
    authorized: bool = Depends(verify_token)
):
    """
    Main HackRX 6.0 evaluation endpoint.
    
    Processes documents and answers questions with <30 second latency requirement.
    Uses concurrent processing for optimal performance.
    
    Args:
        request: HackRX request containing document URL and questions
        authorized: Token verification result
        
    Returns:
        HackRxResponse: List of answers corresponding to questions
    """
    start_time = time.time()
    
    try:
        # Step 1: Process document
        print(f"📄 Processing document: {request.documents}")
        document_data = document_processor.process_document_from_url(request.documents)
        
        # Step 2: Store in vector database
        processing_result = await rag_engine.process_and_store_document(
            document_data, request.documents
        )
        
        # Step 3: Process all questions concurrently
        async def process_single_question(question: str) -> str:
            """Process individual question with context retrieval and LLM generation."""
            try:
                # Retrieve relevant context
                context, citations = await rag_engine.retrieve_context_for_question(
                    question, request.documents
                )
                
                # Format retrieval info consistently
                retrieval_info = {
                    "citations": citations,
                    "chunks_retrieved": len(citations),
                    "context_length": len(context)
                }
                
                # Generate answer using LLM
                answer, enhanced_response = await llm_client.get_enhanced_answer(
                    question, context, document_data['metadata']
                )
                
                # Log query for analytics
                question_time = time.time() - start_time
                db_manager.log_query(
                    document_url=request.documents,
                    question=question,
                    answer=answer,
                    context_used=context,
                    citations=json.dumps(retrieval_info.get('citations', [])),
                    response_time=question_time,
                    tokens_used=enhanced_response.get('performance_metrics', {}).get('tokens_used', 0)
                )
                
                return answer
                
            except Exception as e:
                print(f"❌ Error processing question '{question}': {e}")
                return f"Error processing question: {str(e)}"
        
        # Execute all questions concurrently for optimal latency
        print(f"🔄 Processing {len(request.questions)} questions concurrently...")
        tasks = [process_single_question(q) for q in request.questions]
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        final_answers = []
        for answer in answers:
            if isinstance(answer, Exception):
                final_answers.append(f"Error: {str(answer)}")
            else:
                final_answers.append(answer)
        
        total_time = time.time() - start_time
        print(f"✅ Processed {len(request.questions)} questions in {total_time:.2f}s")
        
        return HackRxResponse(answers=final_answers)
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Error processing request: {e} (took {error_time:.2f}s)")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process request: {str(e)}"
        )


@app.get("/stats")
async def get_system_stats(authorized: bool = Depends(verify_token)):
    """
    Get system statistics and performance metrics.
    
    Returns:
        dict: System statistics including database stats and configuration
    """
    try:
        db_stats = db_manager.get_document_stats()
        return {
            "database_stats": db_stats,
            "system_config": {
                "llm_provider": config.llm_provider,
                "chunk_size": config.chunk_size,
                "top_k_retrieval": config.top_k_retrieval,
                "max_context_length": config.max_context_length
            },
            "supported_formats": config.supported_formats
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


# =============================================================================
# Application Lifecycle Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    print("🚀 Starting HackRX 6.0 Intelligent Query-Retrieval System...")
    print(f"🤖 LLM Provider: {config.llm_provider}")
    print(f"🗄️ Vector DB: Pinecone ({config.pinecone_index_name})")
    print(f"💾 Database: PostgreSQL")
    print("✅ System ready to process requests!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    print("🛑 Shutting down HackRX system...")


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors with helpful message."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found", 
            "message": "Use POST /hackrx/run for document processing",
            "available_endpoints": ["/", "/health", "/hackrx/run", "/stats", "/docs"]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors gracefully."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error", 
            "message": "Please check your request format and try again"
        }
    )


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
