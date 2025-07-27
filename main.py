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
    ============================
    
    Process documents and answer questions with detailed pipeline timing.
    """
    start_time = time.time()
    pipeline_timings = {}
    
    try:
        print(f"🚀 STARTING HACKRX 6.0 PIPELINE")
        print(f"📄 Document: {request.documents}")
        print(f"❓ Questions: {len(request.questions)}")
        print("=" * 60)
        
        # Stage 1: Document Download and Processing
        doc_start = time.time()
        print(f"📥 STAGE 1: Document Download & Processing...")
        
        document_data = document_processor.process_document_from_url(request.documents)
        doc_time = time.time() - doc_start
        pipeline_timings['document_processing'] = doc_time
        
        print(f"✅ Document processed in {doc_time:.2f}s")
        print(f"   📊 Pages: {len(document_data.get('pages', []))}")
        print(f"   📝 Content length: {len(document_data.get('full_text', ''))} chars")
        
        # Stage 2: Check if document exists in database
        db_start = time.time()
        print(f"🗄️ STAGE 2: Database Check...")
        
        existing_doc = db_manager.get_document_by_url(request.documents)
        is_first_time = existing_doc is None
        db_time = time.time() - db_start
        pipeline_timings['database_check'] = db_time
        
        if is_first_time:
            print(f"🆕 FIRST TIME PROCESSING - Complete pipeline will run")
        else:
            print(f"♻️ DOCUMENT ALREADY PROCESSED - Using cached data")
        print(f"✅ Database check completed in {db_time:.3f}s")
        
        # Stage 3: Text Chunking (if first time)
        if is_first_time:
            chunk_start = time.time()
            print(f"✂️ STAGE 3: Text Chunking...")
            
            chunks = rag_engine.create_intelligent_chunks(document_data)
            chunk_time = time.time() - chunk_start
            pipeline_timings['text_chunking'] = chunk_time
            
            print(f"✅ Text chunked in {chunk_time:.2f}s")
            print(f"   📦 Total chunks: {len(chunks)}")
            print(f"   📏 Avg chunk size: {sum(len(c['content']) for c in chunks) // len(chunks)} chars")
        else:
            pipeline_timings['text_chunking'] = 0.0
            print(f"⏭️ STAGE 3: Skipping chunking (already done)")
        
        # Stage 4: Vector Embedding & Storage (if first time)  
        if is_first_time:
            embed_start = time.time()
            print(f"🧠 STAGE 4: Vector Embedding & Storage...")
            
            # This will embed and store in Pinecone
            processing_result = await rag_engine.process_and_store_document(document_data, request.documents)
            embed_time = time.time() - embed_start
            pipeline_timings['embedding_and_storage'] = embed_time
            
            print(f"✅ Embeddings created and stored in {embed_time:.2f}s")
            print(f"   🎯 Vectors stored: {processing_result.get('vectors_stored', 0)}")
        else:
            pipeline_timings['embedding_and_storage'] = 0.0
            print(f"⏭️ STAGE 4: Skipping embedding (already done)")
        
        # Stage 5: Process Questions Concurrently
        questions_start = time.time()
        print(f"🔄 STAGE 5: Processing {len(request.questions)} questions...")
        
        async def process_single_question(question: str, question_num: int) -> str:
            """Process individual question with detailed timing."""
            q_start = time.time()
            print(f"   🤔 Q{question_num}: {question[:50]}...")
            
            try:
                # Stage 5a: Context Retrieval
                retrieval_start = time.time()
                context, citations = await rag_engine.retrieve_context_for_question(
                    question, request.documents
                )
                retrieval_time = time.time() - retrieval_start
                
                print(f"   🔍 Retrieved context in {retrieval_time:.3f}s ({len(citations)} chunks)")
                
                # Format retrieval info consistently
                retrieval_info = {
                    "citations": citations,
                    "chunks_retrieved": len(citations),
                    "context_length": len(context)
                }
                
                # Stage 5b: LLM Response Generation
                llm_start = time.time()
                answer, enhanced_response = await llm_client.get_enhanced_answer(
                    question, context, document_data['metadata']
                )
                llm_time = time.time() - llm_start
                
                print(f"   🧠 LLM response in {llm_time:.3f}s")
                
                # Stage 5c: Database Logging
                log_start = time.time()
                question_time = time.time() - q_start
                db_manager.log_query(
                    document_url=request.documents,
                    question=question,
                    answer=answer,
                    context_used=context,
                    citations=json.dumps(retrieval_info.get('citations', [])),
                    response_time=question_time,
                    tokens_used=enhanced_response.get('performance_metrics', {}).get('tokens_used', 0)
                )
                log_time = time.time() - log_start
                
                print(f"   📝 Logged in {log_time:.3f}s")
                print(f"   ✅ Q{question_num} completed in {question_time:.3f}s")
                
                return answer
                
            except Exception as e:
                error_time = time.time() - q_start
                print(f"   ❌ Q{question_num} failed in {error_time:.3f}s: {e}")
                return f"Error processing question: {str(e)}"
        
        # Execute all questions concurrently
        tasks = [process_single_question(q, i+1) for i, q in enumerate(request.questions)]
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        questions_time = time.time() - questions_start
        pipeline_timings['questions_processing'] = questions_time
        
        # Handle any exceptions in results
        final_answers = []
        for answer in answers:
            if isinstance(answer, Exception):
                final_answers.append(f"Error: {str(answer)}")
            else:
                final_answers.append(answer)
        
        total_time = time.time() - start_time
        pipeline_timings['total_time'] = total_time
        
        print("=" * 60)
        print(f"🎯 PIPELINE COMPLETED!")
        print(f"⏱️ Total time: {total_time:.2f}s")
        print(f"📊 Timing breakdown:")
        for stage, timing in pipeline_timings.items():
            print(f"   {stage}: {timing:.3f}s")
        print(f"✅ Processed {len(request.questions)} questions successfully")
        print("=" * 60)
        
        return HackRxResponse(answers=final_answers)
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ PIPELINE FAILED after {error_time:.2f}s: {e}")
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


@app.post("/reset-system")
async def reset_system(authorized: bool = Depends(verify_token)):
    """Reset system by clearing all database data for fresh processing demonstration."""
    try:
        print("🧹 Starting system reset...")
        
        # Clear PostgreSQL tables
        print("🗄️ Clearing PostgreSQL tables...")
        reset_result = db_manager.reset_all_data()
        
        # Clear Pinecone vectors if index exists
        print("🔍 Clearing Pinecone vectors...")
        vector_result = rag_engine.clear_all_vectors()
        
        reset_info = {
            "status": "success",
            "message": "System reset completed successfully",
            "cleared": {
                "postgresql_records": reset_result.get("cleared_records", 0),
                "pinecone_vectors": vector_result.get("cleared_vectors", 0)
            },
            "next_request_will_show": "Complete pipeline from document download to response"
        }
        
        print("✅ System reset completed!")
        return reset_info
        
    except Exception as e:
        error_msg = f"Failed to reset system: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


# =============================================================================
# Application Lifecycle Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup with diagnostics."""
    print("🚀 Starting HackRX 6.0 Intelligent Query-Retrieval System...")
    
    # ADD CRITICAL DIAGNOSTICS
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=config.pinecone_api_key)
        index = pc.Index(config.pinecone_index_name)
        stats = index.describe_index_stats()
        print(f"📊 Pinecone Index Stats: {stats}")
        print(f"📏 Index Dimension: {stats['dimension']}")
        print(f"🎯 Current Vector Count: {stats['total_vector_count']}")
        
        # Check if dimension matches our model
        if stats['dimension'] != 384:
            print(f"⚠️ DIMENSION MISMATCH! Index expects {stats['dimension']}, model produces 384")
            print(f"💡 Run: python3 recreate_index.py to fix this")
        else:
            print("✅ Dimension matches - Index properly configured")
            
    except Exception as e:
        print(f"⚠️ Pinecone diagnostic failed: {e}")
    
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
