# database.py
import asyncio
from sqlalchemy import create_engine, Column, String, Text, DateTime, Boolean, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from config import DATABASE_URL

Base = declarative_base()

class ProcessedDocument(Base):
    __tablename__ = "processed_documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(String, unique=True, nullable=False, index=True)
    document_type = Column(String, nullable=False)  # pdf, docx, email
    content_hash = Column(String, nullable=False)
    total_chunks = Column(Integer, nullable=False)
    processing_time = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    section_title = Column(String, nullable=True)
    chunk_metadata = Column(Text, nullable=True)  # JSON string for additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)

class QueryLog(Base):
    __tablename__ = "query_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_url = Column(String, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    context_used = Column(Text, nullable=False)
    citations = Column(Text, nullable=True)  # JSON string
    response_time = Column(Float, nullable=False)
    tokens_used = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database connection setup
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class DatabaseManager:
    """Manages database operations for document processing and query logging."""
    
    @staticmethod
    def is_document_processed(url: str) -> bool:
        """Check if a document has already been processed."""
        db = SessionLocal()
        try:
            doc = db.query(ProcessedDocument).filter(
                ProcessedDocument.url == url,
                ProcessedDocument.is_active == True
            ).first()
            return doc is not None
        finally:
            db.close()
    
    @staticmethod
    def save_processed_document(url: str, doc_type: str, content_hash: str, 
                              total_chunks: int, processing_time: float) -> str:
        """Save processed document metadata."""
        db = SessionLocal()
        try:
            doc = ProcessedDocument(
                url=url,
                document_type=doc_type,
                content_hash=content_hash,
                total_chunks=total_chunks,
                processing_time=processing_time
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            return str(doc.id)
        finally:
            db.close()
    
    @staticmethod
    def save_document_chunks(document_id: str, chunks_data: list):
        """Save document chunks to database."""
        db = SessionLocal()
        try:
            chunks = []
            for i, chunk_info in enumerate(chunks_data):
                chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_index=i,
                    content=chunk_info['content'],
                    page_number=chunk_info.get('page_number'),
                    section_title=chunk_info.get('section_title'),
                    chunk_metadata=chunk_info.get('metadata')
                )
                chunks.append(chunk)
            
            db.add_all(chunks)
            db.commit()
        finally:
            db.close()
    
    @staticmethod
    def log_query(document_url: str, question: str, answer: str, 
                  context_used: str, citations: str, response_time: float, 
                  tokens_used: int = None):
        """Log query and response for analytics."""
        db = SessionLocal()
        try:
            query_log = QueryLog(
                document_url=document_url,
                question=question,
                answer=answer,
                context_used=context_used,
                citations=citations,
                response_time=response_time,
                tokens_used=tokens_used
            )
            db.add(query_log)
            db.commit()
        finally:
            db.close()
    
    @staticmethod
    def get_document_stats():
        """Get processing statistics."""
        db = SessionLocal()
        try:
            total_docs = db.query(ProcessedDocument).filter(
                ProcessedDocument.is_active == True
            ).count()
            
            total_queries = db.query(QueryLog).count()
            
            return {
                "total_processed_documents": total_docs,
                "total_queries_handled": total_queries
            }
        finally:
            db.close()

# Initialize database on import
try:
    create_tables()
    print("Database tables created successfully")
except Exception as e:
    print(f"Database initialization error: {e}")
