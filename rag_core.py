# rag_core.py
import asyncio
import time
import hashlib
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple, Optional
import json
import numpy as np

# Import our modules
from llm_client import LLMClient
from doc_processor import DocumentProcessor
from database import DatabaseManager

from config import (
    PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL, MAX_CONTEXT_LENGTH
)
from database import DatabaseManager

class RAGEngine:
    """Advanced RAG engine with semantic search, chunk optimization, and explainable retrieval."""
    
    def __init__(self):
        self.embedder = None
        self.text_splitter = None
        self.pinecone_index = None
        self.db_manager = DatabaseManager()
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all required services with fallbacks for testing"""
        try:
            # Initialize embeddings model
            print("Loading sentence transformer model...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embeddings model loaded")
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            print("✅ Text splitter initialized")
            
            # Try to initialize Pinecone with new serverless API
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=PINECONE_API_KEY)
                
                # Check if index exists
                existing_indexes = pc.list_indexes()
                index_names = [idx['name'] for idx in existing_indexes]
                
                if PINECONE_INDEX_NAME not in index_names:
                    print(f"⚠️  Pinecone index '{PINECONE_INDEX_NAME}' not found.")
                    print("💡 Run 'python setup_pinecone_serverless.py' to create the index.")
                    self.pinecone_index = None
                else:
                    self.pinecone_index = pc.Index(PINECONE_INDEX_NAME)
                    print("✅ Pinecone serverless index connected")
            except Exception as pinecone_error:
                print(f"⚠️  Pinecone initialization failed: {pinecone_error}")
                print("💡 System will run without vector search functionality.")
                self.pinecone_index = None
            
            # Initialize LLM client
            self.llm_client = LLMClient()
            print("✅ LLM client initialized")
            
            # Initialize document processor
            self.doc_processor = DocumentProcessor()
            print("✅ Document processor initialized")
            
            # Initialize database
            self.db_manager = DatabaseManager()
            print("✅ Database connected")
            
        except Exception as e:
            raise Exception(f"Failed to initialize RAG Engine: {e}")
    
    def create_intelligent_chunks(self, document_data: Dict) -> List[Dict]:
        """Create intelligent chunks with metadata preservation."""
        try:
            full_text = document_data['full_text']
            doc_metadata = document_data['metadata']
            doc_type = doc_metadata['document_type']
            
            # Base chunking
            base_chunks = self.text_splitter.split_text(full_text)
            
            enhanced_chunks = []
            
            for i, chunk in enumerate(base_chunks):
                chunk_data = {
                    'content': chunk,
                    'chunk_index': i,
                    'metadata': {
                        'document_type': doc_type,
                        'source_url': doc_metadata.get('source_url', ''),
                        'content_hash': doc_metadata.get('content_hash', ''),
                        'chunk_id': f"{doc_metadata.get('content_hash', 'unknown')}_{i}"
                    }
                }
                
                # Add type-specific metadata
                if doc_type == 'pdf' and 'pages' in document_data:
                    # Try to determine which page this chunk belongs to
                    chunk_data['metadata']['page_number'] = self._find_page_for_chunk(
                        chunk, document_data['pages']
                    )
                
                elif doc_type == 'docx' and 'paragraphs' in document_data:
                    # Try to find paragraph information
                    chunk_data['metadata']['paragraph_info'] = self._find_paragraph_for_chunk(
                        chunk, document_data['paragraphs']
                    )
                
                elif doc_type == 'email':
                    if 'email_data' in document_data:
                        chunk_data['metadata']['email_metadata'] = {
                            'sender': document_data['email_data'].get('sender', ''),
                            'subject': document_data['email_data'].get('subject', ''),
                            'date': document_data['email_data'].get('date', '')
                        }
                
                enhanced_chunks.append(chunk_data)
            
            return enhanced_chunks
            
        except Exception as e:
            raise Exception(f"Error creating chunks: {e}")
    
    def _find_page_for_chunk(self, chunk: str, pages: List[Dict]) -> Optional[int]:
        """Find which page a chunk likely belongs to."""
        # Simple heuristic: find page with highest text overlap
        best_match = None
        best_score = 0
        
        for page in pages:
            page_content = page['content']
            # Calculate simple overlap score
            chunk_words = set(chunk.lower().split())
            page_words = set(page_content.lower().split())
            
            if len(chunk_words) > 0:
                overlap = len(chunk_words.intersection(page_words))
                score = overlap / len(chunk_words)
                
                if score > best_score:
                    best_score = score
                    best_match = page['page_number']
        
        return best_match if best_score > 0.3 else None
    
    def _find_paragraph_for_chunk(self, chunk: str, paragraphs: List[Dict]) -> Optional[Dict]:
        """Find which paragraph a chunk likely belongs to."""
        best_match = None
        best_score = 0
        
        for para in paragraphs:
            para_content = para['content']
            if chunk.lower() in para_content.lower() or para_content.lower() in chunk.lower():
                # Calculate overlap score
                chunk_words = set(chunk.lower().split())
                para_words = set(para_content.lower().split())
                
                if len(chunk_words) > 0:
                    overlap = len(chunk_words.intersection(para_words))
                    score = overlap / len(chunk_words)
                    
                    if score > best_score:
                        best_score = score
                        best_match = {
                            'paragraph_number': para['paragraph_number'],
                            'style': para.get('style')
                        }
        
        return best_match if best_score > 0.3 else None
    
    async def process_and_store_document(self, document_data: Dict, document_url: str) -> Dict:
        """Process document and store in vector database with database tracking."""
        start_time = time.time()
        
        try:
            doc_metadata = document_data['metadata']
            content_hash = doc_metadata['content_hash']
            doc_type = doc_metadata['document_type']
            
            # Check if already processed (both in-memory and database)
            if self.db_manager.is_document_processed(document_url):
                print(f"Document {document_url} already processed in database. Skipping.")
                return {
                    "status": "skipped",
                    "reason": "already_processed",
                    "source": "database"
                }
            
            print(f"Processing new document: {document_url}")
            
            # Create intelligent chunks
            chunks_data = self.create_intelligent_chunks(document_data)
            
            # Initialize vector count tracker
            vectors_stored = 0
            
            # Store in Pinecone if available
            if self.pinecone_index:
                print(f"🚀 SPEED-OPTIMIZED embedding generation for {len(chunks_data)} chunks...")
                embed_start = time.time()
                
                # ULTRA-FAST embedding generation with maximum batch size
                chunk_contents = [chunk['content'] for chunk in chunks_data]
                embeddings = self.embedder.encode(
                    chunk_contents, 
                    batch_size=128,  # Increased from 64 to 128 for maximum speed
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    device='cpu'  # Ensure consistent CPU processing for stability
                )
                embed_time = time.time() - embed_start
                print(f"✅ Embeddings generated in {embed_time:.2f}s")
                
                # OPTIMIZATION 2: Streamlined vector preparation (no extra loops)
                prep_start = time.time()
                vectors_to_upsert = [
                    {
                        "id": f"{content_hash}_{i}",
                        "values": embedding.tolist(),
                        "metadata": {
                            "content": chunk_data['content'][:300],  # Reduced from 500 to 300
                            "chunk_index": i,
                            "document_url": document_url,
                            "content_hash": content_hash,
                            "page_number": chunk_data['metadata'].get('page_number', 1)
                        }
                    }
                    for i, (chunk_data, embedding) in enumerate(zip(chunks_data, embeddings))
                ]
                prep_time = time.time() - prep_start
                print(f"⚡ Vector preparation in {prep_time:.2f}s")
                
                # OPTIMIZATION 3: Larger batch size for Pinecone (less network calls)
                batch_size = 150  # Increased from 100 to 150
                total_batches = (len(vectors_to_upsert) + batch_size - 1) // batch_size
                print(f"📤 FAST upserting {len(vectors_to_upsert)} vectors in {total_batches} batches...")
                
                upsert_start = time.time()
                successful_upserts = 0
                try:
                    for i in range(0, len(vectors_to_upsert), batch_size):
                        batch = vectors_to_upsert[i:i + batch_size]
                        batch_num = (i // batch_size) + 1
                        
                        # Upsert with error handling (no extra logging in loop for speed)
                        upsert_response = self.pinecone_index.upsert(vectors=batch)
                        successful_upserts += len(batch)
                        print(f"   📦 Batch {batch_num}/{total_batches} uploaded ({len(batch)} vectors)")
                    
                    upsert_time = time.time() - upsert_start
                    print(f"✅ All batches processed in {upsert_time:.2f}s - {successful_upserts} vectors uploaded")
                    print(f"⚡ SPEED MODE: Skipping verification (vectors propagate in background)")
                    
                    vectors_stored = successful_upserts
                    
                except Exception as upsert_error:
                    print(f"💥 UPSERT ERROR: {upsert_error}")
                    raise Exception(f"Failed to store vectors: {upsert_error}")
                
                total_storage_time = time.time() - embed_start
                print(f"🎯 TOTAL STORAGE TIME: {total_storage_time:.2f}s")
            else:
                print("⚠️  Pinecone not available - document stored in database only")
            
            processing_time = time.time() - start_time
            
            # Save to database
            doc_id = self.db_manager.save_processed_document(
                url=document_url,
                doc_type=doc_type,
                content_hash=content_hash,
                total_chunks=len(chunks_data),
                processing_time=processing_time
            )
            
            # Save chunks to database
            chunks_for_db = [
                {
                    'content': chunk['content'],
                    'page_number': chunk['metadata'].get('page_number'),
                    'section_title': chunk['metadata'].get('section_title'),
                    'metadata': json.dumps(chunk['metadata'])
                }
                for chunk in chunks_data
            ]
            
            self.db_manager.save_document_chunks(doc_id, chunks_for_db)
            
            print(f"Successfully processed document: {document_url} in {processing_time:.2f}s")
            
            return {
                "status": "processed",
                "chunks_created": len(chunks_data),
                "processing_time": processing_time,
                "document_id": doc_id,
                "vectors_stored": vectors_stored  # CRITICAL FIX: Return actual vector count
            }
            
        except Exception as e:
            raise Exception(f"Error processing document: {e}")
    
    async def retrieve_context_for_question(self, question: str, document_url: str = "") -> Tuple[str, List[Dict]]:
        """Retrieve relevant context with detailed citation information."""
        try:
            # Check if Pinecone is available
            if not self.pinecone_index:
                print("⚠️  Pinecone not available. Falling back to database search.")
                return await self._fallback_retrieval(question, document_url)
            
            # Generate question embedding
            question_embedding = self.embedder.encode([question])[0]
            
            # Build query filters
            query_filter = {}
            if document_url:
                query_filter["document_url"] = document_url
            
            # Query Pinecone with SPEED OPTIMIZATIONS
            query_results = self.pinecone_index.query(
                vector=question_embedding.tolist(),
                top_k=2,  # REDUCED from 3 to 2 for faster processing
                include_metadata=True,
                filter=query_filter if query_filter else None
            )
            
            # Process results with SPEED-FIRST approach
            context_chunks = []
            citations = []
            
            for match in query_results['matches']:
                metadata = match['metadata']
                
                # SPEED OPTIMIZATION: Limit chunk size to prevent slow LLM responses
                chunk_content = metadata['content']
                if len(chunk_content) > 500:  # Limit chunk size for speed
                    chunk_content = chunk_content[:500] + "..."
                
                context_chunks.append(chunk_content)
                
                # Create detailed citation
                citation = {
                    "text": chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content,
                    "source": metadata.get('document_url', 'Unknown'),
                    "chunk_index": metadata.get('chunk_index', 0),
                    "similarity_score": float(match['score']),
                    "document_type": metadata.get('document_type', 'unknown')
                }
                
                # Add type-specific citation info
                if metadata.get('page_number'):
                    citation['page_number'] = metadata['page_number']
                
                if metadata.get('email_metadata'):
                    citation['email_info'] = metadata['email_metadata']
                
                citations.append(citation)
            
            # SPEED OPTIMIZATION: Aggressive context length limits
            combined_context = self._optimize_context_for_speed(context_chunks)
            
            return combined_context, citations
            
        except Exception as e:
            raise Exception(f"Error retrieving context: {e}")
    
    def _optimize_context_length(self, context_chunks: List[str]) -> str:
        """Optimize context length to stay within limits."""
        combined = "\n\n".join(context_chunks)
        
        if len(combined) <= MAX_CONTEXT_LENGTH:
            return combined
        
        # Truncate chunks from the end until we fit
        optimized_chunks = []
        current_length = 0
        
        for chunk in context_chunks:
            chunk_length = len(chunk) + 2  # +2 for "\n\n"
            if current_length + chunk_length > MAX_CONTEXT_LENGTH:
                break
            optimized_chunks.append(chunk)
            current_length += chunk_length
        
        result = "\n\n".join(optimized_chunks)
        if len(result) < len(combined):
            result += "\n\n[Additional context truncated due to length limits...]"
        
        return result
    
    def _optimize_context_for_speed(self, context_chunks: List[str]) -> str:
        """SPEED-FIRST context optimization for sub-30s response times."""
        # AGGRESSIVE limits for maximum speed
        MAX_SPEED_CONTEXT = 1000  # Much smaller than normal for speed
        
        if not context_chunks:
            return ""
        
        # Take only the MOST relevant chunk for maximum speed
        if len(context_chunks) > 1:
            # Keep only the first (most relevant) chunk
            context_chunks = context_chunks[:1]
        
        combined = "\n\n".join(context_chunks)
        
        # Aggressive truncation for speed
        if len(combined) > MAX_SPEED_CONTEXT:
            combined = combined[:MAX_SPEED_CONTEXT] + "\n\n[TRUNCATED FOR SPEED OPTIMIZATION]"
        
        return combined
    
    async def _fallback_retrieval(self, question: str, document_url: str = "") -> Tuple[str, List[Dict]]:
        """Fallback retrieval method when Pinecone is not available."""
        try:
            from database import SessionLocal, DocumentChunk, ProcessedDocument
            import re
            
            # Search database for relevant content
            db = SessionLocal()
            try:
                # Get document chunks for the specified URL
                if document_url:
                    # First get the document ID
                    doc = db.query(ProcessedDocument).filter(
                        ProcessedDocument.url == document_url,
                        ProcessedDocument.is_active == True
                    ).first()
                    
                    if doc:
                        # Extract keywords from question for basic search
                        keywords = self._extract_keywords(question)
                        
                        # Get relevant chunks for this document with limited scope for performance
                        all_chunks = db.query(DocumentChunk).filter(
                            DocumentChunk.document_id == doc.id
                        ).limit(50).all()  # Limit to 50 chunks for performance
                        
                        if all_chunks:
                            # Score chunks based on keyword matches
                            scored_chunks = []
                            for chunk in all_chunks:
                                score = self._calculate_text_similarity(question, chunk.content, keywords)
                                scored_chunks.append((chunk, score))
                            
                            # Sort by score and take top 3 chunks for faster processing
                            scored_chunks.sort(key=lambda x: x[1], reverse=True)
                            top_chunks = scored_chunks[:3]
                            
                            if top_chunks and top_chunks[0][1] > 0:
                                # Use best matching chunks
                                context_parts = []
                                citations = []
                                
                                for i, (chunk, score) in enumerate(top_chunks):
                                    context_parts.append(chunk.content)
                                    
                                    citation = {
                                        "text": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                                        "source": document_url,
                                        "chunk_index": chunk.chunk_index,
                                        "similarity_score": score,
                                        "document_type": "pdf",
                                        "page_number": chunk.page_number
                                    }
                                    citations.append(citation)
                                
                                combined_context = "\n\n".join(context_parts)
                                return combined_context, citations
                            else:
                                # No good matches, use first few chunks
                                chunks = all_chunks[:3]
                                context_parts = []
                                citations = []
                                
                                for chunk in chunks:
                                    context_parts.append(chunk.content)
                                    
                                    citation = {
                                        "text": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                                        "source": document_url,
                                        "chunk_index": chunk.chunk_index,
                                        "similarity_score": 0.5,
                                        "document_type": "pdf",
                                        "page_number": chunk.page_number
                                    }
                                    citations.append(citation)
                                
                                combined_context = "\n\n".join(context_parts)
                                return combined_context, citations
                
                # If no specific document or document not found, return fallback message
                fallback_context = """
                This is a fallback response when vector search is not available.
                The system is configured to work with Pinecone for semantic search,
                but currently running in fallback mode.
                
                To enable full functionality:
                1. Set up a Pinecone account and get API keys
                2. Create an index using: python setup_pinecone.py
                3. Upload documents for semantic search
                """
                
                fallback_citations = [{
                    "text": "System fallback message",
                    "source": "system",
                    "chunk_index": 0,
                    "similarity_score": 0.0,
                    "document_type": "system_message"
                }]
                
                return fallback_context, fallback_citations
                
            finally:
                db.close()
            
        except Exception as e:
            raise Exception(f"Fallback retrieval failed: {e}")
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract important keywords from question for basic text search."""
        import re
        
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'when', 'where', 'why', 'how', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'do', 'does', 'did', 'have', 'has', 'had', 'been', 'being', 'am', 'are', 'was', 'were'}
        
        # Extract words, convert to lowercase, remove punctuation
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _calculate_text_similarity(self, question: str, text: str, keywords: List[str]) -> float:
        """Calculate basic text similarity based on keyword matching."""
        text_lower = text.lower()
        question_lower = question.lower()
        
        score = 0.0
        
        # Direct phrase matching (highest weight)
        question_phrases = question_lower.split()
        for i in range(len(question_phrases) - 1):
            phrase = ' '.join(question_phrases[i:i+2])
            if phrase in text_lower:
                score += 0.3
        
        # Keyword matching
        for keyword in keywords:
            if keyword in text_lower:
                score += 0.1
        
        # Bonus for insurance-specific terms
        insurance_terms = ['accident', 'coverage', 'benefit', 'claim', 'policy', 'insured', 'compensation', 'premium', 'deductible', 'endorsement', 'baggage', 'medical', 'treatment', 'injury', 'finger', 'thumb', 'hearing', 'loss', 'plastic surgery', 'phalanx', 'toe']
        for term in insurance_terms:
            if term in question_lower and term in text_lower:
                score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def get_semantic_answer(self, question: str, document_url: str, document_data: Dict) -> Tuple[str, Dict]:
        """Main method to get semantic answer with full context and citations."""
        try:
            # Process document if needed
            processing_result = await self.process_and_store_document(document_data, document_url)
            
            # Retrieve relevant context
            context, citations = await self.retrieve_context_for_question(question, document_url)
            
            return context, {
                "citations": citations,
                "processing_info": processing_result,
                "retrieval_metadata": {
                    "chunks_retrieved": len(citations),
                    "context_length": len(context)
                }
            }
            
        except Exception as e:
            raise Exception(f"Error in semantic answer generation: {e}")
    
    def clear_all_vectors(self):
        """Clear all vectors from Pinecone index for fresh demonstration."""
        try:
            if self.pinecone_index is None:
                print("⚠️ Pinecone index not available")
                return {"cleared_vectors": 0, "status": "no_index"}
            
            # Get all vector IDs in the index
            stats = self.pinecone_index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            if total_vectors == 0:
                print("📭 No vectors to clear")
                return {"cleared_vectors": 0, "status": "already_empty"}
            
            # Delete all vectors
            self.pinecone_index.delete(delete_all=True)
            print(f"🗑️ Cleared {total_vectors} vectors from Pinecone")
            
            return {
                "cleared_vectors": total_vectors,
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Error clearing Pinecone vectors: {e}")
            return {
                "cleared_vectors": 0,
                "status": "error",
                "error": str(e)
            }

# Backward compatibility functions
processed_docs_cache = set()

async def process_and_store_document(doc_text: str, doc_url: str):
    """Backward compatibility function."""
    if doc_url in processed_docs_cache:
        print(f"Document {doc_url} already processed. Skipping.")
        return
    
    # Create a simple document data structure
    document_data = {
        'full_text': doc_text,
        'metadata': {
            'document_type': 'pdf',  # Default assumption
            'source_url': doc_url,
            'content_hash': hashlib.sha256(doc_text.encode()).hexdigest()
        }
    }
    
    rag_engine = RAGEngine()
    await rag_engine.process_and_store_document(document_data, doc_url)
    processed_docs_cache.add(doc_url)

async def retrieve_context_for_question(question: str) -> str:
    """Backward compatibility function."""
    rag_engine = RAGEngine()
    context, _ = await rag_engine.retrieve_context_for_question(question)
    return context

# Example usage
if __name__ == "__main__":
    async def test_rag():
        rag_engine = RAGEngine()
        
        # Sample document data
        sample_doc_data = {
            'full_text': "This is a sample insurance policy document with various clauses and conditions.",
            'metadata': {
                'document_type': 'pdf',
                'source_url': 'https://example.com/policy.pdf',
                'content_hash': 'sample_hash_123'
            }
        }
        
        try:
            # Process document
            result = await rag_engine.process_and_store_document(
                sample_doc_data, 
                'https://example.com/policy.pdf'
            )
            print(f"Processing result: {result}")
            
            # Retrieve context
            context, citations = await rag_engine.retrieve_context_for_question(
                "What are the conditions?",
                'https://example.com/policy.pdf'
            )
            print(f"Retrieved context: {context}")
            print(f"Citations: {citations}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Run test
    asyncio.run(test_rag())
