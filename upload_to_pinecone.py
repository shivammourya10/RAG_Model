#!/usr/bin/env python3
"""
Upload existing document chunks to Pinecone vector database
This script takes documents already processed in PostgreSQL and uploads them to Pinecone
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from rag_core import RAGEngine
from database import SessionLocal, ProcessedDocument, DocumentChunk
import time

async def upload_documents_to_pinecone():
    """Upload all processed documents to Pinecone vector database"""
    
    print("🚀 HackRX 6.0 - Document Upload to Pinecone")
    print("=" * 60)
    
    try:
        # Initialize RAG engine
        rag_engine = RAGEngine()
        
        if not rag_engine.pinecone_index:
            print("❌ Pinecone index not available")
            return False
        
        # Get all processed documents from database
        db = SessionLocal()
        try:
            documents = db.query(ProcessedDocument).filter(
                ProcessedDocument.is_active == True
            ).all()
            
            print(f"📄 Found {len(documents)} processed documents")
            
            total_chunks_uploaded = 0
            
            for doc in documents:
                print(f"\n🔄 Processing: {doc.url}")
                print(f"   Document Type: {doc.document_type}")
                print(f"   Total Chunks: {doc.total_chunks}")
                
                # Get all chunks for this document
                chunks = db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == doc.id
                ).all()
                
                if not chunks:
                    print("   ⚠️ No chunks found, skipping")
                    continue
                
                # Prepare vectors for batch upload
                vectors_to_upload = []
                
                print(f"   🧠 Generating embeddings for {len(chunks)} chunks...")
                
                for chunk in chunks:
                    try:
                        # Generate embedding for chunk content
                        embedding = rag_engine.embedder.encode([chunk.content])[0]
                        
                        # Create vector ID
                        vector_id = f"{doc.id}_{chunk.chunk_index}"
                        
                        # Prepare metadata
                        metadata = {
                            "content": chunk.content,
                            "document_url": doc.url,
                            "document_type": doc.document_type,
                            "chunk_index": chunk.chunk_index,
                            "document_id": str(doc.id)
                        }
                        
                        if chunk.page_number:
                            metadata["page_number"] = chunk.page_number
                        
                        if chunk.section_title:
                            metadata["section_title"] = chunk.section_title
                        
                        # Add to batch
                        vectors_to_upload.append((
                            vector_id,
                            embedding.tolist(),
                            metadata
                        ))
                        
                    except Exception as e:
                        print(f"   ❌ Error processing chunk {chunk.chunk_index}: {e}")
                        continue
                
                # Upload in batches of 100 (Pinecone limit)
                batch_size = 100
                chunks_uploaded = 0
                
                for i in range(0, len(vectors_to_upload), batch_size):
                    batch = vectors_to_upload[i:i + batch_size]
                    
                    try:
                        rag_engine.pinecone_index.upsert(batch)
                        chunks_uploaded += len(batch)
                        print(f"   ✅ Uploaded batch {i//batch_size + 1}: {len(batch)} vectors")
                        
                        # Small delay to avoid rate limits
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"   ❌ Error uploading batch {i//batch_size + 1}: {e}")
                
                total_chunks_uploaded += chunks_uploaded
                print(f"   🎉 Document complete: {chunks_uploaded}/{len(chunks)} chunks uploaded")
            
            print(f"\n" + "=" * 60)
            print("🎉 Upload Complete!")
            print(f"📊 Total chunks uploaded: {total_chunks_uploaded}")
            
            # Verify upload
            stats = rag_engine.pinecone_index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            print(f"🔍 Pinecone index now contains: {vector_count} vectors")
            
            if vector_count > 0:
                print("✅ Vector search is now fully operational!")
                return True
            else:
                print("❌ No vectors found in index after upload")
                return False
                
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(upload_documents_to_pinecone())
    
    if success:
        print("\n🚀 Your HackRX system is now ready with full vector search!")
        print("💡 Test with: python3 demo_vector_search.py")
    else:
        print("\n⚠️ Upload had issues. Check the logs above.")
