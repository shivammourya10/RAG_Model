#!/usr/bin/env python3
"""
Setup script for Pinecone Serverless Index
Creates and configures a Pinecone index for the HackRX system
"""

import os
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

def setup_pinecone_serverless():
    """Setup Pinecone serverless index for HackRX system."""
    
    print("🚀 Setting up Pinecone Serverless Index for HackRX 6.0")
    print("=" * 60)
    
    try:
        # Initialize Pinecone
        print("📡 Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index already exists
        existing_indexes = pc.list_indexes()
        index_names = [idx['name'] for idx in existing_indexes]
        
        if PINECONE_INDEX_NAME in index_names:
            print(f"✅ Index '{PINECONE_INDEX_NAME}' already exists!")
            
            # Test the connection
            index = pc.Index(PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            print(f"📊 Index Stats: {stats}")
            
        else:
            # Get embedding model dimensions
            print("🧠 Loading embedding model to get dimensions...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            sample_embedding = model.encode(["test"])
            dimensions = len(sample_embedding[0])
            
            print(f"📏 Embedding dimensions: {dimensions}")
            
            # Create serverless index
            print(f"🏗️  Creating serverless index '{PINECONE_INDEX_NAME}'...")
            
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=dimensions,  # 384 for all-MiniLM-L6-v2
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',  # or 'gcp', 'azure'
                    region='us-east-1'  # choose appropriate region
                )
            )
            
            # Wait for index to be ready
            print("⏳ Waiting for index to be ready...")
            while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
            
            print("✅ Index created successfully!")
            
        # Test the index
        print("🧪 Testing index connection...")
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Test with a sample vector
        test_vector = [0.1] * dimensions
        test_id = "test-vector-123"
        
        # Upsert test vector
        index.upsert([(test_id, test_vector, {"test": "true"})])
        print("✅ Test upsert successful!")
        
        # Query test
        query_result = index.query(vector=test_vector, top_k=1, include_metadata=True)
        print(f"✅ Test query successful! Found {len(query_result['matches'])} matches")
        
        # Clean up test vector
        index.delete(ids=[test_id])
        print("🧹 Test vector cleaned up")
        
        print("\n🎉 Pinecone Serverless Setup Complete!")
        print("=" * 60)
        print("✅ Your HackRX system is now ready to use vector search!")
        print(f"📝 Index Name: {PINECONE_INDEX_NAME}")
        print(f"📏 Dimensions: {dimensions}")
        print("🔄 Metric: cosine")
        print("☁️  Mode: Serverless")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up Pinecone: {e}")
        print("\n🔧 Troubleshooting Tips:")
        print("1. Check your PINECONE_API_KEY in .env file")
        print("2. Ensure you have sufficient Pinecone quota")
        print("3. Try a different region (us-west-2, eu-west1, etc.)")
        print("4. Check Pinecone dashboard for any issues")
        return False

if __name__ == "__main__":
    success = setup_pinecone_serverless()
    if success:
        print("\n🚀 You can now run your HackRX system with full vector search!")
        print("💡 Start the server: python main.py")
    else:
        print("\n⚠️  Setup failed. Please check the errors above.")
