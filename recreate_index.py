#!/usr/bin/env python3
"""
Recreate Pinecone Index with Correct Dimensions
This fixes the critical vector storage issue causing performance problems.
"""

from pinecone import Pinecone, ServerlessSpec
import time
from config import config

def recreate_pinecone_index():
    """Recreate Pinecone index with correct dimensions for all-MiniLM-L6-v2 model."""
    
    try:
        print("🔧 FIXING CRITICAL PINECONE DIMENSION MISMATCH")
        print("=" * 60)
        
        # Initialize Pinecone
        pc = Pinecone(api_key=config.pinecone_api_key)
        
        # Check current index
        try:
            existing_indexes = pc.list_indexes()
            index_names = [idx['name'] for idx in existing_indexes]
            
            if config.pinecone_index_name in index_names:
                print(f"📋 Current index '{config.pinecone_index_name}' found")
                
                # Get current stats
                index = pc.Index(config.pinecone_index_name)
                stats = index.describe_index_stats()
                print(f"📊 Current dimension: {stats['dimension']}")
                print(f"🎯 Current vector count: {stats['total_vector_count']}")
                
                if stats['dimension'] == 384:
                    print("✅ Index already has correct dimension (384)")
                    return True
                
                print(f"❌ DIMENSION MISMATCH: Index has {stats['dimension']}, need 384")
                print("🗑️ Deleting incorrect index...")
                pc.delete_index(config.pinecone_index_name)
                
                # Wait for deletion
                print("⏳ Waiting for index deletion...")
                time.sleep(10)
                
            else:
                print(f"📋 Index '{config.pinecone_index_name}' not found")
        
        except Exception as e:
            print(f"⚠️ Error checking existing index: {e}")
        
        # Create new index with correct dimensions
        print("🏗️ Creating new index with dimension=384...")
        pc.create_index(
            name=config.pinecone_index_name,
            dimension=384,  # CRITICAL: Must match all-MiniLM-L6-v2 output
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            )
        )
        
        print("⏳ Waiting for index to initialize...")
        time.sleep(15)
        
        # Verify new index
        index = pc.Index(config.pinecone_index_name)
        stats = index.describe_index_stats()
        
        print("=" * 60)
        print("✅ INDEX RECREATION COMPLETED!")
        print(f"📏 New dimension: {stats['dimension']}")
        print(f"🎯 Vector count: {stats['total_vector_count']}")
        print(f"📊 Status: {stats.get('index_fullness', 'Ready')}")
        
        if stats['dimension'] == 384:
            print("🎉 SUCCESS: Index now properly configured for all-MiniLM-L6-v2!")
            return True
        else:
            print(f"❌ FAILED: Index still has wrong dimension ({stats['dimension']})")
            return False
            
    except Exception as e:
        print(f"💥 CRITICAL ERROR recreating index: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = recreate_pinecone_index()
    if success:
        print("\n🚀 Ready to restart your FastAPI server!")
        print("   The vector storage issue should now be resolved.")
    else:
        print("\n❌ Index recreation failed. Check your Pinecone API key and settings.")
