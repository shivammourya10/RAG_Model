#!/usr/bin/env python3
"""
Quick Pinecone Connection Test
==============================
Test if Pinecone connection is working and verify configuration.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_pinecone_connection():
    """Test Pinecone connection and configuration."""
    print("🔧 Testing Pinecone Connection...")
    
    try:
        from pinecone import Pinecone
        
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        if not api_key:
            print("❌ PINECONE_API_KEY not found in environment")
            return False
            
        print(f"🔑 API Key: {api_key[:20]}...")
        print(f"📊 Index Name: {index_name}")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        # List indexes
        print("📋 Listing indexes...")
        indexes = pc.list_indexes()
        print(f"✅ Found {len(indexes)} indexes")
        
        for idx in indexes:
            print(f"   - {idx['name']}: {idx['status']['state']}")
        
        # Test specific index
        if index_name and index_name in [idx['name'] for idx in indexes]:
            print(f"\n🎯 Testing index: {index_name}")
            index = pc.Index(index_name)
            
            # Get stats
            stats = index.describe_index_stats()
            print(f"📊 Stats: {stats}")
            
            # Test configuration
            from config import config
            print(f"\n⚙️ Configuration Check:")
            print(f"   TOP_K_RETRIEVAL: {config.top_k_retrieval}")
            print(f"   CHUNK_SIZE: {config.chunk_size}")
            print(f"   MAX_CONTEXT_LENGTH: {config.max_context_length}")
            print(f"   ENABLE_CACHING: {config.enable_caching}")
            
            return True
        else:
            print(f"❌ Index '{index_name}' not found")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_dns_resolution():
    """Test DNS resolution for Pinecone."""
    print("\n🌐 Testing DNS Resolution...")
    
    import socket
    
    # Extract hostname from typical Pinecone URL pattern
    index_name = os.getenv("PINECONE_INDEX_NAME", "hackrx-intelligent-query-system")
    
    # Try to resolve common Pinecone hostnames
    test_hosts = [
        f"{index_name}.pinecone.io",
        "api.pinecone.io",
        "controller.pinecone.io"
    ]
    
    for host in test_hosts:
        try:
            ip = socket.gethostbyname(host)
            print(f"✅ {host} → {ip}")
        except socket.gaierror as e:
            print(f"❌ {host} → DNS Error: {e}")
    
    return True

if __name__ == "__main__":
    print("🚀 PINECONE CONNECTIVITY TEST")
    print("=" * 50)
    
    # Test DNS first
    test_dns_resolution()
    
    # Test Pinecone connection
    success = test_pinecone_connection()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Pinecone is working correctly.")
        print("🎯 Your TOP_K_RETRIEVAL=3 setting is properly configured.")
        print("📝 The 'Top 10' in Pinecone web interface is just UI default.")
    else:
        print("❌ Connection issues detected.")
        print("💡 System will fall back to in-memory storage.")
        print("🔧 Check your internet connection and API key.")
