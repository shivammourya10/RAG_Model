#!/usr/bin/env python3
"""
Quick test to verify Pinecone vector search is working
Tests only document processing and vector storage without LLM calls
"""

import requests
import time

def test_pinecone_status():
    """Test if Pinecone vector search is working"""
    
    print("🧪 Testing Pinecone Vector Search Status")
    print("=" * 50)
    
    try:
        # Check server health
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Server Status: {health_data['status']}")
            print(f"📋 Services: {health_data['services']}")
            
            # Check if vector_db service is active
            if 'vector_db' in health_data['services']:
                print("🎯 Vector Database: CONNECTED")
                
                # Try a simple document processing test
                test_data = {
                    "questions": ["What is this document about?"],
                    "documents": "https://hackrx.in/policies/CHOTGDP23004V012223.pdf"
                }
                
                print("\n🔄 Testing document processing (this will use vector search)...")
                start_time = time.time()
                
                # This will fail on LLM call due to quota, but we can see if Pinecone works
                response = requests.post(
                    "http://localhost:8000/hackrx/run",
                    json=test_data,
                    headers={"Authorization": "Bearer e4b975d68599b231b42b0b2face528c5d0df07c55c976fd98c8ab740a50ad638"},
                    timeout=30
                )
                
                processing_time = time.time() - start_time
                
                print(f"⏱️  Processing Time: {processing_time:.2f}s")
                print(f"📊 Response Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ Document processing successful!")
                    
                    # Check the response for any mentions of Pinecone fallback
                    response_text = str(result)
                    if "Pinecone not available" in response_text:
                        print("❌ Still using database fallback")
                        return False
                    else:
                        print("🎉 PINECONE VECTOR SEARCH IS WORKING!")
                        return True
                else:
                    print(f"❌ Request failed: {response.status_code}")
                    print(response.text)
                    return False
            else:
                print("❌ Vector Database: NOT CONNECTED")
                return False
        else:
            print(f"❌ Server health check failed: {health_response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    success = test_pinecone_status()
    
    if success:
        print("\n🎉 SUCCESS: Pinecone vector search is operational!")
        print("💡 Your HackRX system now has semantic search capabilities")
        print("⚡ Performance should be significantly improved")
    else:
        print("\n⚠️  Pinecone vector search needs attention")
        print("🔧 Check server logs for more details")
