#!/usr/bin/env python3
"""
Quick test for single question to check system performance
"""

import requests
import time
import json

def quick_test():
    """Test single question quickly"""
    
    # Single question test
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Authorization": "Bearer hackrx-api-key-2024",
        "Content-Type": "application/json"
    }
    
    data = {
        "documents": "https://hackrx.in/policies/CHOTGDP23004V012223.pdf",
        "questions": [
            "How much compensation is provided for the complete loss of hearing in both ears?"
        ]
    }
    
    print("🚀 Quick HackRX Test - Single Question")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        end_time = time.time()
        
        print(f"⏱️  Response Time: {end_time - start_time:.2f}s")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if 'answers' in result and len(result['answers']) > 0:
                answer = result['answers'][0]
                print(f"\n💡 ANSWER: {answer}")
                print(f"✅ SUCCESS: Found answer in {end_time - start_time:.2f}s")
            else:
                print("❌ No answer found")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    quick_test()
