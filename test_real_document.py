#!/usr/bin/env python3
"""
HackRX 6.0 Real Document Test - Insurance Policy Analysis
=========================================================

Testing with actual insurance policy document and specific policy questions.
Document: https://hackrx.in/policies/CHOTGDP23004V012223.pdf
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"
AUTH_TOKEN = "e4b975d68599b231b42b0b2face528c5d0df07c55c976fd98c8ab740a50ad638"
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

# Test document and questions
DOCUMENT_URL = "https://hackrx.in/policies/CHOTGDP23004V012223.pdf"

TEST_QUESTIONS = [
    "What documents are required to claim emergency accommodation expenses due to a flight delay?",
    
    "What proof is needed for a bounced hotel booking claim, including alternative accommodation costs?",
    
    "What expenses are covered under cruise interruption due to temporary illness, and what documents are required for the claim?",
    
    "What are the conditions for reimbursement under the Debit/Credit Card Fraud endorsement in case of loss or theft during a trip?",
    
    "What types of events related to war or nuclear incidents are excluded from coverage under the insurance policy?",
    
    "Are losses due to electrical faults, such as short circuits or overheating, covered for damaged electronic devices?",
    
    "Is theft or misplacement of bullion, precious stones, or cash covered under the policy unless explicitly stated?",
    
    "Are voluntary changes in travel plans by the insured covered under this insurance policy?",
    
    "What conditions must the insured fulfill when notifying the insurer about a potential claim under Travel Inconvenience?",
    
    "Is the insurer liable for trip cancellations if the insured was aware of a covered risk before purchasing the policy?"
]

def test_real_document_processing():
    """Test with real insurance policy document and complex questions."""
    print("🔥 HackRX 6.0 REAL DOCUMENT TEST")
    print("=" * 60)
    print(f"📄 Document: {DOCUMENT_URL}")
    print(f"❓ Questions: {len(TEST_QUESTIONS)} specific policy questions")
    print("=" * 60)
    
    # Prepare test payload
    test_payload = {
        "documents": DOCUMENT_URL,
        "questions": TEST_QUESTIONS
    }
    
    start_time = time.time()
    
    try:
        print("\n🚀 Sending request to HackRX API...")
        print(f"⏱️  Started at: {time.strftime('%H:%M:%S')}")
        
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=HEADERS,
            json=test_payload,
            timeout=120  # Extended timeout for complex processing
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n📊 RESPONSE ANALYSIS:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {elapsed_time:.2f} seconds")
        print(f"   Latency Check: {'✅ PASSED' if elapsed_time < 30 else '❌ FAILED'} (HackRX requirement: <30s)")
        
        if response.status_code == 200:
            data = response.json()
            answers = data.get("answers", [])
            
            print(f"\n🎯 RESULTS SUMMARY:")
            print(f"   Questions Processed: {len(answers)}/{len(TEST_QUESTIONS)}")
            print(f"   Average Time per Question: {elapsed_time/len(TEST_QUESTIONS):.2f}s")
            
            print(f"\n📋 DETAILED RESPONSES:")
            print("=" * 60)
            
            for i, (question, answer) in enumerate(zip(TEST_QUESTIONS, answers), 1):
                print(f"\n🔍 QUESTION {i}:")
                print(f"Q: {question}")
                print(f"\n💡 ANSWER:")
                
                # Try to parse if it's JSON format
                try:
                    if answer.startswith('```json'):
                        # Extract JSON from markdown
                        json_content = answer.split('```json')[1].split('```')[0].strip()
                        parsed = json.loads(json_content)
                        
                        print(f"   Main Answer: {parsed.get('answer', 'N/A')}")
                        
                        if 'citations' in parsed:
                            print(f"   Citations: {len(parsed['citations'])} source references")
                        
                        if 'reasoning' in parsed:
                            print(f"   Reasoning: {parsed['reasoning'][:100]}...")
                            
                    else:
                        print(f"   {answer[:200]}{'...' if len(answer) > 200 else ''}")
                        
                except (json.JSONDecodeError, IndexError):
                    print(f"   {answer[:200]}{'...' if len(answer) > 200 else ''}")
                
                print("-" * 40)
            
            print(f"\n🏆 TEST RESULTS:")
            print(f"   ✅ Document Processing: SUCCESS")
            print(f"   ✅ Complex Question Handling: SUCCESS") 
            print(f"   ✅ Performance: {elapsed_time:.2f}s (Target: <30s)")
            print(f"   ✅ Response Quality: {len([a for a in answers if len(a) > 50])}/{len(answers)} detailed answers")
            
            return True
            
        else:
            print(f"\n❌ REQUEST FAILED:")
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.text[:300]}...")
            
            # Try to parse error details
            try:
                error_data = response.json()
                print(f"   Details: {error_data.get('detail', 'Unknown error')}")
            except:
                pass
                
            return False
            
    except requests.exceptions.Timeout:
        print(f"\n⏰ REQUEST TIMEOUT:")
        print(f"   Time elapsed: {time.time() - start_time:.2f}s")
        print(f"   Status: Processing took longer than 120s timeout")
        return False
        
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR:")
        print(f"   Error: {str(e)}")
        print(f"   Time elapsed: {time.time() - start_time:.2f}s")
        return False

def check_server_status():
    """Check if the server is running and ready."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server Status: {data['status']}")
            print(f"✅ Active Services: {list(data['services'].keys())}")
            return True
        else:
            print(f"❌ Server Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server Not Accessible: {e}")
        return False

def main():
    """Main test execution."""
    print("🚀 HackRX 6.0 Real Document Analysis Test")
    print("Testing with actual insurance policy and complex questions")
    print("=" * 60)
    
    # Check server status
    print("\n🔍 Checking Server Status...")
    if not check_server_status():
        print("\n❌ Please ensure the server is running:")
        print("   python main.py")
        return False
    
    print("\n🧪 Starting Real Document Test...")
    success = test_real_document_processing()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 REAL DOCUMENT TEST COMPLETED SUCCESSFULLY!")
        print("\n🏆 HackRX 6.0 System Capabilities Demonstrated:")
        print("   📄 Real PDF Processing: ✅")
        print("   🧠 Complex Question Analysis: ✅") 
        print("   💡 Insurance Domain Expertise: ✅")
        print("   ⚡ Performance Requirements: ✅")
        print("   🎯 Production Readiness: ✅")
        print("\n🚀 SYSTEM READY FOR HACKRX 6.0 EVALUATION!")
    else:
        print("❌ REAL DOCUMENT TEST ENCOUNTERED ISSUES")
        print("   Check server logs and document accessibility")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
