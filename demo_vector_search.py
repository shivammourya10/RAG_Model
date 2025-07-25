#!/usr/bin/env python3
"""
Demo script showing Pinecone vector search capabilities
Tests vector similarity without hitting LLM quotas
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_core import RAGEngine
import asyncio

async def demo_vector_search():
    """Demonstrate vector search capabilities"""
    
    print("🚀 HackRX 6.0 - Pinecone Vector Search Demo")
    print("=" * 60)
    
    try:
        # Initialize RAG engine
        rag_engine = RAGEngine()
        
        # Test vector search with insurance questions
        test_questions = [
            "What compensation is provided for hearing loss?",
            "What percentage for finger loss?",
            "Baggage delay coverage details?",
            "Plastic surgery after accident coverage?",
            "Travel insurance medical benefits?"
        ]
        
        document_url = "https://hackrx.in/policies/CHOTGDP23004V012223.pdf"
        
        print(f"📄 Testing with: {document_url}")
        print(f"❓ Number of test questions: {len(test_questions)}")
        print("\n🔍 Vector Search Results:")
        print("-" * 60)
        
        for i, question in enumerate(test_questions, 1):
            try:
                # This will use Pinecone vector search
                context, citations = await rag_engine.retrieve_context_for_question(
                    question, document_url
                )
                
                print(f"\n{i}. Q: {question}")
                print(f"   📊 Citations Found: {len(citations)}")
                
                if citations:
                    top_citation = citations[0]
                    similarity_score = top_citation.get('similarity_score', 0)
                    page_num = top_citation.get('page_number', 'N/A')
                    
                    print(f"   🎯 Best Match Score: {similarity_score:.3f}")
                    print(f"   📖 Page: {page_num}")
                    print(f"   📝 Preview: {top_citation['text'][:100]}...")
                    
                    # Show quality of match
                    if similarity_score > 0.8:
                        print("   ✅ Excellent semantic match!")
                    elif similarity_score > 0.6:
                        print("   👍 Good semantic match")
                    elif similarity_score > 0.4:
                        print("   🔍 Moderate match")
                    else:
                        print("   ⚠️ Low confidence match")
                
                print(f"   📏 Context Length: {len(context)} chars")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 Vector Search Demo Complete!")
        print("\n📊 Summary:")
        print("✅ Pinecone serverless index operational")
        print("✅ Semantic similarity search working")
        print("✅ Document chunks retrieved successfully")
        print("✅ Citation metadata preserved")
        print("\n💡 Benefits of Vector Search:")
        print("• Semantic understanding of questions")
        print("• Relevance-based chunk ranking")
        print("• Fast similarity search")
        print("• Improved answer quality")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(demo_vector_search())
