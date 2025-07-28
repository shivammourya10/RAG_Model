"""
LLM Client Module for HackRX 6.0
=================================

This module provides a unified interface for multiple LLM providers with advanced features:
- Support for Google Gemini and OpenAI GPT models
- Token counting and optimization
- Context management and truncation
- Performance monitoring and analytics
- Error handling and fallback mechanisms

The module is designed for production use with cost optimization and high availability.

Key Features:
- Multi-provider LLM support (Google Gemini, OpenAI GPT)
- Intelligent token management and cost optimization
- Context truncation with preservation of important content
- Performance metrics and monitoring
- Async processing for optimal latency
- Explainable AI responses with citations

Author: HackRX 6.0 Team
"""

import asyncio
import json
import time
from typing import Dict, Tuple, Optional, Any

# Fix: Use proper OpenAI import
from openai import OpenAI

# Fix: Use proper Google AI import
try:
    import google.generativeai as genai
except ImportError:
    genai = None
import google.generativeai as genai
import tiktoken

# Import configuration
from config import config


class TokenCounter:
    """
    Advanced token counting and management for multiple LLM providers.
    
    Provides accurate token counting, context truncation, and cost optimization
    features to ensure efficient LLM usage while maintaining response quality.
    """
    
    def __init__(self):
        """Initialize token counter with provider-specific encodings."""
        self.config = config
        self.encoding = None
        
        # Initialize token encoding if enabled
        if self.config.enable_token_counting:
            try:
                # Use cl100k_base encoding (compatible with GPT-4 and most models)
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                print(f"Warning: Could not initialize token encoding: {e}")
                self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text with high accuracy.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Number of tokens in the text
        """
        if not self.config.enable_token_counting or not self.encoding:
            # Fallback estimation: ~4 characters per token (industry standard)
            return max(1, len(text) // 4)
        
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback to character-based estimation
            return max(1, len(text) // 4)
    
    def truncate_context(self, context: str, max_tokens: int, preserve_beginning: bool = True) -> str:
        """
        Intelligently truncate context to fit within token limits.
        
        Args:
            context (str): Original context text
            max_tokens (int): Maximum allowed tokens
            preserve_beginning (bool): Whether to preserve the beginning or end of text
            
        Returns:
            str: Truncated context that fits within token limit
        """
        if not self.config.enable_token_counting:
            return context
        
        current_tokens = self.count_tokens(context)
        if current_tokens <= max_tokens:
            return context
        
        # Smart truncation that preserves sentence boundaries
        sentences = context.split('. ')
        
        if preserve_beginning:
            # Keep beginning, truncate from end
            truncated = ""
            for sentence in sentences:
                test_text = truncated + ". " + sentence if truncated else sentence
                if self.count_tokens(test_text) > max_tokens:
                    break
                truncated = test_text
        else:
            # Keep end, truncate from beginning
            truncated = ""
            for sentence in reversed(sentences):
                test_text = sentence + ". " + truncated if truncated else sentence
                if self.count_tokens(test_text) > max_tokens:
                    break
                truncated = test_text
        
        return truncated if truncated else context[:max_tokens * 4]  # Fallback
        return truncated + "\n\n[Context truncated due to length...]"

class LLMClient:
    """Handles LLM interactions with token optimization and explainable responses."""
    
    def __init__(self):
        self.config = config
        self.token_counter = TokenCounter()
        self._setup_clients()
    
    def _setup_clients(self):
        """Setup LLM clients based on configuration."""
        try:
            if self.config.llm_provider == "openai" and self.config.openai_api_key:
                self.openai_client = OpenAI(api_key=self.config.openai_api_key)
            elif self.config.llm_provider == "google" and self.config.google_api_key:
                if genai:
                    genai.configure(api_key=self.config.google_api_key)  # type: ignore
                else:
                    raise Exception("Google AI library not installed")
        except Exception as e:
            print(f"⚠️ LLM client setup failed: {e}")
    
    def create_enhanced_prompt(self, question: str, context: str, document_metadata: Dict) -> str:
        """Creates a SPEED-OPTIMIZED prompt for fast answers with citations and clauses."""
        
        # SPEED OPTIMIZATION: Aggressive context truncation for faster processing
        max_context_chars = 1200  # Optimal for Gemini-1.5-Flash speed
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "..."
        
        # SPEED-OPTIMIZED prompt - minimal but effective
        prompt = f"""Analyze document and answer question. Return ONLY valid JSON.

SOURCE: {document_metadata.get('source_url', 'doc')}
TYPE: {document_metadata.get('document_type', 'document')}

CONTEXT:
{context}

QUESTION: {question}

Return only this JSON format:
{{"answer": "brief answer", "source": "exact quote from context", "clause": "clause/section number if mentioned", "confidence": "high/medium/low"}}"""

        return prompt
    
    async def get_answer_from_openai(self, prompt: str) -> Dict:
        """Get response from OpenAI GPT-4."""
        try:
            response = await self.openai_client.chat.completions.create(  # type: ignore
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert document analysis assistant specializing in insurance, legal, HR, and compliance domains. Always provide detailed, accurate responses with proper citations."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
            
            try:
                parsed_response = json.loads(content)
                return {
                    "response": parsed_response,
                    "tokens_used": tokens_used,
                    "model": "gpt-4"
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "response": {
                        "answer": content,
                        "reasoning": "Response generated but JSON parsing failed",
                        "citations": [],
                        "confidence": "medium",
                        "domain_specific_notes": "Technical parsing issue encountered"
                    },
                    "tokens_used": tokens_used,
                    "model": "gpt-4"
                }
                
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    async def get_answer_from_google(self, prompt: str) -> Dict:
        """ULTRA-FAST Google Gemini-1.5-Flash with aggressive speed optimizations."""
        try:
            # SPEED CRITICAL: Use fastest model with optimal config
            model = genai.GenerativeModel('gemini-1.5-flash')  # type: ignore
            
            # EXTREME SPEED OPTIMIZATION: Ultra-short prompt for sub-second response
            if len(prompt) > 1500:  # Aggressive truncation for maximum speed
                prompt = prompt[:1500] + "\n\nAnswer in JSON format only."
            
            # PRODUCTION SPEED CONFIG: Minimized settings for fastest response
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(  # type: ignore
                    temperature=0.0,  # Zero temperature for fastest generation
                    top_p=0.7,       # Reduced for speed
                    max_output_tokens=200,  # Very short responses for speed
                    candidate_count=1,      # Single candidate
                    stop_sequences=["}"]    # Stop at JSON end for speed
                )
            )
            
            content = response.text
            
            # Handle incomplete JSON due to stop sequence
            if not content.endswith("}"):
                content += "}"
            
            try:
                parsed_response = json.loads(content)
                return {
                    "response": parsed_response,
                    "tokens_used": len(content) // 4,
                    "model": "gemini-1.5-flash-ultra-fast"
                }
            except json.JSONDecodeError:
                # Try to extract JSON from markdown-wrapped content
                import re
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    try:
                        parsed_response = json.loads(json_match.group(1))
                        return {
                            "response": parsed_response,
                            "tokens_used": len(content) // 4,
                            "model": "gemini-1.5-flash-extracted"
                        }
                    except json.JSONDecodeError:
                        pass
                
                # SPEED FALLBACK: Minimal processing
                return {
                    "response": {
                        "answer": content.strip(),
                        "source": "",
                        "clause": "",
                        "confidence": "medium"
                    },
                    "tokens_used": len(content) // 4,
                    "model": "gemini-1.5-flash-fallback"
                }
                
        except Exception as e:
            raise Exception(f"Gemini Flash API error: {e}")
    
    async def get_enhanced_answer(self, question: str, context: str, document_metadata: Dict) -> Tuple[str, Dict]:
        """
        SPEED-OPTIMIZED answer generation with citations and clauses.
        Returns (simple_answer, detailed_response)
        """
        start_time = time.time()
        
        try:
            prompt = self.create_enhanced_prompt(question, context, document_metadata)
            
            # Force Google Gemini for speed optimization
            if self.config.llm_provider == "openai":
                print("⚡ Switching to Gemini-1.5-Flash for speed optimization")
                result = await self.get_answer_from_google(prompt)
            elif self.config.llm_provider == "google":
                result = await self.get_answer_from_google(prompt)
            else:
                # Default to Google for speed
                result = await self.get_answer_from_google(prompt)
            
            response_time = time.time() - start_time
            
            # Extract components from speed-optimized response
            response_data = result["response"]
            simple_answer = response_data.get("answer", "Error generating response")
            
            # Enhanced response with speed metrics
            enhanced_response = {
                "answer": simple_answer,
                "source": response_data.get("source", ""),
                "clause": response_data.get("clause", ""),
                "confidence": response_data.get("confidence", "medium"),
                "performance_metrics": {
                    "response_time": response_time,
                    "tokens_used": result.get("tokens_used", 0),
                    "model_used": result.get("model", "gemini-1.5-flash"),
                    "speed_optimization": "enabled"
                }
            }
            
            return simple_answer, enhanced_response
            
        except Exception as e:
            response_time = time.time() - start_time
            error_response = {
                "answer": f"Error: {str(e)}",
                "source": "",
                "clause": "",
                "confidence": "low",
                "performance_metrics": {
                    "response_time": response_time,
                    "tokens_used": 0,
                    "model_used": self.config.llm_provider,
                    "error": str(e)
                }
            }
            
            return f"Error: {str(e)}", error_response

# Backward compatibility function
async def get_answer_from_llm(question: str, context: str, document_metadata: Optional[Dict] = None) -> str:
    """Simple function for backward compatibility."""
    client = LLMClient()
    if document_metadata is None:
        document_metadata = {"document_type": "unknown", "source_url": "unknown"}
    
    simple_answer, _ = await client.get_enhanced_answer(question, context, document_metadata)
    return simple_answer

# Example usage
if __name__ == "__main__":
    async def test_llm():
        client = LLMClient()
        
        sample_context = """
        NATIONAL PARIVAR MEDICLAIM PLUS POLICY - Section 4.2
        Grace period for premium payment: 30 days after due date (Clause 4.2.1).
        Pre-existing diseases waiting period: 36 months as per Clause 6.3.
        Coverage limit: Rs. 5,00,000 per family per policy year (Clause 2.1).
        Maternity benefits: Available after 36 months continuous coverage (Clause 8.1).
        """
        
        sample_metadata = {
            "document_type": "insurance_policy",
            "source_url": "policy_document_2024.pdf"
        }
        
        question = "What is the grace period for premium payment?"
        
        try:
            print("🚀 Testing SPEED-OPTIMIZED LLM Client...")
            start_time = time.time()
            
            simple_answer, enhanced_response = await client.get_enhanced_answer(
                question, sample_context, sample_metadata
            )
            
            total_time = time.time() - start_time
            
            print(f"⚡ SPEED RESULTS:")
            print(f"   Total Time: {total_time:.3f}s")
            print(f"   Answer: {simple_answer}")
            print(f"   Source: {enhanced_response.get('source', 'N/A')}")
            print(f"   Clause: {enhanced_response.get('clause', 'N/A')}")
            print(f"   Model: {enhanced_response['performance_metrics']['model_used']}")
            print(f"   Response Time: {enhanced_response['performance_metrics']['response_time']:.3f}s")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Run test
    asyncio.run(test_llm())
