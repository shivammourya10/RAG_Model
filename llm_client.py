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

import openai
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
        if self.config.llm_provider == "openai" and self.config.openai_api_key:
            openai.api_key = self.config.openai_api_key
        elif self.config.llm_provider == "google" and self.config.google_api_key:
            genai.configure(api_key=self.config.google_api_key)
    
    def create_enhanced_prompt(self, question: str, context: str, document_metadata: Dict) -> str:
        """Creates an enhanced prompt for explainable answers with citations."""
        
        # Optimize context length
        max_context_tokens = self.config.max_tokens_per_request - 500  # Reserve tokens for question and response
        optimized_context = self.token_counter.truncate_context(context, max_context_tokens)
        
        prompt = f"""You are an intelligent document analysis system specialized in insurance, legal, HR, and compliance domains. 

Your task is to answer questions based SOLELY on the provided document context and provide explainable reasoning with precise citations.

DOCUMENT METADATA:
- Document Type: {document_metadata.get('document_type', 'Unknown')}
- Source: {document_metadata.get('source_url', 'Unknown')}

CONTEXT:
{optimized_context}

QUESTION: {question}

RESPONSE REQUIREMENTS:
1. Answer the question accurately and completely based ONLY on the provided context
2. If the context doesn't contain sufficient information, clearly state "Answer not found in context"
3. Provide specific citations from the context that support your answer
4. Explain your reasoning and decision-making process
5. For insurance/legal questions, identify relevant clauses, conditions, and limitations
6. Use clear, professional language appropriate for the domain

RESPONSE FORMAT:
Provide your response as a JSON object with the following structure:
{{
    "answer": "Your complete answer to the question",
    "reasoning": "Step-by-step explanation of how you arrived at this answer",
    "citations": [
        {{
            "text": "Exact text from context that supports the answer",
            "source": "Location/section where this information was found"
        }}
    ],
    "confidence": "high|medium|low - based on clarity of information in context",
    "domain_specific_notes": "Any additional domain-specific considerations (insurance terms, legal implications, etc.)"
}}

If no relevant information is found, respond with:
{{
    "answer": "Answer not found in context",
    "reasoning": "The provided context does not contain information relevant to answering this question",
    "citations": [],
    "confidence": "low",
    "domain_specific_notes": "Unable to provide domain-specific analysis due to lack of relevant information"
}}"""

        return prompt
    
    async def get_answer_from_openai(self, prompt: str) -> Dict:
        """Get response from OpenAI GPT-4."""
        try:
            response = await openai.ChatCompletion.acreate(
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
        """Get response from Google Gemini."""
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1500,
                )
            )
            
            content = response.text
            
            try:
                parsed_response = json.loads(content)
                return {
                    "response": parsed_response,
                    "tokens_used": len(content) // 4,  # Rough estimation
                    "model": "gemini-1.5-flash"
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
                    "tokens_used": len(content) // 4,
                    "model": "gemini-1.5-flash"
                }
                
        except Exception as e:
            raise Exception(f"Google Gemini API error: {e}")
    
    async def get_enhanced_answer(self, question: str, context: str, document_metadata: Dict) -> Tuple[str, Dict]:
        """
        Get enhanced answer with explanations and citations.
        Returns (simple_answer, detailed_response)
        """
        start_time = time.time()
        
        try:
            prompt = self.create_enhanced_prompt(question, context, document_metadata)
            
            if self.config.llm_provider == "openai":
                result = await self.get_answer_from_openai(prompt)
            elif self.config.llm_provider == "google":
                result = await self.get_answer_from_google(prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
            
            response_time = time.time() - start_time
            
            # Extract simple answer for backward compatibility
            simple_answer = result["response"].get("answer", "Error generating response")
            
            # Enhanced response with metadata
            enhanced_response = {
                "detailed_response": result["response"],
                "performance_metrics": {
                    "response_time": response_time,
                    "tokens_used": result.get("tokens_used", 0),
                    "model_used": result.get("model", self.config.llm_provider)
                }
            }
            
            return simple_answer, enhanced_response
            
        except Exception as e:
            error_response = {
                "detailed_response": {
                    "answer": f"Error processing question: {str(e)}",
                    "reasoning": "Technical error occurred during processing",
                    "citations": [],
                    "confidence": "low",
                    "domain_specific_notes": "Unable to process due to technical error"
                },
                "performance_metrics": {
                    "response_time": time.time() - start_time,
                    "tokens_used": 0,
                    "model_used": self.config.llm_provider,
                    "error": str(e)
                }
            }
            
            return f"Error: {str(e)}", error_response

# Backward compatibility function
async def get_answer_from_llm(question: str, context: str, document_metadata: Dict = None) -> str:
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
        The National Parivar Mediclaim Plus Policy provides coverage for medical expenses.
        Grace period for premium payment: 30 days after due date.
        Waiting period for pre-existing diseases: 36 months.
        """
        
        sample_metadata = {
            "document_type": "pdf",
            "source_url": "https://example.com/policy.pdf"
        }
        
        question = "What is the grace period for premium payment?"
        
        try:
            simple_answer, enhanced_response = await client.get_enhanced_answer(
                question, sample_context, sample_metadata
            )
            
            print(f"Simple Answer: {simple_answer}")
            print(f"Enhanced Response: {json.dumps(enhanced_response, indent=2)}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Run test
    asyncio.run(test_llm())
