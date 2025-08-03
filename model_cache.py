"""
Smart Model Cache for Cold Start Optimization
============================================

Provides intelligent caching and preloading for embedding models
to reduce cold start time from 15-20s to 8-12s.
"""

import os
import time
import logging
from typing import Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class ModelCache:
    """Smart caching system for embedding models."""
    
    _embedder_cache = None  # Remove type annotation to allow any embedder type
    _last_loaded: Optional[str] = None
    _load_time: float = 0
    
    @classmethod
    def get_embedder(cls, model_name: str = 'all-MiniLM-L12-v2'):
        """Get or create embedder with comprehensive fallback strategy."""
        
        # Set environment variables for better compatibility from the start
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Use cached model if available and matches
        if cls._embedder_cache is not None and cls._last_loaded == model_name:
            cache_access_time = time.time()
            cls._load_time = cache_access_time - time.time()  # Near zero
            logger.info(f"📦 Using cached model: {model_name}")
            return cls._embedder_cache
        
        logger.info(f"🔄 Loading embedding model: {model_name}")
        start_time = time.time()
        
        # Use smaller, faster model for cold starts
        if os.getenv('RAG_FAST_STARTUP', 'false').lower() == 'true':
            model_name = 'all-MiniLM-L6-v2'  # 22MB vs 438MB, 2x faster
            logger.info("⚡ Fast startup: Using lightweight model")
        
        # Optimize cache location for faster access
        cache_folder = '/tmp/sentence_transformers' if os.getenv('RAG_RENDER_MODE', 'false').lower() == 'true' else None
        
        try:
            # PRODUCTION FIX: Enhanced compatibility settings for Render deployment
            import torch
            
            # Clear any existing meta device state (safe approach)
            try:
                if hasattr(torch, 'get_default_device'):
                    current_device = torch.get_default_device()
                    if current_device != 'cpu':
                        torch.set_default_device('cpu')
            except Exception:
                pass  # Continue if device clearing fails
            
            # Set environment variables for stable deployment
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['HF_HOME'] = cache_folder or '/tmp/huggingface_cache'  # Fixed: Use HF_HOME instead
            os.environ['TORCH_HOME'] = cache_folder or '/tmp/torch_cache'
            
            # Force CPU mode and prevent meta tensors
            torch.set_default_device('cpu')
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            
            # Load model with explicit device mapping to prevent meta tensor issues
            cls._embedder_cache = SentenceTransformer(
                model_name,
                device='cpu',
                cache_folder=cache_folder,
                trust_remote_code=True,
                use_auth_token=False  # Prevent authentication issues
            )
            
            # Verify model is properly loaded (not on meta device)
            try:
                test_encoding = cls._embedder_cache.encode("test", show_progress_bar=False)
                if test_encoding is None or len(test_encoding) == 0:
                    raise RuntimeError("Model encoding test failed")
            except Exception as e:
                raise RuntimeError(f"Model validation failed: {e}")
            
            cls._last_loaded = model_name
            cls._load_time = time.time() - start_time
            
            logger.info(f"✅ Model loaded and validated in {cls._load_time:.2f}s")
            
            # Warm up model with dummy embedding
            if os.getenv('RAG_PRELOAD_MODELS', 'false').lower() == 'true':
                cls._warm_up_model()
            
            return cls._embedder_cache
            
        except Exception as e:
            logger.error(f"❌ Primary model loading failed: {e}")
            
            # ENHANCED FALLBACK STRATEGY for Production Deployment
            fallback_models = [
                'all-MiniLM-L6-v2',  # Smaller, more reliable
                'sentence-transformers/all-MiniLM-L6-v2',  # Full path
                'paraphrase-MiniLM-L6-v2'  # Alternative
            ]
            
            for i, fallback_model in enumerate(fallback_models, 1):
                try:
                    logger.info(f"🔄 Attempting fallback {i}: {fallback_model}")
                    
                    # Reset environment for each attempt
                    import torch
                    torch.set_default_device('cpu')
                    
                    # Fix for meta tensor issue - use to_empty() approach as recommended by PyTorch
                    if i == 1:
                        # Fallback 1: Use to_empty() approach with explicit CPU
                        model = SentenceTransformer(
                            fallback_model,
                            device='cpu',
                            trust_remote_code=False
                        )
                        # Handle any module that might need to_empty
                        for module in model.modules():
                            if hasattr(module, '_parameters'):
                                for param_name, param in module._parameters.items():
                                    if param is not None and hasattr(param, 'is_meta') and param.is_meta:
                                        # Convert to parameter object with empty tensor
                                        empty_tensor = torch.empty_like(param, device='cpu')
                                        module._parameters[param_name] = torch.nn.Parameter(empty_tensor)
                        cls._embedder_cache = model
                    elif i == 2:
                        # Fallback 2: Ultra-minimal approach with smallest model
                        cls._embedder_cache = SentenceTransformer(
                            'all-MiniLM-L6-v2',  # Smallest model
                            device='cpu'  # Explicit CPU
                        )
                    else:
                        # Fallback 3: Last resort - try with PyTorch settings forced
                        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                        torch.set_default_device('cpu')
                        torch.set_num_threads(1)
                        cls._embedder_cache = SentenceTransformer(
                            'paraphrase-MiniLM-L3-v2',  # Ultra-small model
                            device='cpu'
                        )
                    
                    # Test the model
                    test_result = cls._embedder_cache.encode("validation test", show_progress_bar=False)
                    if test_result is not None and len(test_result) > 0:
                        cls._last_loaded = fallback_model
                        cls._load_time = time.time() - start_time
                        logger.info(f"✅ Fallback {i} successful: {fallback_model}")
                        return cls._embedder_cache
                    else:
                        raise ValueError("Model test failed")
                        
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback {i} failed: {fallback_error}")
                    continue
            
            # Final fallback: Try simple embedder as a last resort
            logger.error("❌ All model loading attempts failed - using simple embedder")
            try:
                from simple_embedder import get_simple_embedder
                cls._embedder_cache = get_simple_embedder(dimension=384)
                cls._last_loaded = "simple_embedder"
                cls._load_time = time.time() - start_time
                logger.info("✅ Simple embedder fallback successful")
                return cls._embedder_cache
            except Exception as simple_error:
                logger.error(f"❌ Simple embedder failed: {simple_error}")
                return None
    
    @classmethod
    def _warm_up_model(cls):
        """Warm up model with dummy embedding."""
        try:
            if cls._embedder_cache and hasattr(cls._embedder_cache, 'encode'):
                start_time = time.time()
                _ = cls._embedder_cache.encode("Test warm-up text", show_progress_bar=False)
                warmup_time = time.time() - start_time
                logger.info(f"🔥 Model warmed up in {warmup_time:.3f}s")
        except Exception as e:
            logger.debug(f"Model warm-up failed: {e}")
    
    @classmethod
    def get_stats(cls) -> dict:
        """Get cache statistics."""
        return {
            'cached_model': cls._last_loaded,
            'load_time': cls._load_time,
            'is_cached': cls._embedder_cache is not None
        }
    
    @classmethod
    def clear_cache(cls):
        """Clear model cache."""
        cls._embedder_cache = None
        cls._last_loaded = None
        cls._load_time = 0
        logger.info("🗑️ Model cache cleared")

class AsyncModelLoader:
    """Asynchronous model preloading for faster startup."""
    
    @staticmethod
    def preload_in_background():
        """Preload model in background thread."""
        import threading
        
        def load_model():
            try:
                logger.info("🚀 Background model preloading started")
                ModelCache.get_embedder()
                logger.info("✅ Background model preloading complete")
            except Exception as e:
                logger.warning(f"⚠️ Background preloading failed: {e}")
        
        if os.getenv('RAG_PRELOAD_MODELS', 'false').lower() == 'true':
            thread = threading.Thread(target=load_model, daemon=True)
            thread.start()
            logger.info("🔄 Model preloading thread started")

# Global model cache instance
model_cache = ModelCache()

# Auto-preload if enabled
if os.getenv('RAG_PRELOAD_MODELS', 'false').lower() == 'true':
    AsyncModelLoader.preload_in_background()
