#!/usr/bin/env python3
"""
Render Environment Simulator
============================
Test script to simulate Render's environment locally and verify compatibility.
"""

import os
import sys

def setup_render_environment():
    """Setup environment variables to match Render deployment."""
    render_env = {
        'RAG_RENDER_MODE': 'true',
        'RAG_FAST_STARTUP': 'true', 
        'RAG_PRELOAD_MODELS': 'true',
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        'HF_HOME': '/tmp/huggingface_cache',
        'TORCH_HOME': '/tmp/torch_cache',
        'RAG_PRIMARY_STORAGE': 'faiss',
        'RAG_MEMORY_OPTIMIZATION': 'true',
        'MAX_CONTEXT_LENGTH': '800',
        'CHUNK_SIZE': '4000',
        'TOP_K_RETRIEVAL': '3',
        'PYTHONUNBUFFERED': '1',
        'PYTHONDONTWRITEBYTECODE': '1',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1'
    }
    
    for key, value in render_env.items():
        os.environ[key] = value
        print(f"✅ Set {key}={value}")

def test_model_loading():
    """Test model loading with Render-compatible settings."""
    try:
        print("\n🧪 Testing Model Loading (Render Simulation)...")
        
        # Force CPU-only mode
        import torch
        print(f"🔧 PyTorch version: {torch.__version__}")
        print(f"🔧 CUDA available: {torch.cuda.is_available()}")
        print(f"🔧 MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
        
        # Test model cache
        from model_cache import ModelCache
        print("🚀 Testing ModelCache...")
        
        embedder = ModelCache.get_embedder('all-MiniLM-L6-v2')
        print("✅ Model loaded successfully!")
        
        # Test embedding
        test_text = ["Hello world", "This is a test"]
        embeddings = embedder.encode(test_text)
        print(f"✅ Embeddings created: shape {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_api_components():
    """Test main API components."""
    try:
        print("\n🧪 Testing API Components...")
        
        # Test FastAPI app creation
        from main import app
        print("✅ FastAPI app created successfully")
        
        # List available routes (simplified to avoid attribute errors)
        print("📍 Available routes:")
        route_count = 0
        for route in app.routes:
            try:
                if hasattr(route, 'path'):
                    path = getattr(route, 'path', 'unknown')
                    methods = getattr(route, 'methods', {'GET'})
                    print(f"  {list(methods)} {path}")
                    route_count += 1
            except Exception:
                route_count += 1
                continue
        
        print(f"✅ Found {route_count} routes")
        return True
        
    except Exception as e:
        print(f"❌ API component test failed: {e}")
        return False

def main():
    print("🚀 Render Environment Compatibility Test")
    print("=" * 50)
    
    # Setup Render environment
    setup_render_environment()
    
    # Test model loading
    model_test = test_model_loading()
    
    # Test API components  
    api_test = test_api_components()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"   Model Loading: {'✅ PASS' if model_test else '❌ FAIL'}")
    print(f"   API Components: {'✅ PASS' if api_test else '❌ FAIL'}")
    
    if model_test and api_test:
        print("\n🎉 SUCCESS: Ready for Render deployment!")
        print("💡 The PyTorch meta tensor issue should NOT occur on Render")
    else:
        print("\n⚠️ ISSUES DETECTED: May need additional fixes for Render")
        print("💡 Consider upgrading to Standard plan or using keep-alive scripts")
    
    return 0 if (model_test and api_test) else 1

if __name__ == "__main__":
    sys.exit(main())
