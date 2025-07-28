#!/bin/bash

# Render Health Check Script
# This script can be called during deployment to verify system health

echo "🔍 RAG Model Health Check Starting..."

# Check Python installation
echo "🐍 Python Version:"
python --version

# Check critical packages
echo "📦 Checking Critical Dependencies:"
python -c "
import sys
critical_packages = ['fastapi', 'torch', 'sentence_transformers', 'faiss_cpu', 'numpy']

for package in critical_packages:
    try:
        __import__(package)
        print(f'✅ {package}: OK')
    except ImportError as e:
        print(f'❌ {package}: MISSING - {e}')
        sys.exit(1)
"

# Check environment variables
echo "🔧 Environment Configuration:"
python -c "
import os
required_vars = ['RAG_RENDER_MODE', 'HF_HOME', 'TOKENIZERS_PARALLELISM']
optional_vars = ['GOOGLE_API_KEY', 'PINECONE_API_KEY']

for var in required_vars:
    if os.environ.get(var):
        print(f'✅ {var}: SET')
    else:
        print(f'⚠️ {var}: NOT SET')

for var in optional_vars:
    if os.environ.get(var):
        print(f'✅ {var}: SET')
    else:
        print(f'ℹ️ {var}: NOT SET (optional)')
"

# Check model loading capability
echo "🤖 Model Loading Test:"
python -c "
import os
import sys
os.environ['RAG_RENDER_MODE'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    from model_cache import ModelCache
    embedder = ModelCache.get_embedder('all-MiniLM-L6-v2')
    # Test basic embedding
    test_embedding = embedder.encode(['Hello world'])
    print(f'✅ Model loading: OK (embedding shape: {test_embedding.shape})')
except Exception as e:
    print(f'❌ Model loading: FAILED - {e}')
    sys.exit(1)
"

echo "✅ RAG Model Health Check Completed Successfully!"
