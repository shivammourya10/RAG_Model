# RAG Model Deployment Options

This repository includes several deployment options for your RAG Model:

## 1. GCP with Docker (Recommended)

Full deployment to Google Cloud Platform using Docker and Cloud Run.

```bash
# Step 1: Update configuration in deploy-gcp.sh
# Step 2: Run the deployment script
./deploy-gcp.sh
```

Documentation:
- [GCP-DEPLOYMENT.md](./GCP-DEPLOYMENT.md): Complete GCP deployment guide
- [GCP-POSTGRESQL.md](./GCP-POSTGRESQL.md): Setting up PostgreSQL on GCP

## 2. Render Deployment

Deploy to Render.com using the included `render.yaml` configuration.

1. Push your code to GitHub
2. Connect your GitHub repository to Render
3. Create a new Web Service using the Blueprint

## 3. Local Docker Development

Run locally using Docker for development and testing.

```bash
# Build the Docker image
docker build -t rag-model:local .

# Run the container
docker run -p 8000:8000 --env-file .env rag-model:local
```

## 4. Local Python Development

Run directly using Python for development.

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Environment Variables

All deployment methods require the following environment variables:

- `PINECONE_API_KEY`: For vector database
- `GOOGLE_API_KEY`: For Google Gemini LLM
- `API_BEARER_TOKEN`: For API security
- `DATABASE_URL`: For PostgreSQL connection

See `.env.example` for a complete list of configuration options.

## API Documentation

Once deployed, API documentation is available at:

- Swagger UI: `https://your-deployment-url/docs`
- ReDoc: `https://your-deployment-url/redoc`

## Monitoring and Health Checks

All deployments include a health check endpoint:

```
GET /health
```

This endpoint verifies the status of all services and returns diagnostics.

## Performance Optimization

For production deployment, the following optimizations are enabled:

- Binary embeddings for 100x faster processing
- In-memory vector storage for speed
- Hybrid vector DB with progressive fallbacks
- Cold start optimization with model preloading
