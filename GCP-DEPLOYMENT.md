# RAG Model Deployment to GCP with Docker

This guide provides step-by-step instructions for deploying your RAG Model to Google Cloud Platform (GCP) using Docker and Cloud Run.

## Prerequisites

Before you begin, ensure you have:

1. [Google Cloud SDK (gcloud)](https://cloud.google.com/sdk/docs/install) installed and initialized
2. [Docker](https://docs.docker.com/get-docker/) installed
3. A GCP account with billing enabled
4. Your environment variables in `.env` file

## Deployment Steps

### 1. Update Configuration

Edit the `deploy-gcp.sh` script and update these variables with your GCP project information:

```bash
PROJECT_ID="YOUR_GCP_PROJECT_ID"  # e.g., "rag-model-12345"
REGION="us-central1"              # Choose your preferred region
SERVICE_NAME="rag-model"
MEMORY="2Gi"                      # Memory allocation
CPU="1"                           # CPU allocation
MAX_INSTANCES="10"                # Maximum number of instances
MIN_INSTANCES="1"                 # Minimum number of instances
```

### 2. Check Your .env File

Ensure your `.env` file contains all necessary environment variables:

- `PINECONE_API_KEY`: For vector database
- `GOOGLE_API_KEY`: For LLM integration
- `API_BEARER_TOKEN`: For API security
- `DATABASE_URL`: For PostgreSQL connection
- Performance optimization settings

### 3. Run the Deployment Script

```bash
./deploy-gcp.sh
```

This script will:
1. Set up your GCP project
2. Enable required GCP APIs
3. Build your Docker image
4. Push the image to Google Container Registry
5. Deploy to Cloud Run with your environment variables
6. Display the service URL and monitoring commands

### 4. Test Your Deployment

Once deployed, test your API endpoints:

- Health check: `https://YOUR-SERVICE-URL/health`
- API endpoint: `https://YOUR-SERVICE-URL/api/v1/hackrx/run`

Example with curl:

```bash
curl -X GET https://YOUR-SERVICE-URL/health
```

For the main API:

```bash
curl -X POST https://YOUR-SERVICE-URL/api/v1/hackrx/run \
  -H "Authorization: Bearer YOUR_API_BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": [
      "What is the grace period for premium payment?"
    ]
  }'
```

## Monitoring and Management

### View Logs

```bash
gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=rag-model'
```

### Update Your Service

Make changes to your code, then run the deployment script again:

```bash
./deploy-gcp.sh
```

### Monitor Performance

Visit the GCP Console to monitor your service:

```
https://console.cloud.google.com/run/detail/REGION/rag-model/metrics
```

### Delete the Service

If needed, you can delete the service:

```bash
gcloud run services delete rag-model --platform managed --region REGION
```

## Troubleshooting

### Image Build Failures

If your image fails to build:

1. Check for errors in the Docker build logs
2. Ensure your `requirements.txt` uses binary wheels (`--only-binary=all`)
3. Verify Python version compatibility (3.11 recommended)

### Memory/CPU Issues

If your service crashes due to resource constraints:

1. Increase the `MEMORY` and `CPU` settings in the deployment script
2. Deploy again with updated resources

### Cold Start Performance

To improve cold start times:

1. Ensure `RAG_PRELOAD_MODELS=true` is set in your environment
2. Consider increasing `MIN_INSTANCES` to keep instances warm
3. Use the smallest embedding model possible (`all-MiniLM-L6-v2`)

## Cost Optimization

To optimize costs on GCP:

1. Set appropriate `MAX_INSTANCES` to handle your expected load
2. Use `MIN_INSTANCES=0` in non-production environments to scale to zero
3. Monitor your billing dashboard regularly
