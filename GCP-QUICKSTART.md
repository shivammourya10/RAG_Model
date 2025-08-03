# GCP Deployment Quick Reference Guide

This guide addresses common questions about deploying your RAG Model to GCP with Docker.

## 1. Getting Your GCP Project ID

Your GCP Project ID is found in the Google Cloud Console:
1. Go to https://console.cloud.google.com/
2. Look at the top navigation bar for your project name
3. The Project ID is shown in parentheses or in the project dropdown
4. Use this ID to replace `YOUR_GCP_PROJECT_ID` in the `deploy-gcp.sh` script

## 2. Using Render PostgreSQL with GCP

To use your existing Render PostgreSQL database with GCP:
1. Ensure your `.env` file contains the correct `DATABASE_URL` for your Render database
2. The `deploy-gcp.sh` script will automatically use this URL during deployment
3. Make sure your Render database allows external connections from GCP IPs
   - Go to your Render PostgreSQL dashboard
   - Add GCP IP ranges to the allowed list or enable external connections

## 3. Finding Your Service URL

After running `./deploy-gcp.sh`, the script will:
1. Display your service URL in the terminal
2. Save all URLs to `deployment-urls.txt` for future reference
3. Automatically test your health endpoint if `curl` is installed

If you need to find your URL later:
```bash
gcloud run services describe rag-model --platform managed --region us-central1 --format 'value(status.url)'
```

## 4. Testing Your Deployment

To test your deployed service:

1. **Health Check**:
```bash
curl -X GET https://YOUR-SERVICE-URL/health
```

2. **API Endpoint**:
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

## 5. Differences between Local and Docker Running Commands

- **Local Development**: `python3 main.py`
  - This uses the command in `if __name__ == "__main__"` block in main.py
  - It runs Uvicorn with development settings (reload=True)

- **Docker/GCP Deployment**: Now using `uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 30`
  - This directly calls Uvicorn with production settings
  - Uses consistent settings across all deployment methods
  - The Dockerfile has been updated for consistency

## 6. Monitoring Your Deployment

1. **View Logs**:
```bash
gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=rag-model'
```

2. **Monitor in Console**:
Go to https://console.cloud.google.com/run to see your service metrics

## 7. Troubleshooting

- **Database Connection Issues**: Check if your Render PostgreSQL allows external connections
- **Cold Start Problems**: Increase `MIN_INSTANCES` to keep instances warm
- **Memory Errors**: Increase `MEMORY` in the deployment script
- **API Errors**: Verify all required environment variables are set

For more detailed information, refer to:
- [GCP-DEPLOYMENT.md](./GCP-DEPLOYMENT.md)
- [GCP-POSTGRESQL.md](./GCP-POSTGRESQL.md)
