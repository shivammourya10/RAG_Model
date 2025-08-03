#!/bin/bash
# GCP Docker Deployment Script for RAG Model

# Text formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}===============================================${NC}"
echo -e "${BOLD}${BLUE}   RAG Model Deployment to GCP with Docker     ${NC}"
echo -e "${BOLD}${BLUE}===============================================${NC}"
echo

# Step 1: Variables - Update these with your GCP project information
echo -e "${BOLD}${YELLOW}Step 1: Setting up GCP variables${NC}"
# IMPORTANT: Replace this with your actual GCP Project ID
# Find this in GCP Console → IAM & Admin → Settings or at the top of the console
PROJECT_ID="rag-model-shivam"  # Your newly created project ID

# Other configuration (can leave as defaults)
REGION="asia-south2"             # Choose your preferred region
SERVICE_NAME="rag-model"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"
MEMORY="2Gi"                     # Memory allocation
CPU="1"                          # CPU allocation
MAX_INSTANCES="2"               # Maximum number of instances
MIN_INSTANCES="1"                # Minimum number of instances

# If using your Render PostgreSQL, leave this commented out
# CLOUDSQL_INSTANCE="$PROJECT_ID:$REGION:hackrx-db"

# Check if required tools are installed
echo -e "${BOLD}${YELLOW}Step 2: Checking required tools${NC}"

# Check for gcloud
if ! command -v gcloud &> /dev/null; then
    echo -e "${YELLOW}Google Cloud SDK (gcloud) not found. Please install it from: https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi

# Check for docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker not found. Please install it from: https://docs.docker.com/get-docker/${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All required tools are installed${NC}"
echo

# Step 3: GCP Project Setup
echo -e "${BOLD}${YELLOW}Step 3: GCP Project Setup${NC}"
echo -e "Current project: $(gcloud config get-value project)"
read -p "Do you want to use this project or switch to '$PROJECT_ID'? (y=use current/n=switch): " switch_project

if [[ $switch_project == "n" ]]; then
    echo "Setting project to $PROJECT_ID"
    gcloud config set project $PROJECT_ID
    
    # Check if project exists and create if needed
    if ! gcloud projects describe $PROJECT_ID &> /dev/null; then
        echo "Project $PROJECT_ID doesn't exist. Creating now..."
        gcloud projects create $PROJECT_ID
    fi
else
    PROJECT_ID=$(gcloud config get-value project)
    IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"
    echo "Using current project: $PROJECT_ID"
fi

# Enable required APIs
echo "Enabling required GCP APIs..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com artifactregistry.googleapis.com containerregistry.googleapis.com sqladmin.googleapis.com --project=$PROJECT_ID

echo -e "${GREEN}✓ GCP project setup complete${NC}"
echo

# Step 4: Build and push Docker image
echo -e "${BOLD}${YELLOW}Step 4: Building and pushing Docker image${NC}"
echo "Building Docker image for linux/amd64 platform..."
docker build --platform linux/amd64 -t $IMAGE_NAME .

echo "Authenticating Docker with GCP..."
gcloud auth configure-docker

echo "Pushing image to Google Container Registry..."
docker push $IMAGE_NAME

echo -e "${GREEN}✓ Docker image built and pushed to GCR${NC}"
echo

# Step 5: Create and deploy Cloud Run service
echo -e "${BOLD}${YELLOW}Step 5: Deploying to Cloud Run${NC}"
echo "Creating Cloud Run service..."

# Deploy with environment variables from .env file
echo "Reading environment variables from .env file..."
ENV_VARS=""
if [ -f .env ]; then
    # Process .env file and format for gcloud command
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        if [[ ! $line =~ ^#.* ]] && [[ ! -z $line ]]; then
            ENV_VARS="$ENV_VARS,$(echo $line | tr -d ' ')"
        fi
    done < .env
    
    # Remove leading comma
    ENV_VARS=${ENV_VARS#,}
    
    echo "Found environment variables in .env file"
    
    # Verify DATABASE_URL exists for Render PostgreSQL
    if [[ ! $ENV_VARS =~ DATABASE_URL ]]; then
        echo -e "${YELLOW}WARNING: No DATABASE_URL found in .env file. Make sure your Render PostgreSQL URL is properly set.${NC}"
        read -p "Would you like to enter your Render PostgreSQL URL now? (y/n): " add_db_url
        if [[ $add_db_url == "y" ]]; then
            read -p "Enter your Render PostgreSQL URL: " render_db_url
            ENV_VARS="$ENV_VARS,DATABASE_URL=$render_db_url"
            echo "Added DATABASE_URL to environment variables"
        fi
    else
        echo "✓ Found DATABASE_URL for Render PostgreSQL"
    fi
else
    echo "No .env file found. Proceeding without environment variables."
    read -p "Would you like to enter your Render PostgreSQL URL now? (y/n): " add_db_url
    if [[ $add_db_url == "y" ]]; then
        read -p "Enter your Render PostgreSQL URL: " render_db_url
        ENV_VARS="DATABASE_URL=$render_db_url"
        echo "Added DATABASE_URL to environment variables"
    fi
fi

# Deploy to Cloud Run
echo "Deploying service to Cloud Run..."

# Check if using Cloud SQL
CLOUDSQL_FLAG=""
if [ ! -z ${CLOUDSQL_INSTANCE+x} ]; then
    CLOUDSQL_FLAG="--add-cloudsql-instances $CLOUDSQL_INSTANCE"
    echo "Connecting to Cloud SQL instance: $CLOUDSQL_INSTANCE"
fi

gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --memory $MEMORY \
  --cpu $CPU \
  --min-instances $MIN_INSTANCES \
  --max-instances $MAX_INSTANCES \
  $CLOUDSQL_FLAG \
  --set-env-vars=$ENV_VARS \
  --allow-unauthenticated

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')

echo -e "${GREEN}✓ Deployment to Cloud Run complete!${NC}"
echo -e "${BOLD}Your service is deployed at: ${BLUE}$SERVICE_URL${NC}"
echo -e "Health check endpoint: ${BLUE}${SERVICE_URL}/health${NC}"
echo -e "API endpoint: ${BLUE}${SERVICE_URL}/api/v1/hackrx/run${NC}"
echo

# Save URLs to a file for easy reference
echo "Service URL: $SERVICE_URL" > deployment-urls.txt
echo "Health check: $SERVICE_URL/health" >> deployment-urls.txt
echo "API endpoint: $SERVICE_URL/api/v1/hackrx/run" >> deployment-urls.txt
echo -e "URLs saved to ${BOLD}deployment-urls.txt${NC} for future reference"

# Test health endpoint automatically
echo -e "${YELLOW}Testing health endpoint...${NC}"
if command -v curl &> /dev/null; then
    curl -s "${SERVICE_URL}/health" | head -n 20
    echo -e "\n${GREEN}✓ Health endpoint test complete${NC}"
else
    echo -e "${YELLOW}curl not found. Please test manually:${NC}"
    echo -e "curl -X GET ${SERVICE_URL}/health"
fi
echo

# Step 6: Monitoring and additional steps
echo -e "${BOLD}${YELLOW}Step 6: Additional Information${NC}"
echo -e "1. View logs: ${BLUE}gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME'${NC}"
echo -e "2. Update service: Edit Dockerfile or code, then run this script again"
echo -e "3. Delete service: ${BLUE}gcloud run services delete $SERVICE_NAME --platform managed --region $REGION${NC}"
echo -e "4. Monitor in console: ${BLUE}https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME/metrics${NC}"
echo

echo -e "${BOLD}${GREEN}Deployment process complete!${NC}"
