#!/bin/bash

# Keep-Alive Script for Render Deployment
# =======================================
# Simple bash script to keep your Render service alive by pinging every 10 minutes
# Usage: ./keep_alive.sh https://your-service.onrender.com

if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide your Render service URL"
    echo "Usage: ./keep_alive.sh https://your-service.onrender.com"
    exit 1
fi

SERVICE_URL="$1"
HEALTH_URL="${SERVICE_URL}/health"

echo "🚀 Starting Keep-Alive Service for Render"
echo "📍 Target: $SERVICE_URL"
echo "🔗 Health Check: $HEALTH_URL"
echo "⏰ Pinging every 10 minutes..."
echo "💡 Press Ctrl+C to stop"
echo "-" | tr '-' '-' | head -c 50; echo

# Function to ping the service
ping_service() {
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Use curl to ping the health endpoint
    if response=$(curl -s -w "HTTPSTATUS:%{http_code};TIME:%{time_total}" "$HEALTH_URL" 2>/dev/null); then
        http_code=$(echo "$response" | sed -n 's/.*HTTPSTATUS:\([0-9]*\).*/\1/p')
        time_total=$(echo "$response" | sed -n 's/.*TIME:\([0-9.]*\).*/\1/p')
        
        if [ "$http_code" = "200" ]; then
            echo "✅ $timestamp - Service alive (Status: $http_code, Time: ${time_total}s)"
        else
            echo "⚠️ $timestamp - Unexpected status: $http_code (Time: ${time_total}s)"
        fi
    else
        echo "❌ $timestamp - Connection failed"
    fi
}

# Main loop
while true; do
    ping_service
    sleep 600  # 10 minutes = 600 seconds
done
