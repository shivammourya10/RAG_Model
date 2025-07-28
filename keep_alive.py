#!/usr/bin/env python3
"""
Keep-Alive Service for Render Deployment
========================================

This script prevents Render services from spinning down by sending periodic
health check requests. Run this on a separate server/local machine to keep
your Render service active 24/7.

Usage:
    python keep_alive.py https://your-service.onrender.com

Features:
- Pings service every 10 minutes
- Uses standard library (no external dependencies)
- Detailed logging
- Configurable intervals
"""

import asyncio
import urllib.request
import urllib.error
import time
import sys
from datetime import datetime
import argparse

class KeepAliveService:
    """Keep your Render service alive by periodic health checks."""
    
    def __init__(self, service_url: str, interval_minutes: int = 10):
        self.service_url = service_url.rstrip('/')
        self.health_url = f"{self.service_url}/health"
        self.interval_seconds = interval_minutes * 60
        
    async def start(self):
        """Start the keep-alive service."""
        print(f"🚀 Starting Keep-Alive Service")
        print(f"📍 Target: {self.service_url}")
        print(f"⏰ Interval: {self.interval_seconds//60} minutes")
        print(f"🔗 Health Check URL: {self.health_url}")
        print("-" * 50)
        
        try:
            while True:
                await self.ping_service()
                await asyncio.sleep(self.interval_seconds)
        except KeyboardInterrupt:
            print("\n⏹️ Keep-Alive Service stopped by user")
    
    async def ping_service(self):
        """Send a ping to keep the service alive."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            start_time = time.time()
            
            # Use urllib instead of aiohttp (no external dependencies)
            req = urllib.request.Request(self.health_url, headers={
                'User-Agent': 'KeepAlive/1.0'
            })
            
            with urllib.request.urlopen(req, timeout=30) as response:
                response_time = time.time() - start_time
                status_code = response.getcode()
                
                if status_code == 200:
                    print(f"✅ {timestamp} - Service alive (Status: {status_code}, Time: {response_time:.2f}s)")
                else:
                    print(f"⚠️ {timestamp} - Unexpected status: {status_code} (Time: {response_time:.2f}s)")
                    
        except urllib.error.HTTPError as e:
            response_time = time.time() - start_time
            print(f"⚠️ {timestamp} - HTTP Error: {e.code} (Time: {response_time:.2f}s)")
        except urllib.error.URLError as e:
            print(f"❌ {timestamp} - Connection failed: {e.reason}")
        except Exception as e:
            print(f"💥 {timestamp} - Unexpected error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Keep Render service alive")
    parser.add_argument("url", help="Your Render service URL (e.g., https://your-service.onrender.com)")
    parser.add_argument("--interval", type=int, default=10, 
                       help="Ping interval in minutes (default: 10)")
    
    args = parser.parse_args()
    
    # Validate URL
    if not args.url.startswith(('http://', 'https://')):
        print("❌ Error: URL must start with http:// or https://")
        sys.exit(1)
    
    service = KeepAliveService(args.url, args.interval)
    
    try:
        asyncio.run(service.start())
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
