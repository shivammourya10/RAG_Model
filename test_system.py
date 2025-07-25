#!/usr/bin/env python3
"""
HackRX 6.0 System Test Suite
============================

Comprehensive test suite to verify all system components and HackRX 6.0 compliance.
Tests include basic functionality, performance benchmarks, and API compliance.

Usage:
    python test_system.py
"""

import asyncio
import json
import time
import requests
from typing import Dict, List

# Test configuration
BASE_URL = "http://localhost:8000"
AUTH_TOKEN = "e4b975d68599b231b42b0b2face528c5d0df07c55c976fd98c8ab740a50ad638"
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

class HackRXTester:
    """Comprehensive test suite for HackRX 6.0 system."""
    
    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result with details."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            print(f"✅ {test_name}: PASSED {details}")
        else:
            print(f"❌ {test_name}: FAILED {details}")
        
        self.results.append({
            "test": test_name,
            "passed": passed,
            "details": details
        })
    
    def test_health_check(self):
        """Test basic health check endpoint."""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_result("Health Check", True, f"All services: {list(data.get('services', {}).keys())}")
                    return True
                else:
                    self.log_result("Health Check", False, f"Status: {data.get('status')}")
            else:
                self.log_result("Health Check", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_result("Health Check", False, f"Error: {str(e)}")
        return False
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        try:
            response = requests.get(f"{BASE_URL}/", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if "HackRX 6.0" in data.get("message", ""):
                    self.log_result("Root Endpoint", True, "System identification confirmed")
                    return True
            
            self.log_result("Root Endpoint", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_result("Root Endpoint", False, f"Error: {str(e)}")
        return False
    
    def test_authentication(self):
        """Test authentication with valid and invalid tokens."""
        # Test without token
        try:
            response = requests.post(
                f"{BASE_URL}/hackrx/run",
                json={"documents": "test", "questions": ["test"]},
                timeout=5
            )
            if response.status_code == 401:
                self.log_result("Auth - No Token", True, "Correctly rejected")
            else:
                self.log_result("Auth - No Token", False, f"Expected 401, got {response.status_code}")
        except Exception as e:
            self.log_result("Auth - No Token", False, f"Error: {str(e)}")
        
        # Test with invalid token
        try:
            invalid_headers = {"Authorization": "Bearer invalid_token", "Content-Type": "application/json"}
            response = requests.post(
                f"{BASE_URL}/hackrx/run",
                headers=invalid_headers,
                json={"documents": "test", "questions": ["test"]},
                timeout=5
            )
            if response.status_code == 401:
                self.log_result("Auth - Invalid Token", True, "Correctly rejected")
            else:
                self.log_result("Auth - Invalid Token", False, f"Expected 401, got {response.status_code}")
        except Exception as e:
            self.log_result("Auth - Invalid Token", False, f"Error: {str(e)}")
    
    def test_basic_functionality(self):
        """Test basic HackRX functionality with sample questions."""
        test_payload = {
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
            "questions": [
                "What is the grace period for premium payment?",
                "What are the waiting periods in this policy?",
                "Does this policy cover maternity expenses?"
            ]
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{BASE_URL}/hackrx/run",
                headers=HEADERS,
                json=test_payload,
                timeout=60  # Allow sufficient time for processing
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                answers = data.get("answers", [])
                
                # Check response format
                if isinstance(answers, list) and len(answers) == len(test_payload["questions"]):
                    # Check latency requirement (HackRX: <30 seconds)
                    if elapsed_time < 30:
                        self.log_result(
                            "Basic Functionality", 
                            True, 
                            f"Processed {len(answers)} questions in {elapsed_time:.2f}s"
                        )
                        
                        # Display sample answers
                        print("\n📋 Sample Answers:")
                        for i, (q, a) in enumerate(zip(test_payload["questions"], answers[:2])):
                            print(f"  Q{i+1}: {q}")
                            print(f"  A{i+1}: {a[:100]}{'...' if len(a) > 100 else ''}\n")
                        
                        return True
                    else:
                        self.log_result(
                            "Basic Functionality", 
                            False, 
                            f"Latency too high: {elapsed_time:.2f}s (>30s limit)"
                        )
                else:
                    self.log_result(
                        "Basic Functionality", 
                        False, 
                        f"Invalid response format: expected {len(test_payload['questions'])} answers, got {len(answers) if isinstance(answers, list) else 'non-list'}"
                    )
            else:
                self.log_result(
                    "Basic Functionality", 
                    False, 
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
        
        except Exception as e:
            self.log_result("Basic Functionality", False, f"Error: {str(e)}")
        
        return False
    
    def test_performance_benchmark(self):
        """Test performance with multiple questions."""
        test_payload = {
            "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
            "questions": [
                "What is the sum insured under this policy?",
                "What are the exclusions in this policy?",
                "What is the claim settlement process?",
                "What documents are required for claims?",
                "What is the policy term and renewal process?"
            ]
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{BASE_URL}/hackrx/run",
                headers=HEADERS,
                json=test_payload,
                timeout=60
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                answers = data.get("answers", [])
                
                if len(answers) == len(test_payload["questions"]):
                    avg_time_per_question = elapsed_time / len(test_payload["questions"])
                    
                    if elapsed_time < 30:  # HackRX requirement
                        self.log_result(
                            "Performance Benchmark", 
                            True, 
                            f"{len(answers)} questions in {elapsed_time:.2f}s ({avg_time_per_question:.2f}s/question)"
                        )
                        return True
                    else:
                        self.log_result(
                            "Performance Benchmark", 
                            False, 
                            f"Too slow: {elapsed_time:.2f}s for {len(answers)} questions"
                        )
                else:
                    self.log_result(
                        "Performance Benchmark", 
                        False, 
                        f"Incomplete response: {len(answers)}/{len(test_payload['questions'])} answers"
                    )
            else:
                self.log_result(
                    "Performance Benchmark", 
                    False, 
                    f"HTTP {response.status_code}"
                )
        
        except Exception as e:
            self.log_result("Performance Benchmark", False, f"Error: {str(e)}")
        
        return False
    
    def test_stats_endpoint(self):
        """Test system statistics endpoint."""
        try:
            response = requests.get(f"{BASE_URL}/stats", headers=HEADERS, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                required_keys = ["database_stats", "system_config", "supported_formats"]
                
                if all(key in data for key in required_keys):
                    self.log_result("Stats Endpoint", True, f"Config: {data.get('system_config', {}).get('llm_provider', 'unknown')}")
                    return True
                else:
                    missing = [key for key in required_keys if key not in data]
                    self.log_result("Stats Endpoint", False, f"Missing keys: {missing}")
            else:
                self.log_result("Stats Endpoint", False, f"HTTP {response.status_code}")
        
        except Exception as e:
            self.log_result("Stats Endpoint", False, f"Error: {str(e)}")
        
        return False
    
    def test_error_handling(self):
        """Test error handling with invalid requests."""
        # Test invalid endpoint
        try:
            response = requests.get(f"{BASE_URL}/invalid-endpoint", timeout=5)
            if response.status_code == 404:
                self.log_result("Error Handling - 404", True, "Correct 404 response")
            else:
                self.log_result("Error Handling - 404", False, f"Expected 404, got {response.status_code}")
        except Exception as e:
            self.log_result("Error Handling - 404", False, f"Error: {str(e)}")
        
        # Test invalid JSON payload
        try:
            response = requests.post(
                f"{BASE_URL}/hackrx/run",
                headers=HEADERS,
                json={"invalid": "payload"},
                timeout=10
            )
            if response.status_code in [400, 422]:  # Bad request or validation error
                self.log_result("Error Handling - Invalid Payload", True, f"Correctly rejected invalid payload")
            else:
                self.log_result("Error Handling - Invalid Payload", False, f"Expected 400/422, got {response.status_code}")
        except Exception as e:
            self.log_result("Error Handling - Invalid Payload", False, f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("🧪 Starting HackRX 6.0 System Test Suite")
        print("=" * 50)
        
        # Basic connectivity tests
        print("\n🔍 Testing Basic Connectivity...")
        self.test_health_check()
        self.test_root_endpoint()
        
        # Authentication tests
        print("\n🔐 Testing Authentication...")
        self.test_authentication()
        
        # Core functionality tests
        print("\n⚡ Testing Core Functionality...")
        self.test_basic_functionality()
        
        # Performance tests
        print("\n🚀 Testing Performance...")
        self.test_performance_benchmark()
        
        # Additional endpoints
        print("\n📊 Testing Additional Endpoints...")
        self.test_stats_endpoint()
        
        # Error handling tests
        print("\n❌ Testing Error Handling...")
        self.test_error_handling()
        
        # Final results
        print("\n" + "=" * 50)
        print(f"🎯 TEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"✅ Passed: {self.passed_tests}/{self.total_tests} tests")
        print(f"❌ Failed: {self.total_tests - self.passed_tests}/{self.total_tests} tests")
        
        if self.passed_tests == self.total_tests:
            print("\n🏆 ALL TESTS PASSED! System is HackRX 6.0 ready! 🎉")
        else:
            print(f"\n⚠️  {self.total_tests - self.passed_tests} tests failed. Please check the issues above.")
        
        # Success rate
        success_rate = (self.passed_tests / self.total_tests) * 100
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        return self.passed_tests == self.total_tests


def check_server_running():
    """Check if the server is running."""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=3)
        return response.status_code == 200
    except:
        return False


def main():
    """Main test execution."""
    print("🚀 HackRX 6.0 System Test Suite")
    print("================================")
    
    # Check if server is running
    if not check_server_running():
        print("❌ Server not running! Please start the server first:")
        print("   python main.py")
        print("\nOr run in background:")
        print("   python main.py &")
        return False
    
    print("✅ Server is running, starting tests...\n")
    
    # Run tests
    tester = HackRXTester()
    success = tester.run_all_tests()
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
