#!/usr/bin/env python3
"""
Test script for KrishiKavach API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_crop_prediction():
    """Test the crop prediction endpoint"""
    print("\nTesting crop prediction endpoint...")
    
    test_data = {
        "nitrogen": 90,
        "phosphorus": 42,
        "potassium": 43,
        "temperature": 20.87,
        "humidity": 82,
        "ph": 6.5,
        "rainfall": 202.93
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict_crop", json=test_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_fertilizer_prediction():
    """Test the fertilizer prediction endpoint"""
    print("\nTesting fertilizer prediction endpoint...")
    
    test_data = {
        "crop_year": 2023,
        "area": 2.5,
        "annual_rainfall": 1200,
        "nitrogen": 90,
        "phosphorus": 42,
        "potassium": 43
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict_fertilizer", json=test_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_price_prediction():
    """Test the price prediction endpoint"""
    print("\nTesting price prediction endpoint...")
    
    test_data = {
        "current_price": 25.5,
        "quantity": 1000,
        "storage_cost": 0.5,
        "daily_loss": 0.1,
        "interest_rate": 8.5
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict_price", json=test_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_production_prediction():
    """Test the production prediction endpoint"""
    print("\nTesting production prediction endpoint...")
    
    test_data = {
        "area": 5.0,
        "nitrogen_req": 90,
        "phosphorus_req": 42,
        "potassium_req": 43,
        "temperature": 25,
        "humidity": 70,
        "ph": 6.5,
        "rainfall": 200,
        "wind_speed": 15,
        "solar_radiation": 25
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict_production", json=test_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_api_info():
    """Test the API info endpoint"""
    print("\nTesting API info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/info")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("KrishiKavach API Test Suite")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("API Info", test_api_info),
        ("Crop Prediction", test_crop_prediction),
        ("Fertilizer Prediction", test_fertilizer_prediction),
        ("Price Prediction", test_price_prediction),
        ("Production Prediction", test_production_prediction),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()