#!/usr/bin/env python3
"""
KrishiKavach AI Farming System - Startup Script
This script provides a convenient way to start the Flask application.
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import flask
        import pandas
        import numpy
        import sklearn
        print("✓ All dependencies are available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies using: pip install -r requirements.txt")
        return False

def check_models():
    """Check if model files exist"""
    models_path = os.path.dirname(os.path.abspath(__file__))
    required_models = [
        'crop_recommendation_model.pkl',
        'label_encoder_crop.pkl',
        'feature_info.json'
    ]
    
    missing_models = []
    for model in required_models:
        model_path = os.path.join(models_path, model)
        if not os.path.exists(model_path):
            missing_models.append(model)
    
    if missing_models:
        print(f"⚠ Warning: Missing model files: {missing_models}")
        print("The application will use mock predictions for missing models.")
        print("To train models, run: python model.py")
    else:
        print("✓ Core models are available")
    
    return True

def main():
    """Main function to start the application"""
    print("=" * 50)
    print("KrishiKavach AI Farming System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check models
    check_models()
    
    print("\nStarting Flask application...")
    print("Access the application at: http://localhost:5000")
    print("API documentation available at: http://localhost:5000/api/info")
    print("Health check available at: http://localhost:5000/health")
    print("=" * 50)
    
    # Start the Flask application
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nShutting down KrishiKavach...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()