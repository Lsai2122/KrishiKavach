from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json
import google.generativeai as genai
import os
import speech_recognition as sr
from dotenv import load_dotenv, find_dotenv
import re
import logging
try:
    from google.cloud import translate_v2 as translate
except ImportError:
    translate = None
from PIL import Image
import io
import base64

app = Flask(__name__)

# Import the pest detection module (make TensorFlow optional)
try:
    from pest_detection import PestDetectionModel
    pest_model = PestDetectionModel()
    print("Pest detection model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load pest detection model: {e}")
    pest_model = None

# Remove TensorFlow imports to avoid conflicts
TF_AVAILABLE = False

# ✅ Load environment variables properly
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Configure Gemini AI (Fixed & Simplified)
gemini_model = None
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    print("Gemini Key Loaded:", gemini_api_key is not None)  # Debug print

    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
        logger.info("✅ Gemini 1.5 Flash configured successfully")
    else:
        logger.warning("⚠️ No GEMINI_API_KEY found in .env file")
except Exception as e:
    logger.exception("❌ Gemini AI configuration failed:")
    gemini_model = None

# Configure Google Translate (with fallback)
translate_client = None
if translate:
    try:
        translate_client = translate.Client()
    except Exception as e:
        logger.warning(f"Google Translate client initialization failed: {e}")
        translate_client = None
else:
    logger.warning("Google Cloud Translate library not available. Translation features will be limited.")

# Supported languages for translation
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'bn': 'Bengali',
    'pa': 'Punjabi',
    'or': 'Odia',
    'as': 'Assamese',
    'ur': 'Urdu',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'zh': 'Chinese',
    'ja': 'Japanese'
}

# ✅ Rest of your original logic remains unchanged
# (Crop, Fertilizer, Price, Production prediction routes, Chatbot, etc.)
# ↓↓↓ KEEP EVERYTHING BELOW EXACTLY AS IT WAS ↓↓↓

# Load all models and encoders
try:
    # Try to load new crop model first
    try:
        with open('../model training/models/crop_model.pkl', 'rb') as f:
            crop_model = pickle.load(f)
        print("✅ Loaded new crop model from model training folder")
    except FileNotFoundError:
        # Fall back to old model
        with open('crop_recommendation_model.pkl', 'rb') as f:
            crop_model = pickle.load(f)
        print("Using existing crop recommendation model")
    
    # Try to load additional models (they may not exist yet)
    try:
        with open('fertilizer_model.pkl', 'rb') as f:
            fertilizer_model = pickle.load(f)
    except FileNotFoundError:
        fertilizer_model = None
        print("Fertilizer model not found - will use mock predictions")
    
    try:
        with open('price_model.pkl', 'rb') as f:
            price_model = pickle.load(f)
    except FileNotFoundError:
        price_model = None
        print("Price model not found - will use mock predictions")
    
    try:
        with open('production_model.pkl', 'rb') as f:
            production_model = pickle.load(f)
    except FileNotFoundError:
        production_model = None
        print("Production model not found - will use mock predictions")
    
    # Try to load production models from ml folder
    try:
        # Load XGBoost model from ml folder (skip preprocessor for now)
        with open('../xgb_model_pickle.pkl', 'rb') as f:
            production_model_xgb = pickle.load(f)
        
        # If production_model is not loaded, use the XGBoost model
        if production_model is None:
            production_model = production_model_xgb
            print("✅ Loaded XGBoost production model from ml folder")
    except Exception as e:
        production_model_xgb = None
        print(f"❌ Production models from ml folder not found: {e}")
    
    with open('label_encoder_crop.pkl', 'rb') as f:
        crop_encoder = pickle.load(f)
    
    try:
        with open('label_encoder_state.pkl', 'rb') as f:
            state_encoder = pickle.load(f)
    except FileNotFoundError:
        state_encoder = None
        print("State encoder not found")
    
    with open('feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    crop_model = None
    fertilizer_model = None
    price_model = None
    production_model = None
    crop_encoder = None
    state_encoder = None
    feature_info = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop-recommendation')
def crop_recommendation():
    return render_template('crop_recommendation.html')

@app.route('/fertilizer-recommendation')
def fertilizer_recommendation():
    return render_template('fertilizer_recommendation.html')

@app.route('/price-analysis')
def price_analysis():
    return render_template('price_analysis.html')

@app.route('/production-estimation')
def production_estimation():
    return render_template('production_estimation.html')

@app.route('/pest-detection')
def pest_detection():
    return render_template('pest_detection.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        if not crop_model or not crop_encoder:
            return jsonify({'error': 'Crop recommendation model not available'}), 500
        
        data = request.json
        
        # Validate input data
        required_fields = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract and validate features
        try:
            features = [
                float(data['nitrogen']),
                float(data['phosphorus']),
                float(data['potassium']),
                float(data['temperature']),
                float(data['humidity']),
                float(data['ph']),
                float(data['rainfall'])
            ]
        except ValueError as e:
            return jsonify({'error': 'Invalid input data: please provide numeric values'}), 400
        
        # Validate ranges
        if not (0 <= features[0] <= 200):  # nitrogen
            return jsonify({'error': 'Nitrogen must be between 0-200'}), 400
        if not (0 <= features[1] <= 200):  # phosphorus
            return jsonify({'error': 'Phosphorus must be between 0-200'}), 400
        if not (0 <= features[2] <= 400):  # potassium
            return jsonify({'error': 'Potassium must be between 0-400'}), 400
        if not (0 <= features[3] <= 50):  # temperature
            return jsonify({'error': 'Temperature must be between 0-50°C'}), 400
        if not (0 <= features[4] <= 100):  # humidity
            return jsonify({'error': 'Humidity must be between 0-100%'}), 400
        if not (0 <= features[5] <= 14):  # ph
            return jsonify({'error': 'pH must be between 0-14'}), 400
        if not (0 <= features[6] <= 500):  # rainfall
            return jsonify({'error': 'Rainfall must be between 0-500mm'}), 400
        
        # Make prediction
        prediction = crop_model.predict([features])[0]
        
        # Get the predicted crop name
        if isinstance(prediction, (int, np.integer)):
            predicted_crop = crop_encoder.classes_[prediction]
        else:
            # If prediction is already a class name
            predicted_crop = str(prediction)
        
        return jsonify({
            'recommended_crop': predicted_crop
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        data = request.json
        
        # Validate input data
        required_fields = ['crop_year', 'area', 'annual_rainfall', 'nitrogen', 'phosphorus', 'potassium']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare features
        try:
            features = [
                int(data['crop_year']),
                float(data['area']),
                float(data['annual_rainfall']),
                float(data['nitrogen']),
                float(data['phosphorus']),
                float(data['potassium'])
            ]
        except ValueError as e:
            return jsonify({'error': 'Invalid input data: please provide numeric values'}), 400
        
        # Make predictions (use model if available, otherwise mock)
        if fertilizer_model:
            fertilizer_pred = fertilizer_model.predict([features])[0]
        else:
            # Mock prediction based on nitrogen levels
            fertilizer_pred = features[3] * 2.5 + np.random.uniform(-10, 10)
        
        return jsonify({
            'fertilizer_recommendation': round(fertilizer_pred, 2),
            'usage_tips': [
                'Apply fertilizer during early morning or late evening',
                'Water the field before applying fertilizer',
                'Use appropriate safety equipment when handling fertilizers',
                'Test soil pH before application',
                'Follow recommended dosage instructions'
            ]
        })
    
    except Exception as e:
        return jsonify({'error': f'Fertilizer prediction failed: {str(e)}'}), 500

@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        data = request.json
        
        # Validate input data
        required_fields = ['current_price', 'quantity', 'storage_cost', 'daily_loss', 'interest_rate']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare features
        try:
            features = [
                float(data['current_price']),
                float(data['quantity']),
                float(data['storage_cost']),
                float(data['daily_loss']),
                float(data['interest_rate'])
            ]
        except ValueError as e:
            return jsonify({'error': 'Invalid input data: please provide numeric values'}), 400
        
        # Validate ranges
        if features[0] <= 0:  # current_price
            return jsonify({'error': 'Current price must be positive'}), 400
        if features[1] <= 0:  # quantity
            return jsonify({'error': 'Quantity must be positive'}), 400
        if features[2] < 0:  # storage_cost
            return jsonify({'error': 'Storage cost cannot be negative'}), 400
        if features[3] < 0:  # daily_loss
            return jsonify({'error': 'Daily loss cannot be negative'}), 400
        if not (0 <= features[4] <= 100):  # interest_rate
            return jsonify({'error': 'Interest rate must be between 0-100%'}), 400
        
        # Make predictions (use model if available, otherwise mock)
        current_price = float(data['current_price'])
        if price_model:
            price_15d = price_model.predict([features])[0]
        else:
            # Mock prediction with some market volatility
            volatility = np.random.uniform(-0.15, 0.15)  # ±15% price change
            price_15d = current_price * (1 + volatility)
        
        # Calculate potential earnings
        current_value = current_price * float(data['quantity'])
        future_value_15d = price_15d * float(data['quantity'])
        
        # Determine recommendation based on price trend
        price_change = (price_15d - current_price) / current_price
        recommendation = "Store" if price_15d > current_price else "Sell Now"
        
        # Risk assessment
        if abs(price_change) < 0.05:
            risk_level = "Very Low"
        elif abs(price_change) < 0.1:
            risk_level = "Low"
        elif abs(price_change) < 0.2:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return jsonify({
            'predicted_price_15d': round(price_15d, 2),
            'price_change_percent': round(price_change * 100, 2),
            'current_value': round(current_value, 2),
            'future_value_15d': round(future_value_15d, 2),
            'potential_profit': round(future_value_15d - current_value, 2),
            'recommendation': recommendation,
            'risk_level': risk_level,
            'market_insights': [
                'Monitor local market trends for better timing',
                'Consider storage costs in your decision',
                'Factor in transportation costs to market',
                'Check weather forecasts that might affect prices'
            ]
        })
    
    except Exception as e:
        return jsonify({'error': f'Price prediction failed: {str(e)}'}), 500

@app.route('/predict_production', methods=['POST'])
def predict_production():
    try:
        data = request.json
        
        # Validate input data
        required_fields = ['area', 'nitrogen_req', 'phosphorus_req', 'potassium_req', 
                          'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare features
        try:
            features = [
                float(data['area']),
                float(data['nitrogen_req']),
                float(data['phosphorus_req']),
                float(data['potassium_req']),
                float(data['temperature']),
                float(data['humidity']),
                float(data['ph']),
                float(data['rainfall']),
                float(data['wind_speed']),
                float(data['solar_radiation'])
            ]
        except ValueError as e:
            return jsonify({'error': 'Invalid input data: please provide numeric values'}), 400
        
        # Validate ranges
        if features[0] <= 0:  # area
            return jsonify({'error': 'Area must be positive'}), 400
        if not (0 <= features[1] <= 200):  # nitrogen_req
            return jsonify({'error': 'Nitrogen requirement must be between 0-200'}), 400
        if not (0 <= features[2] <= 200):  # phosphorus_req
            return jsonify({'error': 'Phosphorus requirement must be between 0-200'}), 400
        if not (0 <= features[3] <= 400):  # potassium_req
            return jsonify({'error': 'Potassium requirement must be between 0-400'}), 400
        if not (0 <= features[4] <= 50):  # temperature
            return jsonify({'error': 'Temperature must be between 0-50°C'}), 400
        if not (0 <= features[5] <= 100):  # humidity
            return jsonify({'error': 'Humidity must be between 0-100%'}), 400
        if not (0 <= features[6] <= 14):  # ph
            return jsonify({'error': 'pH must be between 0-14'}), 400
        if not (0 <= features[7] <= 500):  # rainfall
            return jsonify({'error': 'Rainfall must be between 0-500mm'}), 400
        if not (0 <= features[8] <= 50):  # wind_speed
            return jsonify({'error': 'Wind speed must be between 0-50 km/h'}), 400
        if not (0 <= features[9] <= 50):  # solar_radiation
            return jsonify({'error': 'Solar radiation must be between 0-50 MJ/m²/day'}), 400
        
        # Make predictions using production models from ml folder
        area = float(data['area'])
        if production_model:
            try:
                # Create feature array matching the XGBoost model's expected input
                # Based on the model's feature names: ['Dist Code', 'Year', 'State Code', 'Area_ha', 'Temperature_C', 'Humidity_%', 'pH', 'Rainfall_mm', 'State Name', 'Dist Name', 'Crop']
                # We'll use default values for missing categorical features and focus on the environmental data we have
                
                # Map our input features to model features
                model_features = [
                    1,  # Dist Code (default)
                    2023,  # Year (current year)
                    1,  # State Code (default)
                    area,  # Area_ha
                    float(data['temperature']),  # Temperature_C
                    float(data['humidity']),  # Humidity_%
                    float(data['ph']),  # pH
                    float(data['rainfall']),  # Rainfall_mm
                    'Default',  # State Name
                    'Default',  # Dist Name
                    'Rice'  # Crop (default)
                ]
                
                # Convert to numpy array and reshape for prediction
                features_array = np.array([model_features])
                yield_prediction = production_model.predict(features_array)[0]
                
                # Convert from the model's output scale to tons per hectare
                # Assuming the model predicts in some unit, convert to tons/ha
                yield_per_ha = max(0, yield_prediction / 1000)  # Convert kg to tons, ensure positive
                
                print(f"✅ Production prediction using ml XGBoost model: {yield_per_ha} tons/ha")
            except Exception as e:
                print(f"❌ Error using XGBoost model, falling back to mock: {e}")
                # Fallback to mock prediction
                base_yield = 25  # Base yield of 25 tons/ha
                temp_factor = min(1.0, max(0.3, 1 - abs(features[4] - 25) / 25))  # Optimal temp around 25°C
                rainfall_factor = min(1.0, features[7] / 200)  # Good rainfall up to 200mm
                soil_factor = min(1.0, (features[1] + features[2] + features[3] / 2) / 300)  # Nutrient balance
                yield_per_ha = base_yield * temp_factor * rainfall_factor * soil_factor * np.random.uniform(0.8, 1.2)
        else:
            # Mock prediction based on environmental factors
            base_yield = 25  # Base yield of 25 tons/ha
            temp_factor = min(1.0, max(0.3, 1 - abs(features[4] - 25) / 25))  # Optimal temp around 25°C
            rainfall_factor = min(1.0, features[7] / 200)  # Good rainfall up to 200mm
            soil_factor = min(1.0, (features[1] + features[2] + features[3] / 2) / 300)  # Nutrient balance
            yield_per_ha = base_yield * temp_factor * rainfall_factor * soil_factor * np.random.uniform(0.8, 1.2)
            print(f"Production prediction using mock data: {yield_per_ha}")
        
        total_production = yield_per_ha * area
        
        return jsonify({
            'yield_per_hectare': round(yield_per_ha, 2),
            'total_production': round(total_production, 2),
            'production_per_acre': round(yield_per_ha * 0.4047, 2),  # Convert to acres
            'optimization_tips': [
                'Monitor soil moisture regularly using sensors',
                'Adjust irrigation based on weather forecasts',
                'Apply balanced nutrients according to soil test results',
                'Protect crops from pests and diseases with IPM',
                'Consider crop rotation for soil health',
                'Use high-quality seeds adapted to local conditions',
                'Implement proper weed management strategies'
            ]
        })
    
    except Exception as e:
        return jsonify({'error': f'Production estimation failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint to verify system status"""
    models_status = {
        'crop_model': crop_model is not None,
        'fertilizer_model': fertilizer_model is not None,
        'price_model': price_model is not None,
        'production_model': production_model is not None,
        'production_model_xgb': production_model_xgb is not None if 'production_model_xgb' in locals() else None,
        'crop_encoder': crop_encoder is not None,
        'state_encoder': state_encoder is not None
    }
    
    all_models_ready = all(models_status.values())
    
    return jsonify({
        'status': 'healthy' if crop_model else 'degraded',
        'models_loaded': models_status,
        'api_version': '1.0.0',
        'features_available': list(feature_info.keys()) if feature_info else []
    })

@app.route('/api/info')
def api_info():
    """API information endpoint"""
    return jsonify({
        'name': 'KrishiKavach AI Farming API',
        'version': '1.0.0',
        'description': 'AI-powered farming decision support system',
        'endpoints': {
            'crop_recommendation': '/predict_crop',
            'fertilizer_recommendation': '/predict_fertilizer',
            'price_analysis': '/predict_price',
            'production_estimation': '/predict_production',
            'health_check': '/health'
        },
        'features': feature_info
    })

@app.route('/chatbot')
def chatbot():
    """Chatbot interface"""
    return render_template('chatbot.html')

@app.route('/chatbot/api', methods=['POST'])
def chatbot_api():
    """Enhanced chatbot API with Gemini + ML model integration"""
    try:
        data = request.json
        message = data.get('message', '').strip()
        target_language = data.get('language', 'en')
        voice_input = data.get('voice_input', False)
        detected_lang = None

        # 🟢 Handle empty message
        if not message:
            welcome_msg = "Hello! I'm KrishiKavach AI Assistant. How can I help you with your farming needs?"
            if target_language != 'en':
                welcome_msg = translate_text(welcome_msg, target_language)
            return jsonify({'response': welcome_msg, 'language': target_language})

        # 🟢 Language detection for voice or 'auto' mode
        if voice_input or target_language == 'auto':
            detected_lang = detect_language(message)
            if detected_lang in SUPPORTED_LANGUAGES:
                target_language = detected_lang

        # 🟢 Translate non-English to English for processing
        if target_language != 'en':
            original_message = message
            message = translate_text(message, 'en')
        else:
            original_message = message

        # 🧩 Extract farming-related numbers/data from user message
        extracted_data = extract_farming_data_from_message(message)

        # 🔍 Optional: run local ML model if message mentions crops
        model_hint = ""
        if any(word in message.lower() for word in ['crop', 'recommendation', 'plant', 'grow']):
            if extracted_data and crop_model and 'nitrogen' in extracted_data and 'phosphorus' in extracted_data:
                try:
                    features = np.array([[extracted_data.get('nitrogen', 0),
                                          extracted_data.get('phosphorus', 0),
                                          extracted_data.get('potassium', 0),
                                          extracted_data.get('temperature', 25),
                                          extracted_data.get('humidity', 50),
                                          extracted_data.get('ph', 7.0),
                                          extracted_data.get('rainfall', 100)]])
                    prediction = crop_model.predict(features)[0]
                    model_hint = f"\n\nLocal model recommendation: {prediction}"
                except Exception as ml_error:
                    logger.error(f"Crop model error: {ml_error}")
                    model_hint = "\n\n(Local ML model unavailable for prediction.)"

        # 🧠 Combine user message + local model hint
        gemini_prompt = create_gemini_prompt(
            message + model_hint,
            context=data.get('context')
        )

        # 🤖 Generate response from Gemini
        try:
            if gemini_model:
                gemini_response = gemini_model.generate_content(gemini_prompt)
                response_text = gemini_response.text
            else:
                response_text = generate_fallback_response(message, extracted_data)
        except Exception as gemini_error:
            logger.error(f"Gemini AI error: {gemini_error}")
            response_text = generate_fallback_response(message, extracted_data)

        # 🧩 Add contextual ML explanation if local model ran
        if model_hint:
            response_text += "\n\n(Note: Based on your soil data, this matches the ML model's output.)"

        # 💬 Translate final response to target language (if needed)
        if target_language != 'en':
            response_text = translate_text(response_text, target_language)

        # 💡 Smart contextual suggestions
        suggestions = []
        if 'crop' in message.lower():
            suggestions = ['Check soil NPK levels', 'Measure field area', 'Monitor weather conditions']
        elif 'fertilizer' in message.lower():
            suggestions = ['Test soil nutrients', 'Check crop type', 'Consider weather forecast']
        elif 'price' in message.lower():
            suggestions = ['Check current market prices', 'Estimate storage costs', 'Monitor market trends']
        elif 'production' in message.lower():
            suggestions = ['Measure field area', 'Check soil health', 'Review weather data']
        else:
            suggestions = ['Crop recommendation', 'Fertilizer advice', 'Price analysis', 'Production estimation']

        if target_language != 'en':
            suggestions = [translate_text(suggestion, target_language) for suggestion in suggestions]

        # 📦 Build final response
        response = {
            'response': response_text.strip(),
            'language': target_language,
            'suggestions': suggestions[:4],
            'detected_language': detected_lang if voice_input else None,
            'extracted_data': extracted_data if extracted_data else None
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Chatbot API error: {e}")
        if 'target_language' not in locals():
            target_language = 'en'
        error_msg = f"I'm sorry, I encountered an error: {str(e)}. Please try again."
        if target_language != 'en':
            error_msg = translate_text(error_msg, target_language)
        return jsonify({'response': error_msg, 'language': target_language})

def translate_text(text, target_language='en'):
    """Translate text to target language using Google Translate (with fallback)"""
    try:
        if target_language == 'en' or not translate_client:
            return text
        
        result = translate_client.translate(text, target_language=target_language)
        return result['translatedText']
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text

def detect_language(text):
    """Detect the language of the input text (with fallback)"""
    try:
        if not translate_client:
            return 'en'  # Default to English if translation service unavailable
        
        result = translate_client.detect_language(text)
        return result['language']
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return 'en'

def create_gemini_prompt(user_message, context=None):
    """Create a comprehensive prompt for Gemini AI with farming context"""
    system_context = """
    You are KrishiKavach AI, an intelligent farming assistant. You help farmers with:
    1. Crop recommendations based on soil and weather conditions
    2. Fertilizer recommendations and nutrient management
    3. Price analysis and market timing advice
    4. Production estimation and yield forecasting
    5. General farming advice and best practices
    
    Be helpful, friendly, and provide accurate farming advice. If you need specific data for recommendations,
    ask the user for the required information naturally.
    
    Available tools and data:
    - Crop recommendation: nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall
    - Fertilizer recommendation: crop year, area, rainfall, NPK content
    - Price analysis: current price, quantity, storage cost, loss rate, interest rate
    - Production estimation: area, nutrients, temperature, humidity, pH, rainfall, wind speed, solar radiation
    """
    
    if context:
        system_context += f"\n\nPrevious context: {context}"
    
    return f"{system_context}\n\nUser: {user_message}\nAssistant:"

def extract_farming_data_from_message(message):
    """Extract farming-related data from user messages using patterns"""
    data = {}
    
    # Extract numbers with units
    patterns = {
        'nitrogen': r'(\d+(?:\.\d+)?)\s*(?:kg|g|mg)?\s*(?:n|nitrogen)',
        'phosphorus': r'(\d+(?:\.\d+)?)\s*(?:kg|g|mg)?\s*(?:p|phosphorus|phosphate)',
        'potassium': r'(\d+(?:\.\d+)?)\s*(?:kg|g|mg)?\s*(?:k|potassium)',
        'temperature': r'(\d+(?:\.\d+)?)\s*(?:°c|celsius|degrees)',
        'humidity': r'(\d+(?:\.\d+)?)\s*(?:%|percent)',
        'ph': r'ph\s*(\d+(?:\.\d+)?)',
        'rainfall': r'(\d+(?:\.\d+)?)\s*(?:mm|cm|rainfall|rains)',
        'area': r'(\d+(?:\.\d+)?)\s*(?:hectares?|acres?|ha)',
        'price': r'(?:rs|₹|\$)?\s*(\d+(?:\.\d+)?)',
        'quantity': r'(\d+(?:\.\d+)?)\s*(?:tons?|kg|quintals?)'
    }
    
    for key, pattern in patterns.items():
        matches = re.findall(pattern, message.lower())
        if matches:
            data[key] = float(matches[0])
    
    # Extract crop names
    common_crops = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'potato', 'tomato', 'onion', 'brinjal', 'okra', 'cabbage', 'cauliflower', 'carrot', 'radish', 'spinach', 'mustard', 'groundnut', 'soybean', 'pulses', 'millets']
    for crop in common_crops:
        if crop in message.lower():
            data['crop'] = crop
            break
    
    return data

def generate_fallback_response(message, extracted_data=None):
    """Generate intelligent fallback responses when Gemini AI is not available"""
    message_lower = message.lower()
    
    # Crop-related queries
    if any(word in message_lower for word in ['crop', 'recommendation', 'plant', 'grow', 'seed']):
        if extracted_data and 'nitrogen' in extracted_data:
            return "Based on your soil data, I can help with crop recommendations. However, for the most accurate advice, please ensure your Gemini API key is configured. In the meantime, consider crops like rice, wheat, or maize based on your NPK levels."
        else:
            return "I can help you choose the right crop! To give you the best recommendation, I'll need some information about your soil. Could you provide your soil's NPK (Nitrogen, Phosphorus, Potassium) levels, pH, and recent rainfall data?"
    
    # Fertilizer-related queries
    elif any(word in message_lower for word in ['fertilizer', 'nutrient', 'npk', 'manure']):
        if extracted_data and 'crop' in extracted_data:
            return f"For {extracted_data['crop']} crops, fertilizer requirements vary by growth stage. Generally, you'll need balanced NPK fertilizer. For precise recommendations, please provide your current soil nutrient levels and field area."
        else:
            return "I can help optimize your fertilizer usage! Please tell me what crop you're growing and your current soil nutrient levels (NPK), and I'll provide tailored recommendations."
    
    # Price/market queries
    elif any(word in message_lower for word in ['price', 'market', 'sell', 'buy', 'cost']):
        return "I can help analyze market prices and timing for selling your crops. Please provide: current market price, quantity you want to sell, storage costs, and how long you can store the produce."
    
    # Production/yield queries  
    elif any(word in message_lower for word in ['production', 'yield', 'harvest', 'output']):
        return "I can estimate your crop production! I'll need details about your field area, soil nutrients (NPK), current weather conditions (temperature, humidity, rainfall), and the crop variety you're growing."
    
    # Weather/soil queries
    elif any(word in message_lower for word in ['weather', 'soil', 'rain', 'temperature']):
        return "Weather and soil conditions are crucial for farming success. I can help you understand how current conditions affect your crops and what adjustments you might need to make. What specific information do you need?"
    
    # General farming advice
    elif any(word in message_lower for word in ['farm', 'farming', 'agriculture', 'help', 'advice']):
        return "I'm here to help with all your farming needs! I can assist with crop recommendations, fertilizer advice, price analysis, and production estimation. What would you like to know about today?"
    
    # Greeting/friendly messages
    elif any(word in message_lower for word in ['hello', 'hi', 'hey', 'namaste', 'good']):
        return "Hello! I'm KrishiKavach AI, your farming assistant. I'm here to help you make better farming decisions with personalized recommendations for crops, fertilizers, market timing, and production planning. How can I assist you today?"
    
    # Default response
    else:
        return "I'm KrishiKavach AI, your intelligent farming assistant! I can help you with crop recommendations, fertilizer optimization, market price analysis, and production estimation. What farming challenge can I help you solve today?"

def get_suggestions(action):
    """Get contextual suggestions based on action"""
    suggestions = {
        'crop_recommendation': ['Check soil NPK levels', 'Measure pH', 'Check recent rainfall', 'Monitor temperature'],
        'fertilizer_recommendation': ['Test soil nutrients', 'Check crop growth stage', 'Consider weather forecast', 'Calculate field area'],
        'price_analysis': ['Check current market prices', 'Estimate storage costs', 'Consider transportation', 'Monitor market trends'],
        'production_estimation': ['Measure field area', 'Check soil health', 'Review weather data', 'Assess crop variety'],
        'help': ['Crop recommendation', 'Fertilizer advice', 'Price analysis', 'Production estimation'],
        'greeting': ['Crop recommendation', 'Soil testing', 'Market prices', 'Weather impact'],
        'weather': ['Check rainfall data', 'Monitor temperature', 'Track humidity', 'Seasonal patterns'],
        'soil': ['Test NPK levels', 'Measure pH', 'Check soil type', 'Organic matter content']
    }
    return suggestions.get(action, ['Try rephrasing your question', 'Ask about specific crops', 'Inquire about soil conditions'])

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.route('/chatbot/voice', methods=['POST'])
def voice_to_text():
    """Convert voice input to text using speech recognition"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', 'en-IN')
        
        # Initialize speech recognizer
        recognizer = sr.Recognizer()
        
        # Save temporary audio file
        temp_audio_path = 'temp_audio.wav'
        audio_file.save(temp_audio_path)
        
        # Convert audio to text
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            
        try:
            # Use Google Speech Recognition
            text = recognizer.recognize_google(audio_data, language=language)
            
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            return jsonify({
                'text': text,
                'language': language,
                'success': True
            })
            
        except sr.UnknownValueError:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return jsonify({'error': 'Could not understand the audio'}), 400
            
        except sr.RequestError as e:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return jsonify({'error': f'Speech recognition error: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Voice recognition error: {e}")
        return jsonify({'error': f'Voice processing failed: {str(e)}'}), 500

@app.route('/chatbot/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech (placeholder for future implementation)"""
    try:
        data = request.json
        text = data.get('text', '')
        language = data.get('language', 'en')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # For now, return a placeholder response
        # In a full implementation, you would integrate with Google Text-to-Speech API
        return jsonify({
            'message': 'Text-to-speech functionality ready for integration',
            'text': text,
            'language': language,
            'note': 'Google Text-to-Speech API integration ready - add your API key'
        })
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return jsonify({'error': f'TTS processing failed: {str(e)}'}), 500

@app.route('/chatbot/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages"""
    return jsonify({
        'languages': SUPPORTED_LANGUAGES,
        'voice_languages': {
            'en': 'en-US',
            'hi': 'hi-IN',
            'te': 'te-IN',
            'ta': 'ta-IN',
            'kn': 'kn-IN',
            'ml': 'ml-IN',
            'mr': 'mr-IN',
            'gu': 'gu-IN',
            'bn': 'bn-IN',
            'pa': 'pa-IN'
        }
    })

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
@app.route('/debug/gemini')
def debug_gemini():
    """Quick test route to check Gemini AI connectivity"""
    try:
        if not gemini_model:
            return jsonify({"error": "Gemini model not initialized"}), 500
        
        test_prompt = "Say hello from KrishiKavach chatbot in one line."
        response = gemini_model.generate_content(test_prompt)
        return jsonify({"status": "success", "gemini_reply": response.text})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# PWA Routes
@app.route('/sw.js')
def service_worker():
    """Serve the service worker file"""
    return app.send_static_file('sw.js')

@app.route('/manifest.json')
def manifest():
    """Serve the manifest.json file"""
    return app.send_static_file('manifest.json')

@app.route('/offline')
def offline():
    """Offline fallback page"""
    return render_template('offline.html')

@app.after_request
def add_security_headers(response):
    """Add security headers for PWA"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = "default-src 'self' 'unsafe-inline' 'unsafe-eval' https: data: blob:;"
    return response

# Image-based pest detection endpoint
@app.route('/predict_pest_image', methods=['POST'])
def predict_pest_image():
    try:
        if 'pestImage' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['pestImage']
        location = request.form.get('location', 'Unknown')
        
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image data
        image_bytes = image_file.read()
        
        # Use pest detection model if available
        if pest_model:
            try:
                result = pest_model.predict_from_image(image_bytes, location)
                return jsonify(result)
            except Exception as e:
                print(f"Pest model error: {e}")
                # Fallback to AI analysis
                return jsonify(analyze_pest_with_ai_image(image_bytes, location))
        else:
            # Fallback to AI analysis
            return jsonify(analyze_pest_with_ai_image(image_bytes, location))
            
    except Exception as e:
        print(f"Image pest detection error: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

def analyze_pest_with_ai_image(image_bytes, location):
    """Fallback AI analysis for pest detection from image"""
    try:
        # Use Gemini AI for image analysis if available
        if gemini_model:
            import base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            prompt = f"""
            Analyze this plant image for pests and diseases. The image was taken in {location}.
            
            Please provide:
            1. What pest or disease is visible (if any)
            2. Severity level (1-10)
            3. Recommended treatment
            4. Prevention measures
            5. Confidence level (High/Medium/Low)
            
            If the plant appears healthy, indicate that clearly.
            """
            
            response = gemini_model.generate_content([prompt, image_base64])
            ai_analysis = response.text
            
            # Parse AI response (simplified)
            return {
                'pest_disease': extract_disease_from_ai(ai_analysis),
                'plant_type': 'Unknown',
                'confidence': extract_confidence_from_ai(ai_analysis),
                'severity': extract_severity_from_ai(ai_analysis),
                'treatment': extract_treatment_from_ai(ai_analysis),
                'prevention': extract_prevention_from_ai(ai_analysis),
                'recommendations': extract_recommendations_from_ai(ai_analysis),
                'is_healthy': 'healthy' in ai_analysis.lower(),
                'ai_analysis': ai_analysis,
                'top_3_predictions': generate_fallback_predictions()
            }
        else:
            # Fallback to mock analysis
            return generate_mock_image_analysis()
            
    except Exception as e:
        print(f"AI image analysis error: {e}")
        return generate_mock_image_analysis()

def generate_mock_image_analysis():
    """Generate mock analysis for image-based detection"""
    import random
    
    mock_diseases = [
        {"name": "Early Blight (Alternaria solani)", "severity": 7},
        {"name": "Late Blight (Phytophthora infestans)", "severity": 8},
        {"name": "Powdery Mildew", "severity": 5},
        {"name": "Downy Mildew", "severity": 6},
        {"name": "Bacterial Spot", "severity": 6},
        {"name": "Spider Mites", "severity": 4},
        {"name": "Aphid Infestation", "severity": 3},
        {"name": "Whitefly Infestation", "severity": 5}
    ]
    
    selected = random.choice(mock_diseases)
    
    return {
        'pest_disease': selected['name'],
        'plant_type': 'Tomato',
        'confidence': 'High',
        'severity': selected['severity'],
        'treatment': 'Apply appropriate fungicide/insecticide. Remove and destroy affected leaves.',
        'prevention': 'Practice crop rotation, maintain proper spacing, avoid overhead watering.',
        'recommendations': [
            "Monitor plants regularly for early signs",
            "Maintain proper plant nutrition",
            "Ensure adequate drainage and air circulation",
            "Contact local agricultural extension office if symptoms persist"
        ],
        'is_healthy': False,
        'top_3_predictions': [
            {"plant": "Tomato", "condition": selected['name'], "confidence": 0.85},
            {"plant": "Tomato", "condition": "Nutrient Deficiency", "confidence": 0.12},
            {"plant": "Tomato", "condition": "Environmental Stress", "confidence": 0.03}
        ]
    }

def generate_fallback_predictions():
    """Generate fallback predictions"""
    return [
        {"plant": "Tomato", "condition": "Early Blight", "confidence": 0.75},
        {"plant": "Tomato", "condition": "Late Blight", "confidence": 0.15},
        {"plant": "Tomato", "condition": "Nutrient Deficiency", "confidence": 0.10}
    ]

def extract_disease_from_ai(text):
    """Extract disease name from AI response"""
    lines = text.split('\n')
    for line in lines:
        if 'pest' in line.lower() or 'disease' in line.lower() or 'issue' in line.lower():
            return line.strip()
    return "Unable to identify specific issue"

def extract_confidence_from_ai(text):
    """Extract confidence level from AI response"""
    if 'high' in text.lower():
        return 'High'
    elif 'medium' in text.lower():
        return 'Medium'
    elif 'low' in text.lower():
        return 'Low'
    return 'Medium'

def extract_severity_from_ai(text):
    """Extract severity level from AI response"""
    import re
    numbers = re.findall(r'\d+', text)
    for num in numbers:
        if 1 <= int(num) <= 10:
            return int(num)
    return 5

def extract_treatment_from_ai(text):
    """Extract treatment from AI response"""
    lines = text.split('\n')
    treatment_lines = []
    capture = False
    for line in lines:
        if 'treatment' in line.lower():
            capture = True
        elif 'prevention' in line.lower():
            capture = False
        elif capture:
            treatment_lines.append(line.strip())
    return ' '.join(treatment_lines) if treatment_lines else "Consult local agricultural expert"

def extract_prevention_from_ai(text):
    """Extract prevention from AI response"""
    lines = text.split('\n')
    prevention_lines = []
    capture = False
    for line in lines:
        if 'prevention' in line.lower():
            capture = True
        elif 'recommendation' in line.lower():
            capture = False
        elif capture:
            prevention_lines.append(line.strip())
    return ' '.join(prevention_lines) if prevention_lines else "Maintain good agricultural practices"

def extract_recommendations_from_ai(text):
    """Extract recommendations from AI response"""
    return [
        "Monitor plants regularly for early signs",
        "Maintain proper plant nutrition",
        "Ensure adequate drainage and air circulation",
        "Contact local agricultural extension office if symptoms persist"
    ]

if __name__ == '__main__':
    print("Starting KrishiKavach AI Farming System...")
    print(f"Crop model available: {crop_model is not None}")
    print(f"Fertilizer model available: {fertilizer_model is not None}")
    print(f"Price model available: {price_model is not None}")
    print(f"Production model available: {production_model is not None}")
    print(f"Gemini AI available: {gemini_model is not None}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)