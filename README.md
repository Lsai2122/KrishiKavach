# KrishiKavach AI Farming System

An AI-powered farming decision support system that provides crop recommendations, fertilizer guidance, price analysis, and production estimation.

## Features

- 🌱 **Crop Recommendation**: Get personalized crop suggestions based on soil and climate conditions
- 🧪 **Fertilizer Guide**: Optimize fertilizer usage with AI-powered recommendations  
- 📈 **Price Analysis**: Analyze market prices and risks to maximize profits
- 🚜 **Production Estimator**: Estimate crop production based on field conditions

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Application
```bash
python run.py
```

### 3. Access the Application
Open your browser and go to: http://localhost:5000

## API Endpoints

### Health Check
- `GET /health` - Check system status and loaded models

### API Information
- `GET /api/info` - Get API documentation and available features

### Predictions
- `POST /predict_crop` - Get crop recommendations
- `POST /predict_fertilizer` - Get fertilizer recommendations
- `POST /predict_price` - Analyze market prices
- `POST /predict_production` - Estimate crop production

## API Usage Examples

### Crop Recommendation
```bash
curl -X POST http://localhost:5000/predict_crop \
  -H "Content-Type: application/json" \
  -d '{
    "nitrogen": 90,
    "phosphorus": 42,
    "potassium": 43,
    "temperature": 20.87,
    "humidity": 82,
    "ph": 6.5,
    "rainfall": 202.93
  }'
```

### Fertilizer Recommendation
```bash
curl -X POST http://localhost:5000/predict_fertilizer \
  -H "Content-Type: application/json" \
  -d '{
    "crop_year": 2023,
    "area": 2.5,
    "annual_rainfall": 1200,
    "nitrogen": 90,
    "phosphorus": 42,
    "potassium": 43
  }'
```

### Price Analysis
```bash
curl -X POST http://localhost:5000/predict_price \
  -H "Content-Type: application/json" \
  -d '{
    "current_price": 25.5,
    "quantity": 1000,
    "storage_cost": 0.5,
    "daily_loss": 0.1,
    "interest_rate": 8.5
  }'
```

### Production Estimation
```bash
curl -X POST http://localhost:5000/predict_production \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Input Validation

All endpoints include comprehensive input validation:
- Required field validation
- Numeric range validation
- Data type validation
- Error messages with helpful guidance

## Model Status

The application gracefully handles missing models:
- ✅ **Crop Model**: Available (crop_recommendation_model.pkl)
- ⚠️ **Fertilizer Model**: Uses mock predictions if not available
- ⚠️ **Price Model**: Uses mock predictions if not available  
- ⚠️ **Production Model**: Uses mock predictions if not available

To train additional models, see the `model.py` file.

## File Structure

```
ml/crop-prediction/
├── app.py                    # Main Flask application
├── run.py                     # Startup script with checks
├── requirements.txt           # Python dependencies
├── model.py                   # Model training scripts
├── crop_recommendation_model.pkl    # Trained crop model
├── label_encoder_crop.pkl     # Crop label encoder
├── label_encoder_state.pkl    # State label encoder
├── feature_info.json          # Feature information
├── Crop_recommendation.csv    # Training data
└── templates/                 # HTML templates
```

## Development

### Training Models
If you want to train additional models:
```bash
python model.py
```

### Adding New Features
1. Add the endpoint in `app.py`
2. Update the API documentation in `/api/info`
3. Add corresponding HTML template if needed
4. Update input validation and error handling

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in `app.py` or kill the existing process
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Model loading errors**: Check that model files exist in the correct location

### Debug Mode
The application runs in debug mode by default. For production deployment:
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

## Support

For issues and questions:
- Check the health endpoint: http://localhost:5000/health
- Review API documentation: http://localhost:5000/api/info
- Check console logs for detailed error messages