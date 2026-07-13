from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any
import os

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="AI-powered microservice for predicting employee attrition",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
model = None
scaler = None
encoder_data = None

@app.on_event("startup")
async def load_models():
    """Load trained model artifacts"""
    global model, scaler, encoder_data
    
    try:
        with open('models/attrition_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('models/encoder.pkl', 'rb') as f:
            encoder_data = pickle.load(f)
        
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        raise

class EmployeeFeatures(BaseModel):
    """Employee feature schema"""
    age: int = Field(..., ge=18, le=65, description="Employee age")
    department: str = Field(..., description="Department name")
    distance_from_home: int = Field(..., ge=0, description="Distance from home (km)")
    education: int = Field(..., ge=1, le=5, description="Education level (1-5)")
    environment_satisfaction: int = Field(..., ge=1, le=4, description="Environment satisfaction (1-4)")
    job_involvement: int = Field(..., ge=1, le=4, description="Job involvement (1-4)")
    job_level: int = Field(..., ge=1, le=5, description="Job level (1-5)")
    job_satisfaction: int = Field(..., ge=1, le=4, description="Job satisfaction (1-4)")
    monthly_income: float = Field(..., ge=0, description="Monthly income")
    num_companies_worked: int = Field(..., ge=0, description="Number of companies worked")
    over_time: str = Field(..., description="Overtime (Yes/No)")
    percent_salary_hike: int = Field(..., ge=0, le=100, description="Salary hike percentage")
    performance_rating: int = Field(..., ge=1, le=4, description="Performance rating (1-4)")
    stock_option_level: int = Field(..., ge=0, le=3, description="Stock option level (0-3)")
    total_working_years: int = Field(..., ge=0, description="Total working years")
    training_times_last_year: int = Field(..., ge=0, description="Training times last year")
    work_life_balance: int = Field(..., ge=1, le=4, description="Work-life balance (1-4)")
    years_at_company: int = Field(..., ge=0, description="Years at company")
    years_in_current_role: int = Field(..., ge=0, description="Years in current role")
    years_since_last_promotion: int = Field(..., ge=0, description="Years since last promotion")
    years_with_curr_manager: int = Field(..., ge=0, description="Years with current manager")

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    probability: float
    risk_level: str
    recommendation: str
    confidence: float

def get_recommendation(probability: float) -> tuple:
    """Generate risk level and recommendation based on probability"""
    if probability < 0.30:
        risk_level = "Low"
        recommendation = "Maintain current engagement strategies. Employee shows low attrition risk."
    elif probability < 0.60:
        risk_level = "Medium"
        recommendation = "Schedule a one-on-one review meeting. Address concerns about job satisfaction or work-life balance."
    else:
        risk_level = "High"
        recommendation = "Immediate action required: Consider salary adjustment, reduce overtime, offer career development opportunities, or flexible work arrangements."
    
    return risk_level, recommendation

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "Employee Attrition Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = all([model is not None, scaler is not None, encoder_data is not None])
    
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_attrition(employee: EmployeeFeatures):
    """
    Predict employee attrition probability
    
    Returns probability, risk level, and recommendations
    """
    if model is None or scaler is None or encoder_data is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Prepare feature dictionary
        features_dict = {
            'Age': employee.age,
            'Department': employee.department,
            'DistanceFromHome': employee.distance_from_home,
            'Education': employee.education,
            'EnvironmentSatisfaction': employee.environment_satisfaction,
            'JobInvolvement': employee.job_involvement,
            'JobLevel': employee.job_level,
            'JobSatisfaction': employee.job_satisfaction,
            'MonthlyIncome': employee.monthly_income,
            'NumCompaniesWorked': employee.num_companies_worked,
            'OverTime': employee.over_time,
            'PercentSalaryHike': employee.percent_salary_hike,
            'PerformanceRating': employee.performance_rating,
            'StockOptionLevel': employee.stock_option_level,
            'TotalWorkingYears': employee.total_working_years,
            'TrainingTimesLastYear': employee.training_times_last_year,
            'WorkLifeBalance': employee.work_life_balance,
            'YearsAtCompany': employee.years_at_company,
            'YearsInCurrentRole': employee.years_in_current_role,
            'YearsSinceLastPromotion': employee.years_since_last_promotion,
            'YearsWithCurrManager': employee.years_with_curr_manager
        }
        
        # Create DataFrame with correct column order
        df = pd.DataFrame([features_dict])
        df = df[encoder_data['feature_names']]
        
        # Encode categorical features
        for col in encoder_data['categorical_columns']:
            if col in df.columns:
                value = df[col].iloc[0]
                if value in encoder_data['encodings'][col]:
                    df[col] = encoder_data['encodings'][col][value]
                else:
                    # Handle unknown categories
                    df[col] = 0
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Predict
        probability = float(model.predict_proba(X_scaled)[0][1])
        confidence = float(max(model.predict_proba(X_scaled)[0]))
        
        # Get recommendation
        risk_level, recommendation = get_recommendation(probability)
        
        return PredictionResponse(
            probability=round(probability, 4),
            risk_level=risk_level,
            recommendation=recommendation,
            confidence=round(confidence, 4)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost Classifier",
        "features": encoder_data['feature_names'],
        "categorical_features": encoder_data['categorical_columns'],
        "n_features": len(encoder_data['feature_names'])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    