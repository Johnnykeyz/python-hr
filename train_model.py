import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import pickle
import os

def create_sample_dataset():
    """Create synthetic IBM HR Analytics dataset if not available"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(18, 60, n_samples),
        'Department': np.random.choice(['Sales', 'Research & Development', 'Human Resources'], n_samples),
        'DistanceFromHome': np.random.randint(1, 30, n_samples),
        'Education': np.random.randint(1, 6, n_samples),
        'EnvironmentSatisfaction': np.random.randint(1, 5, n_samples),
        'JobInvolvement': np.random.randint(1, 5, n_samples),
        'JobLevel': np.random.randint(1, 6, n_samples),
        'JobSatisfaction': np.random.randint(1, 5, n_samples),
        'MonthlyIncome': np.random.randint(1000, 20000, n_samples),
        'NumCompaniesWorked': np.random.randint(0, 10, n_samples),
        'OverTime': np.random.choice(['Yes', 'No'], n_samples),
        'PercentSalaryHike': np.random.randint(10, 26, n_samples),
        'PerformanceRating': np.random.randint(3, 5, n_samples),
        'StockOptionLevel': np.random.randint(0, 4, n_samples),
        'TotalWorkingYears': np.random.randint(0, 40, n_samples),
        'TrainingTimesLastYear': np.random.randint(0, 7, n_samples),
        'WorkLifeBalance': np.random.randint(1, 5, n_samples),
        'YearsAtCompany': np.random.randint(0, 40, n_samples),
        'YearsInCurrentRole': np.random.randint(0, 20, n_samples),
        'YearsSinceLastPromotion': np.random.randint(0, 15, n_samples),
        'YearsWithCurrManager': np.random.randint(0, 18, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create attrition based on logical rules
    attrition_prob = (
        (df['JobSatisfaction'] <= 2).astype(int) * 0.3 +
        (df['WorkLifeBalance'] <= 2).astype(int) * 0.2 +
        (df['OverTime'] == 'Yes').astype(int) * 0.2 +
        (df['YearsSinceLastPromotion'] > 5).astype(int) * 0.15 +
        (df['MonthlyIncome'] < 5000).astype(int) * 0.15
    )
    
    df['Attrition'] = (np.random.random(n_samples) < attrition_prob).astype(int)
    df['Attrition'] = df['Attrition'].map({0: 'No', 1: 'Yes'})
    
    return df

def train_attrition_model():
    """Train XGBoost model for employee attrition prediction"""
    
    print("=" * 60)
    print("EMPLOYEE ATTRITION PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load or create dataset
    data_path = 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv'
    
    if os.path.exists(data_path):
        print(f"\n✓ Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("\n✓ Creating synthetic dataset (IBM HR Analytics style)")
        df = create_sample_dataset()
        df.to_csv(data_path, index=False)
    
    print(f"  Dataset shape: {df.shape}")
    print(f"  Attrition distribution:\n{df['Attrition'].value_counts()}")
    
    # Select features
    feature_columns = [
        'Age', 'Department', 'DistanceFromHome', 'Education',
        'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
        'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
        'OverTime', 'PercentSalaryHike', 'PerformanceRating',
        'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
        'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ]
    
    X = df[feature_columns].copy()
    y = df['Attrition'].map({'No': 0, 'Yes': 1})
    
    print(f"\n✓ Features: {len(feature_columns)}")
    print(f"  Target distribution: {dict(y.value_counts())}")
    
    # Encode categorical features
    print("\n✓ Encoding categorical features...")
    categorical_columns = ['Department', 'OverTime']
    encoder = LabelEncoder()
    
    encoded_mappings = {}
    for col in categorical_columns:
        X[col] = encoder.fit_transform(X[col])
        encoded_mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    
    print(f"  Encoded columns: {categorical_columns}")
    
    # Scale numerical features
    print("\n✓ Scaling numerical features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ Data split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Testing set: {X_test.shape[0]} samples")
    
    # Train XGBoost model
    print("\n✓ Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\n✓ Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'=' * 60}")
    print("MODEL PERFORMANCE METRICS")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition']))
    
    # Save model and preprocessors
    print("\n✓ Saving model artifacts...")
    
    with open('models/attrition_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("  ✓ attrition_model.pkl")
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("  ✓ scaler.pkl")
    
    encoder_data = {
        'feature_names': feature_columns,
        'categorical_columns': categorical_columns,
        'encodings': encoded_mappings
    }
    with open('models/encoder.pkl', 'wb') as f:
        pickle.dump(encoder_data, f)
    print("  ✓ encoder.pkl")
    
    print(f"\n{'=' * 60}")
    print("✓ MODEL TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'=' * 60}\n")
    
    return model, scaler, encoder_data, accuracy, roc_auc

if __name__ == "__main__":
    train_attrition_model()
