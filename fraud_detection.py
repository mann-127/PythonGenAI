"""
Day 19: Fraud Detection ML Workflow
A complete end-to-end example for a 1-hour tutorial
"""
 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, f1_score
)
import warnings
warnings.filterwarnings('ignore')
 
# =============================================================================
# STEP 1: Generate Synthetic Fraud Data
# =============================================================================
print("=" * 60)
print("STEP 1: Data Generation")
print("=" * 60)
 
np.random.seed(42)
n_samples = 10000
fraud_ratio = 0.02  # 2% fraud rate
 
n_fraud = int(n_samples * fraud_ratio)
n_legit = n_samples - n_fraud
 
# Legitimate transactions
legit_data = {
    'amount': np.random.exponential(100, n_legit),
    'hour': np.random.normal(14, 4, n_legit).clip(0, 23),
    'day_of_week': np.random.randint(0, 7, n_legit),
    'distance_from_home': np.random.exponential(10, n_legit),
    'transaction_velocity': np.random.poisson(2, n_legit),
    'merchant_risk_score': np.random.beta(2, 5, n_legit),
    'is_fraud': np.zeros(n_legit)
}
 
# Fraudulent transactions (different patterns)
fraud_data = {
    'amount': np.random.exponential(500, n_fraud),  # Higher amounts
    'hour': np.random.choice([2, 3, 4, 23], n_fraud),  # Late night
    'day_of_week': np.random.randint(0, 7, n_fraud),
    'distance_from_home': np.random.exponential(100, n_fraud),  # Far away
    'transaction_velocity': np.random.poisson(8, n_fraud),  # More frequent
    'merchant_risk_score': np.random.beta(5, 2, n_fraud),  # Riskier merchants
    'is_fraud': np.ones(n_fraud)
}
 
# Combine into DataFrame
df = pd.concat([pd.DataFrame(legit_data), pd.DataFrame(fraud_data)], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
 
print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
print(f"\nFeature summary:")
print(df.describe().round(2))
 
# =============================================================================
# STEP 2: Exploratory Data Analysis
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)
 
print("\nMean values by class:")
print(df.groupby('is_fraud').mean().round(2))
 
print("\nKey insights:")
print("- Fraud transactions have higher amounts on average")
print("- Fraud occurs more at unusual hours")
print("- Fraud shows higher transaction velocity")
print("- Fraud involves riskier merchants")
 
# =============================================================================
# STEP 3: Data Preprocessing
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Data Preprocessing")
print("=" * 60)
 
# Feature engineering
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
df['high_velocity'] = (df['transaction_velocity'] > 5).astype(int)
df['log_amount'] = np.log1p(df['amount'])
 
# Prepare features and target
feature_cols = ['amount', 'hour', 'distance_from_home',
                'transaction_velocity', 'merchant_risk_score',
                'is_night', 'high_velocity', 'log_amount']
 
X = df[feature_cols]
y = df['is_fraud']
 
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Features: {feature_cols}")
 
# =============================================================================
# STEP 4: Model Training
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Model Training")
print("=" * 60)
 
# Model 1: Logistic Regression (baseline)
print("\n--- Logistic Regression ---")
lr_model = LogisticRegression(class_weight='balanced', random_state=42)
lr_model.fit(X_train_scaled, y_train)
 
# Model 2: Random Forest
print("--- Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
 
print("Models trained successfully!")
 
# =============================================================================
# STEP 5: Model Evaluation
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Model Evaluation")
print("=" * 60)
 
def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation for fraud detection"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'='*40}")
    print(f"{model_name} Results")
    print(f"{'='*40}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Legit  Fraud")
    print(f"Actual Legit  {tn:5d}  {fp:5d}")
    print(f"Actual Fraud  {fn:5d}  {tp:5d}")
    
    # Key Metrics
    print(f"\nKey Metrics:")
    print(f"  ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  F1 Score:      {f1_score(y_test, y_pred):.4f}")
    
    # Business Metrics
    print(f"\nBusiness Impact:")
    print(f"  Fraud caught (Recall):     {tp}/{tp+fn} = {tp/(tp+fn):.1%}")
    print(f"  False alarms (FP Rate):    {fp}/{fp+tn} = {fp/(fp+tn):.1%}")
    print(f"  Precision:                 {tp}/{tp+fp} = {tp/(tp+fp):.1%}")
    
    return roc_auc_score(y_test, y_prob)
 
lr_auc = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
rf_auc = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
 
 
# =============================================================================
# STEP 6: Feature Importance
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Feature Importance (Random Forest)")
print("=" * 60)
 
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
 
print("\nFeature Importance Ranking:")
for i, row in importance_df.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"  {row['feature']:25s} {row['importance']:.3f} {bar}")
 
# =============================================================================
# STEP 7: Threshold Optimization
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Threshold Optimization")
print("=" * 60)
 
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
 
# Find threshold for different business objectives
print("\nThreshold Analysis:")
print("-" * 50)
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 50)
 
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    # p = (y_pred_thresh & y_test.values).sum() / max(y_pred_thresh.sum(), 1)
    # r = (y_pred_thresh & y_test.values).sum() / max(y_test.sum(), 1)
    true_positives = (y_pred_thresh * y_test.values).sum() 
    p = true_positives / max(y_pred_thresh.sum(), 1) # Precision
    r = true_positives / max(y_test.sum(), 1)        # Recall
    f1 = 2 * p * r / max(p + r, 0.001)
    print(f"{thresh:>10.1f} {p:>10.2%} {r:>10.2%} {f1:>10.3f}")
 
# =============================================================================
# STEP 8: Production Simulation
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: Production Simulation")
print("=" * 60)
 
def predict_fraud(transaction, model, scaler, threshold=0.5):
    """Simulate production fraud prediction"""
    features = np.array([[
        transaction['amount'],
        transaction['hour'],
        transaction['distance_from_home'],
        transaction['transaction_velocity'],
        transaction['merchant_risk_score'],
        1 if transaction['hour'] >= 22 or transaction['hour'] <= 5 else 0,
        1 if transaction['transaction_velocity'] > 5 else 0,
        np.log1p(transaction['amount'])
    ]])
    
    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0, 1]
    
    return {
        'fraud_probability': prob,
        'is_flagged': prob >= threshold,
        'risk_level': 'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.4 else 'LOW'
    }
 
# Test transactions
test_transactions = [
    {'amount': 50, 'hour': 14, 'distance_from_home': 5,
     'transaction_velocity': 2, 'merchant_risk_score': 0.2,
     'description': 'Normal lunch purchase'},
    
    {'amount': 2000, 'hour': 3, 'distance_from_home': 500,
     'transaction_velocity': 10, 'merchant_risk_score': 0.8,
     'description': 'Suspicious late-night, far away, high amount'},
    
    {'amount': 500, 'hour': 20, 'distance_from_home': 50,
     'transaction_velocity': 5, 'merchant_risk_score': 0.5,
     'description': 'Borderline case'},
]
 
print("\nReal-time Prediction Examples:")
print("-" * 60)
 
for txn in test_transactions:
    result = predict_fraud(txn, rf_model, scaler, threshold=0.5)
    print(f"\n{txn['description']}")
    print(f"  Amount: ${txn['amount']}, Hour: {txn['hour']}, "
          f"Distance: {txn['distance_from_home']}km")
    print(f"  → Fraud Probability: {result['fraud_probability']:.1%}")
    print(f"  → Risk Level: {result['risk_level']}")
    print(f"  → Action: {'🚨 FLAG FOR REVIEW' if result['is_flagged'] else '✅ APPROVE'}")
 
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("TUTORIAL SUMMARY")
print("=" * 60)
print("""
Key Takeaways:
1. Fraud detection requires handling imbalanced data (use class_weight)
2. Evaluation: Focus on Precision, Recall, F1, AUC - not accuracy
3. Feature engineering improves model performance significantly
4. Threshold tuning lets you balance fraud catch rate vs. false alarms
5. Always consider business impact when setting thresholds
 
Next Steps:
- Try SMOTE or other resampling techniques
- Experiment with XGBoost or neural networks
- Add more features (time since last transaction, etc.)
- Implement model monitoring for drift detection
""")
