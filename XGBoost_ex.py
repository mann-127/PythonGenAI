import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
# 1. Load and Split Data
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
print(f"--- 1. Data Loading and Splitting ---")
print(f"Total Samples: {len(X)}")
print(f"Training Samples (X_train): {len(X_train)}")
print(f"Testing Samples (X_test): {len(X_test)}\n")
 
 
# 2. Initialize the Model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,           # Set to 100 trees/boosting rounds
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',      # Metric to track during training
    random_state=42
)
 
print(f"--- 2. Model Initialization ---")
print(f"Model Objective: {model.objective} (Outputting probability of class 1)")
print(f"Number of Trees (n_estimators): {model.n_estimators}")
print(f"Learning Rate (eta): {model.learning_rate} (Shrinking each tree's contribution)\n")
 
 
# 3. Training the Model with Verbose Output
# eval_set: Pass the test set to monitor performance during training
# verbose=20: Print a log message every 20 boosting rounds/trees
print(f"--- 3. Starting Iterative Training (Boosting) ---")
print("Log shows performance after every 20 trees (rounds 0 to 99).")
 
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=20
)
 
print(f"\n--- 4. Training Complete ---")
print(f"Final Model is the ensemble of {model.n_estimators} trees.\n")
 
 
# 5. Make Final Predictions and Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
 
print(f"--- 5. Prediction and Evaluation ---")
print(f"Predictions made on {len(X_test)} unseen samples.")
print(f"Final XGBoost Model Accuracy on Test Set: {accuracy:.4f}")
