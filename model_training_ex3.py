import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score
)
 
print("=" * 80)
print("MODEL TRAINING PROCESS - STEP BY STEP EXPLANATION")
print("=" * 80)
 
# STEP 1: CREATE DATASET
print("\nSTEP 1: CREATING DATASET")
print("-" * 80)
np.random.seed(42)
 
# Create features (like age, income, purchase_history, etc.)
n_samples = 500
X = np.random.randn(n_samples, 3)  # 3 features
# Create target based on features
y = (X[:, 0] * 1.5 + X[:, 1] * 0.8 - X[:, 2] * 0.3 + np.random.randn(n_samples) * 0.5 > 0).astype(int)
 
print(f"Dataset created with {n_samples} samples and {X.shape[1]} features")
print(f"Target distribution: Class 0: {sum(y==0)}, Class 1: {sum(y==1)}")
print(f"\nFirst 5 samples:")
print(f"Features (X):\n{X[:5]}")
print(f"Labels (y): {y[:5]}")
 
# STEP 2: SPLIT DATA
print("\n\nSTEP 2: SPLITTING DATA INTO TRAIN AND TEST SETS")
print("-" * 80)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print("Why split? To evaluate how model performs on unseen data")
 
# STEP 3: INITIALIZE MODEL
print("\n\nSTEP 3: INITIALIZE LOGISTIC REGRESSION MODEL")
print("-" * 80)
model = LogisticRegression(random_state=42, max_iter=1000)
print(f"Model: {type(model).__name__}")
print(f"Initial parameters (weights): Not yet trained")
print("\nLogistic Regression learns these parameters:")
print("  - Coefficients (weights) for each feature")
print("  - Intercept (bias term)")
 
# STEP 4: TRAIN MODEL (BEFORE TRAINING)
print("\n\nSTEP 4: MODEL STATE BEFORE TRAINING")
print("-" * 80)
print("Coefficients: None (not fitted yet)")
print("Intercept: None (not fitted yet)")
 
# STEP 5: ACTUAL TRAINING
print("\n\nSTEP 5: TRAINING THE MODEL")
print("-" * 80)
print("Training process:")
print("  1. Model starts with random weights")
print("  2. Makes predictions on training data")
print("  3. Calculates error (how wrong predictions are)")
print("  4. Updates weights to reduce error")
print("  5. Repeats until error is minimized or max iterations reached")
print("\nTraining in progress...")
 
model.fit(X_train, y_train)
 
print("✓ Training completed!")
 
# STEP 6: MODEL PARAMETERS AFTER TRAINING
print("\n\nSTEP 6: MODEL PARAMETERS AFTER TRAINING")
print("-" * 80)
print("Learned Coefficients (weights for each feature):")
for i, coef in enumerate(model.coef_[0]):
    print(f"  Feature {i+1} weight: {coef:.6f}")
print(f"\nIntercept (bias): {model.intercept_[0]:.6f}")
 
print("\nWhat these mean:")
print("  - Positive weight: Feature increases probability of Class 1")
print("  - Negative weight: Feature decreases probability of Class 1")
print("  - Larger absolute value: More important feature")
 
# STEP 7: MAKE PREDICTIONS
print("\n\nSTEP 7: MAKING PREDICTIONS ON TEST DATA")
print("-" * 80)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
 
 
print("First 10 predictions:")
print(f"{'Actual':<10} {'Predicted':<12} {'Prob Class 0':<15} {'Prob Class 1':<15} {'Correct?'}")
print("-" * 70)
for i in range(10):
    correct = "✓" if y_test[i] == y_pred[i] else "✗"
    print(f"{y_test[i]:<10} {y_pred[i]:<12} {y_pred_proba[i][0]:.4f}{'':11} {y_pred_proba[i][1]:.4f}{'':11} {correct}")
 
# STEP 8: CALCULATE METRICS
print("\n\nSTEP 8: EVALUATING MODEL PERFORMANCE")
print("-" * 80)
 
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
 
print("Confusion Matrix:")
print(f"                 Predicted 0    Predicted 1")
print(f"Actual 0         {cm[0][0]:<14} {cm[0][1]:<14}")
print(f"Actual 1         {cm[1][0]:<14} {cm[1][1]:<14}")
 
print(f"\nTrue Negatives (TN):  {cm[0][0]} - Correctly predicted Class 0")
print(f"False Positives (FP): {cm[0][1]} - Incorrectly predicted Class 1")
print(f"False Negatives (FN): {cm[1][0]} - Incorrectly predicted Class 0")
print(f"True Positives (TP):  {cm[1][1]} - Correctly predicted Class 1")
 
print(f"\nMetrics:")
print(f"  Accuracy:  {accuracy:.4f} - Overall correctness = (TP+TN)/(TP+TN+FP+FN)")
print(f"  Precision: {precision:.4f} - Of predicted positives, how many correct = TP/(TP+FP)")
print(f"  Recall:    {recall:.4f} - Of actual positives, how many found = TP/(TP+FN)")
print(f"  F1-Score:  {f1:.4f} - Balance of precision and recall")
 
# STEP 9: SHOW HOW PARAMETERS AFFECT PREDICTIONS
print("\n\nSTEP 9: HOW PARAMETERS MAKE PREDICTIONS")
print("-" * 80)
print("For a single test sample, let's trace the prediction:")
sample_idx = 0
sample_x = X_test[sample_idx]
print(f"\nSample features: {sample_x}")
 
# Manual calculation
weights = model.coef_[0]
intercept = model.intercept_[0]
z = np.dot(weights, sample_x) + intercept
prob_class1 = 1 / (1 + np.exp(-z))  # Sigmoid function
prob_class0 = 1 - prob_class1
prediction = 1 if prob_class1 > 0.5 else 0
 
print(f"\nCalculation:")
print(f"  Step 1: Linear combination (z) = w1*x1 + w2*x2 + w3*x3 + intercept")
print(f"          z = {weights[0]:.4f}*{sample_x[0]:.4f} + {weights[1]:.4f}*{sample_x[1]:.4f} + {weights[2]:.4f}*{sample_x[2]:.4f} + {intercept:.4f}")
print(f"          z = {z:.4f}")
print(f"\n  Step 2: Apply sigmoid function to get probability")
print(f"          P(Class 1) = 1 / (1 + e^(-z)) = {prob_class1:.4f}")
print(f"          P(Class 0) = 1 - P(Class 1) = {prob_class0:.4f}")
print(f"\n  Step 3: Predict class with higher probability")
print(f"          Prediction: Class {prediction}")
print(f"          Actual: Class {y_test[sample_idx]}")
 
# STEP 10: TRAINING WITH DIFFERENT DATA SIZES
print("\n\nSTEP 10: HOW TRAINING DATA SIZE AFFECTS PARAMETERS AND PERFORMANCE")
print("-" * 80)
 
training_sizes = [50, 100, 200, 350]
print(f"{'Size':<8} {'Coef1':<12} {'Coef2':<12} {'Coef3':<12} {'Intercept':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 100)
 
for size in training_sizes:
    # Train with subset
    X_subset = X_train[:size]
    y_subset = y_train[:size]
    
    model_temp = LogisticRegression(random_state=42, max_iter=1000)
    model_temp.fit(X_subset, y_subset)
    
    # Predict and calculate metrics
    y_pred_temp = model_temp.predict(X_test)
    prec = precision_score(y_test, y_pred_temp)
    rec = recall_score(y_test, y_pred_temp)
    f1_temp = f1_score(y_test, y_pred_temp)
    
    print(f"{size:<8} {model_temp.coef_[0][0]:<12.6f} {model_temp.coef_[0][1]:<12.6f} {model_temp.coef_[0][2]:<12.6f} {model_temp.intercept_[0]:<12.6f} {prec:<12.4f} {rec:<12.4f} {f1_temp:<12.4f}")
 
print("\nObservations:")
print("  • Parameters (coefficients) stabilize as training data increases")
print("  • More data → More stable parameters → Better generalization")
print("  • Metrics improve as model learns true patterns from more examples")
 
# STEP 11: KEY INSIGHTS
print("\n\n" + "=" * 80)
print("KEY INSIGHTS - WHAT HAPPENS DURING TRAINING")
print("=" * 80)
print("""
1. INITIALIZATION:
   • Model starts with random or zero parameters
 
2. TRAINING PROCESS (Iterative):
   • Forward Pass: Make predictions using current parameters
   • Calculate Loss: Measure how wrong predictions are
   • Backward Pass: Calculate gradients (direction to adjust parameters)
   • Update Parameters: Adjust weights to reduce loss
   • Repeat until convergence
 
3. PARAMETER CHANGES:
   • Weights adjust to capture feature importance
   • Positive weights → feature increases probability of positive class
   • Negative weights → feature decreases probability of positive class
   
4. MORE TRAINING DATA:
   • Parameters become more stable and accurate
   • Better generalization to unseen data
   • Metrics (Precision, Recall, F1) improve
   
5. FINAL MODEL:
   • Learned parameters encode patterns from training data
   • Can make predictions on new, unseen data
   • Performance measured by Precision, Recall, F1-Score, Confusion Matrix
""")
 
print("=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
