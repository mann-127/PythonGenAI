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
import matplotlib.pyplot as plt
import seaborn as sns
 
print("=" * 70)
print("TRAINING MODEL & WATCHING METRICS EVOLVE")
print("=" * 70)
 
# 1. CREATE DATASET
print("\n1. CREATING DATASET")
print("-" * 70)
np.random.seed(42)
 
# Generate synthetic dataset (e.g., predicting if a customer will buy)
n_samples = 1000
X = np.random.randn(n_samples, 5)  # 5 features
# Create target: y = 1 if weighted sum > threshold
y = (X[:, 0] * 2 + X[:, 1] * 1.5 - X[:, 2] * 0.5 + np.random.randn(n_samples) * 0.5 > 0).astype(int)
 
print(f"Total samples: {n_samples}")
print(f"Features: {X.shape[1]}")
print(f"Class distribution: Class 0: {sum(y==0)}, Class 1: {sum(y==1)}")
 
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
 
# 2. TRAIN MODEL WITH DIFFERENT TRAINING SIZES
print("\n2. TRAINING MODEL WITH INCREASING DATA")
print("-" * 70)
 
training_sizes = [50, 100, 200, 400, 700]  # Different training set sizes
metrics_history = {
    'size': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'accuracy': []
}
 
confusion_matrices = []
 
for size in training_sizes:
    # Train with subset of data
    X_train_subset = X_train[:size]
    y_train_subset = y_train[:size]
    
    # Train logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_subset, y_train_subset)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Store metrics
    metrics_history['size'].append(size)
    metrics_history['precision'].append(precision)
    metrics_history['recall'].append(recall)
    metrics_history['f1'].append(f1)
    metrics_history['accuracy'].append(accuracy)
    confusion_matrices.append(cm)
    
    print(f"\nTraining Size: {size}")
    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Accuracy: {accuracy:.4f}")
 
# 3. DETAILED VIEW OF FIRST AND LAST TRAINING
print("\n3. COMPARISON: EARLY vs FINAL TRAINING")
print("-" * 70)
 
print("\nEARLY TRAINING (50 samples):")
print(f"Confusion Matrix:\n{confusion_matrices[0]}")
print(f"Precision: {metrics_history['precision'][0]:.4f}")
print(f"Recall: {metrics_history['recall'][0]:.4f}")
print(f"F1-Score: {metrics_history['f1'][0]:.4f}")
 
print("\nFINAL TRAINING (700 samples):")
print(f"Confusion Matrix:\n{confusion_matrices[-1]}")
print(f"Precision: {metrics_history['precision'][-1]:.4f}")
print(f"Recall: {metrics_history['recall'][-1]:.4f}")
print(f"F1-Score: {metrics_history['f1'][-1]:.4f}")
 
# 4. VISUALIZATIONS
fig = plt.figure(figsize=(16, 10))
 
# Plot 1: Metrics Evolution
ax1 = plt.subplot(2, 3, 1)
plt.plot(metrics_history['size'], metrics_history['precision'], 'o-', label='Precision', linewidth=2, markersize=8)
plt.plot(metrics_history['size'], metrics_history['recall'], 's-', label='Recall', linewidth=2, markersize=8)
plt.plot(metrics_history['size'], metrics_history['f1'], '^-', label='F1-Score', linewidth=2, markersize=8)
plt.plot(metrics_history['size'], metrics_history['accuracy'], 'd-', label='Accuracy', linewidth=2, markersize=8)
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Metrics Evolution During Training', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(0.5, 1.0)
 
# Plot 2: Metrics Improvement
ax2 = plt.subplot(2, 3, 2)
improvement = {
    'Precision': metrics_history['precision'][-1] - metrics_history['precision'][0],
    'Recall': metrics_history['recall'][-1] - metrics_history['recall'][0],
    'F1-Score': metrics_history['f1'][-1] - metrics_history['f1'][0],
    'Accuracy': metrics_history['accuracy'][-1] - metrics_history['accuracy'][0]
}
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in improvement.values()]
bars = plt.bar(improvement.keys(), improvement.values(), color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Improvement', fontsize=12)
plt.title('Metric Improvement\n(Final - Initial)', fontsize=14, fontweight='bold')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:+.3f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
 
# Plot 3: Confusion Matrix - Early Training
ax3 = plt.subplot(2, 3, 4)
sns.heatmap(confusion_matrices[0], annot=True, fmt='d', cmap='Reds',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'], cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - Early\n(Training Size: {training_sizes[0]})', fontsize=12, fontweight='bold')
 
# Plot 4: Confusion Matrix - Final Training
ax4 = plt.subplot(2, 3, 5)
sns.heatmap(confusion_matrices[-1], annot=True, fmt='d', cmap='Greens',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'], cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - Final\n(Training Size: {training_sizes[-1]})', fontsize=12, fontweight='bold')
 
# Plot 5: Metrics Comparison Bar Chart
ax5 = plt.subplot(2, 3, 3)
x = np.arange(4)
width = 0.35
metrics_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
early_values = [metrics_history['precision'][0], metrics_history['recall'][0],
                metrics_history['f1'][0], metrics_history['accuracy'][0]]
final_values = [metrics_history['precision'][-1], metrics_history['recall'][-1],
                metrics_history['f1'][-1], metrics_history['accuracy'][-1]]
 
bars1 = plt.bar(x - width/2, early_values, width, label='Early (50 samples)', alpha=0.8, color='coral')
bars2 = plt.bar(x + width/2, final_values, width, label='Final (700 samples)', alpha=0.8, color='seagreen')
 
plt.ylabel('Score', fontsize=12)
plt.title('Early vs Final Performance', fontsize=14, fontweight='bold')
plt.xticks(x, metrics_names, rotation=45, ha='right')
plt.legend(fontsize=10)
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3, axis='y')
 
# Plot 6: Key Insights
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
insights_text = f"""
KEY OBSERVATIONS:
 
📈 Training Data Impact:
   • More data → Better metrics
   • Precision: {metrics_history['precision'][0]:.3f} → {metrics_history['precision'][-1]:.3f}
   • Recall: {metrics_history['recall'][0]:.3f} → {metrics_history['recall'][-1]:.3f}
   • F1-Score: {metrics_history['f1'][0]:.3f} → {metrics_history['f1'][-1]:.3f}
 
🎯 Confusion Matrix Changes:
   • Fewer misclassifications
   • Better balance between classes
   
💡 Why Metrics Improve:
   • Model learns better patterns
   • Reduces overfitting
   • Better generalization
"""
plt.text(0.1, 0.5, insights_text, fontsize=11, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
 
plt.tight_layout()
plt.show()
 
print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("✓ As training data increases, all metrics generally improve")
print("✓ Model becomes more reliable with more examples")
print("✓ Confusion matrix shows fewer errors with more training")
print("=" * 70)
