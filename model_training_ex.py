"""
Complete Example: Training a Neural Network Model
This script demonstrates the difference between untrained and trained models
using the MNIST dataset for digit classification.
"""
 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
 
# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
 
print("=" * 60)
print("MACHINE LEARNING MODEL TRAINING DEMONSTRATION")
print("=" * 60)
 
# ============================================
# 1. LOAD AND PREPARE DATA
# ============================================
print("\n1. Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
 
# Normalize pixel values to 0-1 range
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
 
# Flatten images from 28x28 to 784
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
 
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")
print(f"   Input features: {X_train.shape[1]}")
 
# ============================================
# 2. CREATE UNTRAINED MODEL
# ============================================
print("\n2. Creating untrained model...")
 
def create_model():
    """Create a simple neural network"""
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model
 
# Create untrained model
untrained_model = create_model()
print("   Model architecture created with random weights")
 
# ============================================
# 3. TEST UNTRAINED MODEL
# ============================================
print("\n3. Testing UNTRAINED model performance...")
 
# Make predictions with untrained model
untrained_predictions = untrained_model.predict(X_test[:1000], verbose=0)
untrained_pred_classes = np.argmax(untrained_predictions, axis=1)
true_classes = y_test[:1000]
 
# Calculate accuracy
untrained_accuracy = np.mean(untrained_pred_classes == true_classes)
print(f"   Untrained Model Accuracy: {untrained_accuracy:.2%}")
print(f"   (Random guessing would be ~10% for 10 classes)")
 
# ============================================
# 4. COMPILE MODEL FOR TRAINING
# ============================================
print("\n4. Compiling model for training...")
 
trained_model = create_model()
trained_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("   Optimizer: Adam")
print("   Loss function: Sparse Categorical Crossentropy")
 
# ============================================
# 5. TRAIN THE MODEL
# ============================================
print("\n5. Training the model...")
print("   This may take a few minutes...")
 
history = trained_model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)
 
print("\n   Training completed!")
 
# ============================================
# 6. TEST TRAINED MODEL
# ============================================
print("\n6. Testing TRAINED model performance...")
 
# Evaluate on test set
test_loss, test_accuracy = trained_model.evaluate(X_test, y_test, verbose=0)
print(f"   Trained Model Accuracy: {test_accuracy:.2%}")
print(f"   Test Loss: {test_loss:.4f}")
 
# Make predictions
trained_predictions = trained_model.predict(X_test[:1000], verbose=0)
trained_pred_classes = np.argmax(trained_predictions, axis=1)
 
# ============================================
# 7. COMPARISON AND RESULTS
# ============================================
print("\n" + "=" * 60)
print("COMPARISON: UNTRAINED vs TRAINED MODEL")
print("=" * 60)
 
print(f"\nUNTRAINED Model:")
print(f"  Accuracy: {untrained_accuracy:.2%}")
print(f"  Status: Making random predictions ❌")
 
print(f"\nTRAINED Model:")
print(f"  Accuracy: {test_accuracy:.2%}")
print(f"  Status: Making intelligent predictions ✓")
 
improvement = (test_accuracy - untrained_accuracy) / untrained_accuracy * 100
print(f"\nImprovement: {improvement:.1f}% better performance!")
 
# ============================================
# 8. VISUALIZE TRAINING PROGRESS
# ============================================
print("\n8. Training progress visualization...")
 
print("\nEpoch-by-Epoch Accuracy:")
for epoch, (train_acc, val_acc) in enumerate(zip(
    history.history['accuracy'],
    history.history['val_accuracy']
), 1):
    print(f"  Epoch {epoch:2d}: Train={train_acc:.2%}, Val={val_acc:.2%}")
 
# ============================================
# 9. EXAMPLE PREDICTIONS
# ============================================
print("\n9. Sample predictions on test images:")
print("-" * 60)
 
# Show 5 example predictions
for i in range(5):
    true_label = y_test[i]
    
    # Untrained prediction
    untrained_pred = np.argmax(untrained_model.predict(
        X_test[i:i+1], verbose=0
    ))
    
    # Trained prediction
    trained_pred = np.argmax(trained_model.predict(
        X_test[i:i+1], verbose=0
    ))
    
    print(f"Image {i+1}:")
    print(f"  True Label: {true_label}")
    print(f"  Untrained Model Prediction: {untrained_pred} {'✗' if untrained_pred != true_label else '✓'}")
    print(f"  Trained Model Prediction: {trained_pred} {'✗' if trained_pred != true_label else '✓'}")
    print()
 
# ============================================
# 10. KEY INSIGHTS
# ============================================
print("=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
 
print("""
1. UNTRAINED MODEL:
   - Starts with random weights
   - Makes random predictions
   - Low accuracy (~10% for 10 classes)
   - Completely useless for real applications
 
2. TRAINING PROCESS:
   - Learns patterns from data
   - Adjusts weights through backpropagation
   - Minimizes prediction error
   - Takes time and computational resources
 
3. TRAINED MODEL:
   - Optimized weights
   - Makes accurate predictions
   - High accuracy (>90% typically)
   - Ready for production use
 
4. CONCLUSION:
   Training transforms a random guesser into an intelligent system!
""")
 
# ============================================
# 11. SAVE TRAINED MODEL (OPTIONAL)
# ============================================
print("\n11. Saving trained model...")
trained_model.save('trained_mnist_model.h5')
print("   Model saved as 'trained_mnist_model.h5'")
print("   You can load it later with: keras.models.load_model('trained_mnist_model.h5')")
 
print("\n" + "=" * 60)
print("DEMONSTRATION COMPLETE!")
print("=" * 60)
