from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
 
# Larger sample dataset - reviews and their sentiment labels
# 1 = positive, 0 = negative
reviews = [
    # Positive reviews (1)
    "This product is amazing! I love it so much.",
    "Best purchase I've made this year!",
    "Excellent service and great product.",
    "Absolutely fantastic experience!",
    "I'm very happy with my purchase.",
    "Outstanding quality and fast delivery.",
    "Exceeded my expectations in every way.",
    "Highly recommend to everyone!",
    "Perfect! Exactly what I needed.",
    "Great value for money!",
    "Wonderful product, very satisfied.",
    "Superb quality and excellent design.",
    "Love everything about this purchase.",
    "The best choice I could have made.",
    "Incredible product, worth every penny.",
    "Fantastic quality, shipped quickly.",
    "Very pleased with this item.",
    "Impressive performance and durability.",
    "Amazing features, easy to use.",
    "Top notch product and service.",
    "Brilliant purchase, no regrets.",
    "Exceptional value, highly satisfied.",
    "Really good product, works perfectly.",
    "Great experience from start to finish.",
    "Excellent quality at a reasonable price.",
    "Very happy customer, will buy again.",
    "Superior quality compared to others.",
    "Delighted with this purchase.",
    "Really impressed with the quality.",
    "Best product in its category.",
    "Awesome purchase, totally worth it.",
    "Very good quality and affordable.",
    "Really satisfied with my order.",
    "Fantastic product, exactly as described.",
    "Love it! Better than expected.",
    "Great buy, highly recommend it.",
    "Very impressed with the quality.",
    "Excellent purchase, no complaints.",
    "Really happy with this choice.",
    "Perfect product for my needs.",
    
    # Negative reviews (0)
    "Terrible quality, waste of money.",
    "Completely disappointed with this item.",
    "Would not recommend to anyone.",
    "Poor quality and bad customer service.",
    "This is the worst product ever.",
    "Total garbage, don't buy this.",
    "Regret buying this item.",
    "Broke after one day of use.",
    "Horrible experience from start to finish.",
    "Complete disappointment.",
    "Awful product, nothing works.",
    "Very poor quality, fell apart quickly.",
    "Waste of time and money.",
    "Terrible experience, avoid this.",
    "Not worth the price at all.",
    "Disappointing quality and performance.",
    "Bad product, stopped working immediately.",
    "Really unhappy with this purchase.",
    "Poor design and cheap materials.",
    "Does not work as advertised.",
    "Very disappointed, expected better.",
    "Terrible customer service experience.",
    "Cheap quality, breaks easily.",
    "Not what I expected, very bad.",
    "Horrible quality control issues.",
    "Regret this purchase completely.",
    "Utterly useless product.",
    "Poor value for money spent.",
    "Defective item, very frustrating.",
    "Worst purchase I've ever made.",
    "Dreadful quality, returned it.",
    "Terrible build quality throughout.",
    "Very poor performance, disappointing.",
    "Not recommended, save your money.",
    "Bad experience from beginning to end.",
    "Cheaply made, broke instantly.",
    "Awful product, total waste.",
    "Very dissatisfied with everything.",
    "Poor quality and overpriced.",
    "Terrible product, avoid at all costs."
]
 
# Labels: 1 for positive (first 40), 0 for negative (last 40)
labels = [1] * 40 + [0] * 40
 
# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.2, random_state=42, stratify=labels
)
 
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Positive samples in training: {sum(y_train)}")
print(f"Negative samples in training: {len(y_train) - sum(y_train)}\n")
 
# Convert text to numerical features using TF-IDF
# Using character n-grams helps capture more patterns
vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),  # Use both unigrams and bigrams
    min_df=1,
    stop_words='english'
)
 
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
 
# Train a Logistic Regression classifier (works better with small datasets)
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train_tfidf, y_train)
 
# Make predictions
y_pred = classifier.predict(X_test_tfidf)
 
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}\n")
 
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
 
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
 
# Test with new reviews
new_reviews = [
    "This is absolutely wonderful!",
    "I hate this product.",
    "Pretty good overall experience.",
    "Worst thing I've ever bought.",
    "Decent product, does the job.",
    "Total waste of money, broken on arrival."
]
 
new_reviews_tfidf = vectorizer.transform(new_reviews)
predictions = classifier.predict(new_reviews_tfidf)
probabilities = classifier.predict_proba(new_reviews_tfidf)
 
print("\nPredictions for new reviews:")
print("-" * 80)
for review, prediction, proba in zip(new_reviews, predictions, probabilities):
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = proba[prediction] * 100
    print(f"Review: '{review}'")
    print(f"  -> {sentiment} (confidence: {confidence:.1f}%)\n")
