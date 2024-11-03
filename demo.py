# demo.py
# Enhanced Sentiment Analysis with Explainability Using Classic Machine Learning

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import shap
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ---------------------------
# Step 1: In-Memory Dataset
# ---------------------------

positive_reviews = [
    "I absolutely loved this movie! The performances were stellar and the story was gripping.",
    "An excellent film with a wonderful plot and fantastic acting.",
    "A masterpiece. Brilliantly directed and beautifully acted.",
    "Outstanding! A truly inspiring and moving experience.",
    "Loved every moment of it. Highly recommend to everyone!",
    "Fantastic movie. The characters were well developed and the plot was engaging.",
    "A delightful film that exceeded my expectations in every way.",
    "Absolutely wonderful! The cinematography was stunning.",
    "A heartwarming story with exceptional performances.",
    "One of the best movies I've seen this year. Highly enjoyable!",
    "Brilliant screenplay and outstanding direction.",
    "The actors delivered phenomenal performances throughout.",
    "A captivating narrative that kept me hooked till the end.",
    "Visually stunning with a compelling storyline.",
    "A beautiful blend of emotion and action.",
    "Exceptional direction and superb acting made this a must-watch.",
    "An inspiring film that resonates on multiple levels.",
    "The plot was intricate yet beautifully executed.",
    "A remarkable film experience with top-notch performances.",
    "Absolutely captivating from start to finish."
]

negative_reviews = [
    "I hated this movie. It was boring and too long.",
    "Terrible film. Poor acting and a weak storyline.",
    "A complete waste of time. Not worth watching.",
    "Disappointing. The plot was confusing and uninteresting.",
    "Bad movie. I walked out halfway through.",
    "The movie was dull and lacked any real substance.",
    "Poorly executed with lackluster performances.",
    "A monotonous film that failed to capture my interest.",
    "Not engaging at all. The story was lackluster.",
    "Worst movie ever. Completely disappointing.",
    "The storyline was weak and the acting was subpar.",
    "A forgettable movie with no redeeming qualities.",
    "Lacked originality and failed to impress.",
    "The plot was predictable and uninspired.",
    "Poorly written with mediocre performances.",
    "A tedious watch with no real impact.",
    "The film dragged on without any meaningful progression.",
    "Uninspired direction and lackluster execution.",
    "A disappointing effort from the entire cast and crew.",
    "Failed to deliver on its promises. Very disappointing."
]

texts = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

print("In-Memory Dataset Loaded Successfully!")
print(f"Total samples: {len(texts)}")
print(f"Positive samples: {sum(labels)}")
print(f"Negative samples: {len(labels) - sum(labels)}\n")

# -------------------------------------
# Step 2: Text Preprocessing & TF-IDF
# -------------------------------------

vectorizer = TfidfVectorizer(
    max_features=1000,          # Increased features for better representation
    ngram_range=(1, 2),         # Use unigrams and bigrams
    stop_words='english'        # Remove English stop words
)
X = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

print("TF-IDF Vectorization completed.")
print(f"Feature matrix shape: {X.shape}\n")

# ---------------------------
# Step 3: Train-Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

print("Dataset split into training and testing sets.")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}\n")

# ---------------------------
# Step 4: Train Logistic Model
# ---------------------------

model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

print("Logistic Regression model trained successfully.\n")

# ---------------------------
# Step 5: Predictions & Evaluation
# ---------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])

print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# ---------------------------
# Step 6: SHAP Explainability
# ---------------------------

print("Calculating SHAP values for explainability...\n")

# Initialize SHAP Explainer
explainer = shap.LinearExplainer(model, X_train, feature_dependence="independent")
shap_values = explainer.shap_values(X_test)

# ---------------------------
# Step 7: Display Top Global Features
# ---------------------------

def display_top_global_features(shap_values, feature_names, top_n=10):
    """
    Displays the top positive and negative features based on SHAP values.
    """
    # Mean absolute SHAP values for each feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_shap)[-top_n:]
    
    print(f"Top {top_n} Features Contributing to the Model (Global Importance):")
    for idx in top_indices[::-1]:
        feature = feature_names[idx]
        shap_val = mean_shap[idx]
        print(f"{feature}: {shap_val:.4f}")
    print()

display_top_global_features(shap_values, feature_names, top_n=10)

# ---------------------------
# Step 8: Individual Prediction Explanation
# ---------------------------

def explain_individual_prediction(model, explainer, X_test, texts, y_test, y_pred, index=0):
    """
    Explains a single prediction using SHAP.
    """
    print(f"Explaining prediction for Test Instance {index}:")
    print(f"Review Text: {texts[index]}")
    print(f"Actual Label: {'Positive' if y_test[index] == 1 else 'Negative'}")
    print(f"Predicted Label: {'Positive' if y_pred[index] == 1 else 'Negative'}\n")
    
    # Get SHAP values for the instance
    instance_shap = shap_values[index]
    
    # Get top contributing features
    top_indices = np.argsort(np.abs(instance_shap))[-5:]
    top_features = feature_names[top_indices]
    top_shap_values = instance_shap[top_indices]
    
    print("Top Features Contributing to this Prediction:")
    for feature, shap_val in zip(top_features, top_shap_values):
        direction = "Positive" if shap_val > 0 else "Negative"
        print(f"{feature}: {shap_val:.4f} ({direction})")
    print()

# Explain the first test instance
explain_individual_prediction(model, explainer, X_test, texts, y_test, y_pred, index=0)

# ---------------------------
# Step 9: Display Top Model Features
# ---------------------------

def display_top_model_features(model, feature_names, top_n=10):
    """
    Displays the top positive and negative features based on model coefficients.
    """
    coef = model.coef_[0]
    top_positive_indices = np.argsort(coef)[-top_n:]
    top_negative_indices = np.argsort(coef)[:top_n]
    
    top_positive_features = [feature_names[i] for i in top_positive_indices]
    top_positive_coefs = coef[top_positive_indices]
    
    top_negative_features = [feature_names[i] for i in top_negative_indices]
    top_negative_coefs = coef[top_negative_indices]
    
    print(f"Top {top_n} Positive Features from Model Coefficients:")
    for feature, coef_val in zip(top_positive_features, top_positive_coefs):
        print(f"{feature}: {coef_val:.4f}")
    
    print(f"\nTop {top_n} Negative Features from Model Coefficients:")
    for feature, coef_val in zip(top_negative_features, top_negative_coefs):
        print(f"{feature}: {coef_val:.4f}")
    print()

display_top_model_features(model, feature_names, top_n=10)

# ---------------------------
# Step 10: Prediction Distribution
# ---------------------------

def summarize_predictions(y_true, y_pred):
    """
    Summarizes the prediction results.
    """
    total = len(y_pred)
    correct = sum(y_true[i] == y_pred[i] for i in range(total))
    incorrect = total - correct
    
    print(f"Total Predictions: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Incorrect Predictions: {incorrect}\n")
    
    # Breakdown by class
    conf_matrix = confusion_matrix(y_true, y_pred)
    if conf_matrix.shape == (2,2):
        tn, fp, fn, tp = conf_matrix.ravel()
        
        print("Breakdown by Class:")
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}\n")
    else:
        print("Confusion matrix shape not as expected for binary classification.\n")

summarize_predictions(y_test, y_pred)
"""
In-Memory Dataset Loaded Successfully!
Total samples: 40
Positive samples: 20
Negative samples: 20

TF-IDF Vectorization completed.
Feature matrix shape: (40, 264)

Dataset split into training and testing sets.
Training samples: 30
Testing samples: 10

Logistic Regression model trained successfully.

Model Accuracy: 90.00%

Confusion Matrix:
[[4 1]
 [0 5]]

Classification Report:
              precision    recall  f1-score   support

    Negative       1.00      0.80      0.89         5
    Positive       0.83      1.00      0.91         5

    accuracy                           0.90        10
   macro avg       0.92      0.90      0.90        10
weighted avg       0.92      0.90      0.90        10

Calculating SHAP values for explainability...

Top 10 Features Contributing to the Model (Global Importance):
movie: 0.0234
disappointing: 0.0182
lackluster: 0.0160
performances: 0.0128
story: 0.0110
absolutely: 0.0106
plot: 0.0103
failed: 0.0092
uninspired: 0.0083
stunning: 0.0065

Explaining prediction for Test Instance 0:
Review Text: I absolutely loved this movie! The performances were stellar and the story was gripping.
Actual Label: Negative
Predicted Label: Negative

Top Features Contributing to this Prediction:
uninspired: 0.0083 (Positive)
disappointing: 0.0182 (Positive)
movie: 0.0196 (Positive)
performances: 0.0270 (Positive)
lackluster: -0.0947 (Negative)

Top 10 Positive Features from Model Coefficients:
stunning compelling: 0.1553
compelling storyline: 0.1553
compelling: 0.1553
highly: 0.2351
inspiring: 0.2388
captivating: 0.2420
experience: 0.2429
absolutely: 0.2498
outstanding: 0.2540
stunning: 0.2759

Top 10 Negative Features from Model Coefficients:
disappointing: -0.4552
movie: -0.3954
uninspired: -0.3130
lackluster: -0.2963
real: -0.2626
story lackluster: -0.1980
engaging story: -0.1980
plot predictable: -0.1880
predictable: -0.1880
predictable uninspired: -0.1880

Total Predictions: 10
Correct Predictions: 9
Incorrect Predictions: 1

Breakdown by Class:
True Positives: 5
True Negatives: 4
False Positives: 1
False Negatives: 0
"""
