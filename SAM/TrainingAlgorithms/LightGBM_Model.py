import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv('cyberbullying_tweets.csv')

# Separate the features (tweet_text) and labels (cyberbullying_type)
X = df['tweet_text'].values
y = df['cyberbullying_type'].values

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Data balancing - oversampling
oversampler = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train_vectorized, y_train)

# Hyperparameter tuning with RandomizedSearchCV
lgbm = LGBMClassifier(random_state=42)
param_distributions = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 5, 10],
    'min_child_samples': [1, 5, 10]
}
random_search = RandomizedSearchCV(lgbm, param_distributions, n_iter=10, cv=5, random_state=42)
random_search.fit(X_train_balanced, y_train_balanced)

# Train the LightGBM classifier with the best parameters
lgbm = random_search.best_estimator_
lgbm.fit(X_train_balanced, y_train_balanced)

# Save the trained model
joblib.dump(lgbm, 'LightGBM_cyberbullying_model.pkl')

# Evaluate the LightGBM model
lgbm_predictions = lgbm.predict(X_test_vectorized)
lgbm_accuracy = accuracy_score(y_test, lgbm_predictions)
lgbm_classification_report = classification_report(y_test, lgbm_predictions)
print('LightGBM Accuracy:', lgbm_accuracy)
print('LightGBM Classification Report:')
print(lgbm_classification_report)