import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import time

# Load the dataset from the CSV file
data = pd.read_csv("cyberbullying_tweets.csv")

# Split the dataset into features (X) and labels (y)
X = data['tweet_text']  # Assuming 'tweet_text' contains the sentence content
y = data['cyberbullying_type']  # Assuming 'cyberbullying_type' contains the class labels

# Convert the text data into numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Address Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

# Split the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Fine-tune hyperparameters
param_grid = {
    'n_estimators': [300],
    'learning_rate': [0.1],
    'max_depth': [5]
}

# Train a Gradient Boosting classifier with hyperparameter tuning
print("Gradient Boosting")
start_time = time.time()
gb = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
end_time = time.time()  # Stop measuring the duration
duration = end_time - start_time
duration_minutes = duration / 60

# Predict the sentiment for the test set using Gradient Boosting
y_pred_gb = grid_search.predict(X_test)

# Save the trained model
joblib.dump(grid_search, 'GB_cyberbullying_model.pkl')

# Print classification report for Gradient Boosting
print("Classification Report - Gradient Boosting:")
report_gb = classification_report(y_test, y_pred_gb)
print(report_gb)

# Calculate accuracy and error percentage
accuracy_gb = (y_pred_gb == y_test).mean() * 100
error_percentage_gb = 100 - accuracy_gb

# Display duration, accuracy, and error percentage
print("Duration: {:.2f} minutes".format(duration_minutes))
print("Accuracy: {:.2f}%".format(accuracy_gb))
print("Error Percentage: {:.2f}%".format(error_percentage_gb))
print()
