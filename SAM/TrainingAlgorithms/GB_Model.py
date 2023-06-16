import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import json
import pickle
import base64
import time
import joblib

# Load the dataset from the CSV file
data = pd.read_csv("cyberbullying_tweets.csv")

# Split the dataset into features (X) and labels (y)
X = data['tweet_text']  # Assuming 'tweet_text' contains the sentence content
y = data['cyberbullying_type']  # Assuming 'cyberbullying_type' contains the class labels

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Convert the text data into numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train a Gradient Boosting classifier
print("Gradient Boosting")
start_time = time.time()
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
end_time = time.time()  # Stop measuring the duration
duration = end_time - start_time
duration_minutes = duration / 60

# Convert the classifier and vectorizer to serialized representations
vectorizer_serialized = base64.b64encode(pickle.dumps(vectorizer)).decode('utf-8')
classifier_serialized = base64.b64encode(pickle.dumps(gb)).decode('utf-8')

# Create a dictionary to hold the serialized objects
model_dict = {
    'vectorizer': vectorizer_serialized,
    'label_encoder': label_encoder.classes_.tolist(),
    'classifier': classifier_serialized
}

# Save the model dictionary as a JSON file
with open('GB_cyberbullying_model.json', 'w') as json_file:
    json.dump(model_dict, json_file)

# Save the trained model
joblib.dump(gb, 'GB_cyberbullying_model.pkl')

# Predict the sentiment for the test set using Gradient Boosting
y_pred_gb = gb.predict(X_test)

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
