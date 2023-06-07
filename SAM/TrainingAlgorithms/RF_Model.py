import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import time

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

# Train a Random Forest classifier
print("Random Forest")
start_time = time.time()
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

end_time = time.time()  # Stop measuring the duration
duration = end_time - start_time
duration_minutes = duration / 60

# Predict the sentiment for the test set using Random Forest
y_pred_rf = rf.predict(X_test)

# Save the trained model
joblib.dump(rf, 'RF_cyberbullying_model.pkl')

# Print classification report for Random Forest
print("Classification Report - Random Forest:")
report_rf = classification_report(y_test, y_pred_rf)
print(report_rf)

# Calculate accuracy and error percentage
accuracy_rf = (y_pred_rf == y_test).mean() * 100
error_percentage_rf = 100 - accuracy_rf

# Display duration, accuracy, and error percentage
print("Duration: {:.2f} minutes".format(duration_minutes))
print("Accuracy: {:.2f}%".format(accuracy_rf))
print("Error Percentage: {:.2f}%".format(error_percentage_rf))
print()
