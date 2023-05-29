import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset from the CSV file
data = pd.read_csv("cyberbullying_tweets.csv")

# Print the column names
print(data.columns)

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
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# Predict the sentiment for the test set using Gradient Boosting
y_pred_gb = gb.predict(X_test)

# Save the trained model
joblib.dump(gb, 'GB_cyberbullying_model.pkl')

# Print classification report for Gradient Boosting
print("Classification Report - Gradient Boosting:")
report_gb = classification_report(y_test, y_pred_gb)
print(report_gb)

# Calculate accuracy and error percentage
accuracy_gb = (y_pred_gb == y_test).mean() * 100
error_percentage_gb = 100 - accuracy_gb

# Display accuracy and error percentage
print("Accuracy: {:.2f}%".format(accuracy_gb))
print("Error Percentage: {:.2f}%".format(error_percentage_gb))
print()
