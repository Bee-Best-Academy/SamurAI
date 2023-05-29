import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow_hub as hub
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset from the CSV file
data = pd.read_csv("cyberbullying_tweets.csv")

# Extract the tweet text and cyberbullying type columns
tweets = data['tweet_text'].tolist()
labels = data['cyberbullying_type'].tolist()

# Train a Universal Sentence Encoder classifier
print("Universal Sentence Encoder (USE)")
# Load the Universal Sentence Encoder module
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(module_url)

# Encode the tweet text using the Universal Sentence Encoder
tweet_embeddings = embed(tweets).numpy()

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tweet_embeddings, y_encoded, test_size=0.2, random_state=42)

# Train a logistic regression classifier on the training set
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred_use = classifier.predict(X_test)

# Save the trained model
joblib.dump(classifier, 'USE_cyberbullying_model.pkl')

# Evaluate the performance of the USE
print("Classification Report - USE:")
report = classification_report(y_test, y_pred_use)
print(report)

# Calculate accuracy and error percentage
y_pred_use = np.array(y_pred_use)
y_test = np.array(y_test)
accuracy_use = (y_pred_use == y_test).mean() * 100
error_percentage_use = 100 - accuracy_use

# Display accuracy and error percentage
print("Accuracy: {:.2f}%".format(accuracy_use))
print("Error Percentage: {:.2f}%".format(error_percentage_use))
print()
