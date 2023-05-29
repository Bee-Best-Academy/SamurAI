from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub

# Load the dataset
df = pd.read_csv('cyberbullying_tweets.csv')

# Separate the features (tweet_text) and labels (cyberbullying_type)
X = df['tweet_text'].values
y = df['cyberbullying_type'].values

# Example sentences for prediction
example_sentences = [
    "Look that nigger boy, is crazy.",  # ethnicity
    "I'm not sexist, I just don't like women.",  # gender
    "I need a gif of an old cat laughing."  # not_cyberbullying
]

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Vectorize the example sentences using the same vectorizer
vectorized_sentences = vectorizer.transform(example_sentences)

# Load the trained SVM model
svm = joblib.load('SVM_cyberbullying_model.pkl')

# Make the predictions using SVM
predictions = svm.predict(vectorized_sentences)
probabilities = svm.predict_proba(vectorized_sentences)

# Decode the predictions
predicted_labels = label_encoder.inverse_transform(predictions)

# Display the predictions
print(" - Support Vector Machines (SVM)")
for sentence, prediction, probs in zip(example_sentences, predicted_labels, probabilities):
    print("Example Sentence:", sentence)
    print("Predicted Label:", prediction)
    print("Probabilities:")
    
    # Create a list of tuples with label names and probabilities
    label_probabilities = list(zip(label_encoder.classes_, probs))
    label_probabilities.sort(key=lambda x: x[1], reverse=True)  # Sort probabilities in descending order
    
    for label, probability in label_probabilities:
        percentage = probability * 100  # Convert probability to percentage
        print(f"{label}: {percentage:.2f}%")
    
    print()

print()
# Train a Random Forest classifier

# Load the trained Random Forest model
rf = joblib.load('RF_cyberbullying_model.pkl')

# Make the predictions using Random Forest
predictions = rf.predict(vectorized_sentences)
probabilities = rf.predict_proba(vectorized_sentences)

# Decode the predictions
predicted_labels = label_encoder.inverse_transform(predictions)

# Display the predictions
print(" - Random Forest")
for sentence, prediction, probs in zip(example_sentences, predicted_labels, probabilities):
    print("Example Sentence:", sentence)
    print("Predicted Label:", prediction)
    print("Probabilities:")
    
    # Create a list of tuples with label names and probabilities
    label_probabilities = list(zip(label_encoder.classes_, probs))
    label_probabilities.sort(key=lambda x: x[1], reverse=True)  # Sort probabilities in descending order
    
    for label, probability in label_probabilities:
        percentage = probability * 100  # Convert probability to percentage
        print(f"{label}: {percentage:.2f}%")
    
    print()

print()
# Train a Gradient Boosting classifier

# Load the trained Random Forest model
gb = joblib.load('GB_cyberbullying_model.pkl')

# Make the predictions using Random Forest
predictions = gb.predict(vectorized_sentences)
probabilities = gb.predict_proba(vectorized_sentences)

# Decode the predictions
predicted_labels = label_encoder.inverse_transform(predictions)

# Display the predictions
print(" - Gradient Boosting")
for sentence, prediction, probs in zip(example_sentences, predicted_labels, probabilities):
    print("Example Sentence:", sentence)
    print("Predicted Label:", prediction)
    print("Probabilities:")
    
    # Create a list of tuples with label names and probabilities
    label_probabilities = list(zip(label_encoder.classes_, probs))
    label_probabilities.sort(key=lambda x: x[1], reverse=True)  # Sort probabilities in descending order
    
    for label, probability in label_probabilities:
        percentage = probability * 100  # Convert probability to percentage
        print(f"{label}: {percentage:.2f}%")
    
    print()

print()
# Train a Universal Sentence Encoder classifier

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(module_url)

# Vectorize the example sentences using the Universal Sentence Encoder
vectorized_sentences_use = embed(example_sentences).numpy()

# Load the trained Random Forest model
use = joblib.load('USE_cyberbullying_model.pkl')

# Make the predictions using Random Forest
predictions = use.predict(vectorized_sentences_use)
probabilities = use.predict_proba(vectorized_sentences_use)

# Decode the predictions
predicted_labels = label_encoder.inverse_transform(predictions)

# Display the predictions
print(" - Universal Sentence Encoder (USE)")
for sentence, prediction, probs in zip(example_sentences, predicted_labels, probabilities):
    print("Example Sentence:", sentence)
    print("Predicted Label:", prediction)
    print("Probabilities:")
    
    # Create a list of tuples with label names and probabilities
    label_probabilities = list(zip(label_encoder.classes_, probs))
    label_probabilities.sort(key=lambda x: x[1], reverse=True)  # Sort probabilities in descending order
    
    for label, probability in label_probabilities:
        percentage = probability * 100  # Convert probability to percentage
        print(f"{label}: {percentage:.2f}%")
    
    print()

print()
# Test a Gated Recurrent Unit classifier

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

# Convert text sequences to numerical sequences
X_seq = tokenizer.texts_to_sequences(X)
max_sequence_length = max([len(seq) for seq in X_seq])
X_padded = pad_sequences(X_seq, maxlen=max_sequence_length)

# Load the trained model
gru = load_model('GRU_cyberbullying_model.h5')

new_texts_seq = tokenizer.texts_to_sequences(example_sentences)
new_texts_padded = pad_sequences(new_texts_seq, maxlen=max_sequence_length)
predictions = gru.predict(new_texts_padded)

# Decode the predictions
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

# Display the predictions
print(" - Gated Recurrent Unit (GRU)")
for sentence, prediction, probs in zip(example_sentences, predicted_labels, predictions):
    print("Example Sentence:", sentence)
    print("Predicted Label:", prediction)
    print("Probabilities:")
    
    # Create a list of tuples with label names and probabilities
    label_probabilities = list(zip(label_encoder.classes_, probs))
    label_probabilities.sort(key=lambda x: x[1], reverse=True)  # Sort probabilities in descending order
    
    for label, probability in label_probabilities:
        percentage = probability * 100  # Convert probability to percentage
        print(f"{label}: {percentage:.2f}%")
    
    print()

print()
# Test the LightGBM classifier

# Load the trained LightGBM model
lgbm = joblib.load('LightGBM_cyberbullying_model.pkl')

# Make the predictions using LightGBM
predictions = lgbm.predict(vectorized_sentences)
probabilities = lgbm.predict_proba(vectorized_sentences)

# Decode the predictions
predicted_labels = label_encoder.inverse_transform(predictions)

# Display the predictions
print(" - LightGBM")
for sentence, prediction, probs in zip(example_sentences, predicted_labels, probabilities):
    print("Example Sentence:", sentence)
    print("Predicted Label:", prediction)
    print("Probabilities:")

    # Create a list of tuples with label names and probabilities
    label_probabilities = list(zip(label_encoder.classes_, probs))
    label_probabilities.sort(key=lambda x: x[1], reverse=True)  # Sort probabilities in descending order

    for label, probability in label_probabilities:
        percentage = probability * 100  # Convert probability to percentage
        print(f"{label}: {percentage:.2f}%")

    print()
