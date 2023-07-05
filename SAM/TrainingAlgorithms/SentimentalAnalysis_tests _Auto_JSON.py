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
import json
import base64
import pickle
import tensorflow as tf

# Load the dataset
df = pd.read_csv('cyberbullying_tweets.csv')

# Separate the features (tweet_text) and labels (cyberbullying_type)
X = df['tweet_text'].values
y = df['cyberbullying_type'].values

# Example sentences for prediction
example_sentences = [
    "Look that nigger boy, is crazy.",  # ethnicity
    "That is why you earn less than a man!",  # gender
    "I need a gif of a cat laughing."  # not_cyberbullying
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
#svm = joblib.load('SVM_cyberbullying_model.pkl')

# Load the JSON file containing the serialized vectorizer and classifier
with open('SVM_cyberbullying_model.json', 'r') as json_file:
    model_dict = json.load(json_file)

# Decode the serialized objects
vectorizer_serialized = base64.b64decode(model_dict['vectorizer'])
classifier_serialized = base64.b64decode(model_dict['classifier'])

# Load the vectorizer and classifier
vectorizer = pickle.loads(vectorizer_serialized)
classifier = pickle.loads(classifier_serialized)

# Convert the example sentences into numerical features using the loaded vectorizer
vectorized_sentences = vectorizer.transform(example_sentences)

# Make predictions using the loaded classifier
predictions = classifier.predict(vectorized_sentences)
probabilities = classifier.predict_proba(vectorized_sentences)

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
#rf = joblib.load('RF_cyberbullying_model.pkl')

# Load the JSON file containing the serialized vectorizer and classifier
with open('RF_cyberbullying_model.json', 'r') as json_file:
    model_dict = json.load(json_file)

# Decode the serialized objects
vectorizer_serialized = base64.b64decode(model_dict['vectorizer'])
classifier_serialized = base64.b64decode(model_dict['classifier'])

# Load the vectorizer and classifier
vectorizer = pickle.loads(vectorizer_serialized)
classifier = pickle.loads(classifier_serialized)

# Convert the example sentences into numerical features using the loaded vectorizer
vectorized_sentences = vectorizer.transform(example_sentences)

# Make predictions using the loaded classifier
predictions = classifier.predict(vectorized_sentences)
probabilities = classifier.predict_proba(vectorized_sentences)

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

# # Load the trained Gradient Boosting model
#gb = joblib.load('GB_cyberbullying_model.pkl')

# Load the JSON file containing the serialized vectorizer and classifier
with open('GB_cyberbullying_model.json', 'r') as json_file:
    model_dict = json.load(json_file)

# Decode the serialized objects
vectorizer_serialized = base64.b64decode(model_dict['vectorizer'])
classifier_serialized = base64.b64decode(model_dict['classifier'])

# Load the vectorizer and classifier
vectorizer = pickle.loads(vectorizer_serialized)
classifier = pickle.loads(classifier_serialized)

# Convert the example sentences into numerical features using the loaded vectorizer
vectorized_sentences = vectorizer.transform(example_sentences)

# Make predictions using the loaded classifier
predictions = classifier.predict(vectorized_sentences)
probabilities = classifier.predict_proba(vectorized_sentences)

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

# Load the JSON file containing the serialized classifier
with open('USE_cyberbullying_model.json', 'r') as json_file:
    model_dict = json.load(json_file)

# Decode the serialized objects
classifier_serialized = base64.b64decode(model_dict['classifier'])

# Load the classifier
classifier = pickle.loads(classifier_serialized)

# Vectorize the example sentences using the Universal Sentence Encoder
vectorized_sentences_use = embed(example_sentences).numpy()

# Make the predictions using the model's decision function
decision_values = classifier.decision_function(vectorized_sentences_use)

# Normalize the decision values to obtain probabilities
probabilities = np.exp(decision_values)
probabilities /= np.sum(probabilities, axis=1)[:, np.newaxis]

# Determine the predicted labels
predictions = np.argmax(probabilities, axis=1)

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
max_sequence_length = 849
X_padded = pad_sequences(X_seq, maxlen=max_sequence_length)

# Load the trained model
# gru = load_model('GRU_cyberbullying_model.h5')

# Load the JSON file containing the serialized vectorizer and classifier
with open('GRU_cyberbullying_model.json', 'r') as json_file:
    model_dict = json.load(json_file)

# Decode the serialized objects
classifier_serialized = base64.b64decode(model_dict['classifier'])

# Load the vectorizer and classifier
classifier = pickle.loads(classifier_serialized)

# Convert the example sentences into numerical features using the loaded vectorizer
vectorized_sentences = vectorizer.transform(example_sentences)
vectorized_sentences = vectorized_sentences.tocoo()
sorted_indices = np.lexsort((vectorized_sentences.row, vectorized_sentences.col))
vectorized_sentences = tf.sparse.reorder(tf.sparse.SparseTensor(
    indices=np.column_stack((vectorized_sentences.row[sorted_indices], vectorized_sentences.col[sorted_indices])),
    values=vectorized_sentences.data[sorted_indices],
    dense_shape=vectorized_sentences.shape
))

# Convert SparseTensor to a dense matrix and apply padding
vectorized_sentences = tf.sparse.to_dense(vectorized_sentences)
vectorized_sentences = pad_sequences(vectorized_sentences, maxlen=max_sequence_length)

# Make predictions using the loaded classifier
predictions = classifier.predict(vectorized_sentences)
probabilities = classifier.predict(vectorized_sentences)

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
#lgbm = joblib.load('LightGBM_cyberbullying_model.pkl')

# Load the JSON file containing the serialized vectorizer and classifier
with open('LightGBM_cyberbullying_model.json', 'r') as json_file:
    model_dict = json.load(json_file)

# Decode the serialized objects
vectorizer_serialized = base64.b64decode(model_dict['vectorizer'])
classifier_serialized = base64.b64decode(model_dict['classifier'])

# Load the vectorizer and classifier
vectorizer = pickle.loads(vectorizer_serialized)
classifier = pickle.loads(classifier_serialized)

# Convert the example sentences into numerical features using the loaded vectorizer
vectorized_sentences = vectorizer.transform(example_sentences)

# Make predictions using the loaded classifier
predictions = classifier.predict(vectorized_sentences)
probabilities = classifier.predict_proba(vectorized_sentences)

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
