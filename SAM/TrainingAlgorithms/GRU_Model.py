import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dropout, Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import time

# Load the dataset from the CSV file
data = pd.read_csv("cyberbullying_tweets.csv")

# Separate the features (tweet_text) and labels (cyberbullying_type)
X = data['tweet_text'].values
y = data['cyberbullying_type'].values

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

# Convert text sequences to numerical sequences
X_seq = tokenizer.texts_to_sequences(X)
max_sequence_length = max([len(seq) for seq in X_seq])
X_padded = pad_sequences(X_seq, maxlen=max_sequence_length)

# Split the data into training and testing sets with a fixed random seed
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Define the model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
hidden_units = 64

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
model.add(GRU(hidden_units))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

# Train the model with early stopping
start_time = time.time()  # Start measuring the duration
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=8, batch_size=32, callbacks=[early_stopping])
end_time = time.time()  # Stop measuring the duration
duration_seconds = end_time - start_time
duration_minutes = duration_seconds / 60

# Save the trained model
model.save('GRU_cyberbullying_model.h5')

# Evaluate the performance of the GRU
print("Classification Report - GRU:")

# Define the target names corresponding to each class
target_names = ['age', 'ethnicity', 'gender', 'not_cyberbullying', 'other_cyberbullying', 'religion']

predictions_gru = model.predict(X_test)

# Generate the classification report with custom target names
report = classification_report(y_test, np.argmax(predictions_gru, axis=1), target_names=target_names)
print(report)

# Calculate accuracy and error percentage
accuracy_gru = (np.argmax(predictions_gru, axis=1) == y_test).mean() * 100
error_percentage_gru = 100 - accuracy_gru

# Display duration, accuracy, and error percentage
print("Duration: {:.2f} minutes".format(duration_minutes))
print("Accuracy: {:.2f}%".format(accuracy_gru))
print("Error Percentage: {:.2f}%".format(error_percentage_gru))
print()