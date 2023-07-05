import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from win10toast import ToastNotifier
import joblib
import time
import json
import pickle
import base64
import glob
import re
import spacy
import csv

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def train_rf_model():
    # Load the dataset from the CSV file
    data = pd.read_csv("cyberbullying_tweets.csv")

    # Split the dataset into features (X) and labels (y)
    X = data['tweet_text'] 
    y = data['cyberbullying_type'] 

    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Convert the text data into numerical features using TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Send alert for model creation
    send_alert_created_model("RF model has been created.")

    # Train a Random Forest classifier
    print("Random Forest")
    start_time = time.time()
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    end_time = time.time()  # Stop measuring the duration
    duration = end_time - start_time
    duration_minutes = duration / 60

    # Convert the classifier and vectorizer to serialized representations
    vectorizer_serialized = base64.b64encode(pickle.dumps(vectorizer)).decode('utf-8')
    classifier_serialized = base64.b64encode(pickle.dumps(rf)).decode('utf-8')

    # Create a dictionary to hold the serialized objects
    model_dict = {
        'vectorizer': vectorizer_serialized,
        'label_encoder': label_encoder.classes_.tolist(),
        'classifier': classifier_serialized
    }

    current_datetime = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Save the model dictionary as a JSON file
    with open(f'RF_Auto_cyberbullying_model_{current_datetime}.json', 'w') as json_file:
        json.dump(model_dict, json_file)

    # Save the trained model
    joblib.dump(rf, f'RF_Auto_cyberbullying_model_{current_datetime}.pkl')

    # Send alert for model save
    send_alert_saved_model("RF Model has been saved.")

    # Evaluate the RF model
    rf_predictions = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions) * 100
    rf_error_percentage = 100 - rf_accuracy

    # Display duration, accuracy, and error percentage
    print('RF Classification Report:')
    print("Duration: {:.2f} minutes".format(duration_minutes))
    print("Accuracy: {:.2f}%".format(rf_accuracy))
    print("Error Percentage: {:.2f}%".format(rf_error_percentage))
    print()

    # Anonymize the text in the test set
    X_test_anonymized = [anonymize_text(text) for text in X]

    # Create lists to store the data
    original_texts = []
    original_labels = []
    anonymized_texts = []
    anonymized_labels = []

    # Iterate over the data and separate into original and anonymized lists
    for original_text, anonymized_text, label in zip(X, X_test_anonymized, y):
        if original_text != anonymized_text:
            anonymized_texts.append(anonymized_text)
            anonymized_labels.append(label)
        else:
            original_texts.append(original_text)
            original_labels.append(label)

    # Save the data to a CSV file
    with open('anonymization_cyberbullying_tweets_RF.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['tweet_text', 'cyberbullying_type'])

        for original_text, original_label in zip(original_texts, original_labels):
            # Convert the original_label to the corresponding integer key
            writer.writerow([original_text, original_label])

        for anonymized_text, anonymized_label in zip(anonymized_texts, anonymized_labels):
            # Convert the anonymized_label to the corresponding integer key
            writer.writerow([anonymized_text, anonymized_label])

    # Send alert if accuracy is below 80%
    if rf_accuracy < 80:
        send_alert_accuracy("Model accuracy dropped below 80%.")

def integrate_models(model_files):
    # Load the RF model
    rf_model = joblib.load(model_files[0])

    # Get a list of matching JSON file paths
    json_files = glob.glob('RF_Auto_cyberbullying_model_*.json')

    # Sort the file paths based on the date in the file name
    sorted_json_files = sorted(json_files, key=lambda x: x.split('_')[-1].split('.')[0], reverse=True)

    # Check if any JSON file exists
    if sorted_json_files:
        # Open the most recent JSON file
        most_recent_json_file = sorted_json_files[0]

        with open(most_recent_json_file, 'r') as json_file:
            rf_model_dict = json.load(json_file)
            rf_vectorizer_serialized = rf_model_dict['vectorizer']
            rf_vectorizer = pickle.loads(base64.b64decode(rf_vectorizer_serialized))
            rf_label_encoder = LabelEncoder()
            rf_label_encoder.classes_ = np.array(rf_model_dict['label_encoder'])
    else:
        # Handle the case when no JSON file exists
        print("No JSON file found.")
        return

    # Load other models
    other_models = []
    for file_name in model_files[1:]:
        model = joblib.load(file_name)
        other_models.append(model)

    # Example new tweets
    new_tweets = [
        "I need a gif of a cat laughing.",  # not_cyberbullying
        "I hate to call her a bitch, would one of my female followers do it for me?",  # gender
        "i’d let the beat bully me in school"  # age
    ]

    # Preprocess new data
    new_tweets_vectorized = rf_vectorizer.transform(new_tweets)

    # Aggregate predictions from all models
    all_predictions = []

    # Obtain predictions from the RF model
    rf_predictions = rf_model.predict(new_tweets_vectorized)
    all_predictions.append(rf_predictions)

    # Obtain predictions from other models
    for model in other_models:
        predictions = model.predict(new_tweets_vectorized)
        all_predictions.append(predictions)

    # Process and aggregate predictions
    final_predictions = []

    # Aggregate predictions from all models
    for predictions in zip(*all_predictions):
        majority_vote = np.argmax(np.bincount(predictions))
        final_predictions.append(majority_vote)

    # Convert final predictions to numpy array
    final_predictions = np.array(final_predictions)

    # Define the true labels for the new_tweets
    y_true = ['not_cyberbullying', 'gender', 'age']  # Replace with the true labels for your new_tweets

    # Define the thresholds for decision rules
    thresholds = [0.7, 0.6, 0]

   # Apply decision rules and thresholds
    threshold_indices = [3, 2, 0]  # Indices of the sentiment labels to apply decision rules and thresholds
    
   # Get the sentiment labels from rf_label_encoder
    label_encoder_labels = rf_label_encoder.classes_

    # Print the threshold results
    # Check if the lengths of thresholds and threshold_indices are the same
    if len(thresholds) == len(threshold_indices):
        # Print the threshold results
        for i in range(len(threshold_indices)):
            # Get the corresponding sentiment label
            sentiment_label = label_encoder_labels[threshold_indices[i]]

            # Check if the prediction exceeds the threshold
            if i < len(final_predictions):
                if final_predictions[i] >= thresholds[i]:
                    threshold_result = "Positive"
                else:
                    threshold_result = "Negative"

                # Print the threshold result
                print("Threshold Result (",thresholds[i],") for sentiment label", sentiment_label, ":", threshold_result)
    else:
        print("Mismatched lengths of threshold_indices and thresholds. Please ensure they have the same length.")


    # Convert final predictions to corresponding labels
    final_predictions = rf_label_encoder.inverse_transform(final_predictions)

    # Calculate and print the accuracy
    accuracy = sum(y_t == y_p for y_t, y_p in zip(y_true, final_predictions)) / len(y_true)
    print("Final Accuracy: {:.2f}%".format(accuracy * 100))
    print()


    # Collect user feedback
    user_feedback = collect_user_feedback(new_tweets, final_predictions)

    # Print the final predictions
    for tweet, prediction, true_label in zip(new_tweets, final_predictions, y_true):
        print("Tweet:", tweet)
        print("Prediction:", prediction)
        print("True Label:", true_label)
        print()

    # Refine the models based on user feedback
    refine_models(user_feedback, y_true)

def collect_user_feedback(new_tweets, predictions):
    # Collect user feedback on sentiment predictions
    user_feedback = {}

    for tweet, prediction in zip(new_tweets, predictions):
        print("Tweet:", tweet)
        print("Prediction:", prediction)

        # Prompt the user to provide feedback
        feedback = input("Please provide feedback for the prediction (correct/incorrect): ")

        # Validate user input
        while feedback.lower() not in ['correct', 'incorrect']:
            print("Invalid feedback. Please enter 'correct' or 'incorrect'.")
            feedback = input("Please provide feedback for the prediction (correct/incorrect): ")

        # Store the feedback in the user_feedback dictionary
        user_feedback[tweet] = {'prediction': prediction, 'feedback': feedback.lower()}

        print()

    return user_feedback

def refine_models(user_feedback, y_true):
    # Create a new DataFrame to store the corrected tweets and labels
    corrected_df = pd.DataFrame(columns=['tweet_text', 'cyberbullying_type'])

    for tweet, feedback_data in user_feedback.items():
        prediction = feedback_data['prediction']
        feedback = feedback_data['feedback']

        if feedback == 'correct':
            corrected_tweet = tweet
            correct_label = prediction

            # Add the corrected tweet and label to the corrected_df DataFrame
            corrected_df = corrected_df._append({'tweet_text': corrected_tweet, 'cyberbullying_type': correct_label}, ignore_index=True)
        elif feedback == 'incorrect':
            corrected_tweet = tweet
            index = list(user_feedback.keys()).index(tweet)
            correct_label = y_true[index]

            # Add the corrected tweet and label to the corrected_df DataFrame
            corrected_df = corrected_df._append({'tweet_text': corrected_tweet, 'cyberbullying_type': correct_label}, ignore_index=True)

    # Load the original DataFrame
    updated_df = pd.read_csv('cyberbullying_tweets.csv')

    # Update the labels in the original DataFrame based on the corrected tweets
    for _, row in corrected_df.iterrows():
        tweet_text = row['tweet_text']
        cyberbullying_type = row['cyberbullying_type']

        mask = updated_df['tweet_text'] == tweet_text
        updated_df.loc[mask, 'cyberbullying_type'] = cyberbullying_type

    # Save the updated DataFrame back to the CSV file
    updated_df.to_csv('cyberbullying_tweets.csv', index=False)

    # Retrain the RF model
    train_rf_model()


def send_alert_accuracy(message):
    # Create a ToastNotifier object
    toaster = ToastNotifier()

    # Display a popup notification
    toaster.show_toast("ALERT Accuracy", message, duration=10)

def send_alert_created_model(message):
    # Create a ToastNotifier object
    toaster = ToastNotifier()

    # Display a popup notification
    toaster.show_toast("Model Created", message, duration=10)

def send_alert_saved_model(message):
    # Create a ToastNotifier object
    toaster = ToastNotifier()

    # Display a popup notification
    toaster.show_toast("Model Saved", message, duration=10)

# Anonymize the tweet text
def anonymize_text(text):
    # Replace usernames with "@username"
    text = re.sub(r'@\w+', '@username', text)

    # Replace email addresses with "email@example.com"
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', 'email@example.com', text)

    # Replace phone numbers with "XXX-XXX-XXXX"
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'XXX-XXX-XXXX', text)

    # Anonymize named entities (e.g., names) using spaCy
    doc = nlp(text)
    anonymized_tokens = []

    for token in doc:
        if token.ent_type_ == "PERSON":
            anonymized_tokens.append("PersonName")
        else:
            anonymized_tokens.append(token.text_with_ws)

    text = "".join(anonymized_tokens)
    
    return text

# Monitoring and Alerts
def monitor_and_alert(predictions, y_test):
    # Implement monitoring and alert system to track the performance and output of the predictions
    # You can define conditions for triggering alerts based on specific metrics or changes in predictions
    accuracy = (predictions == y_test).mean()
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    if accuracy < 0.8:
        send_alert_accuracy("Model accuracy dropped below 80%.")

def continuous_training():
    while True:
        # Load the existing dataset
        existing_df = pd.read_csv('cyberbullying_tweets.csv')

        # Check if new data is added
        if is_new_data_added(existing_df):
            # Generate the new dataset file name with the current date and hour
            current_datetime = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            new_dataset_file = f'cyberbullying_tweets_{current_datetime}.csv'

            # Save the new data in the new dataset file
            current_df = pd.read_csv('cyberbullying_tweets.csv')
            current_df.to_csv(new_dataset_file, index=False)

            # Perform continuous training
            train_rf_model()
        else:
            print("No new data added. Waiting for new data...")

        # Wait for a certain period of time before checking for new data again
        time.sleep(3600)  # Wait for 1 hour

def is_new_data_added(existing_df):
    # Implement your logic to check if new data is added
    # You can compare the current state of the dataset with the existing state
    # For example, you can compare the number of rows or check for specific changes in the dataset
    # Return True if new data is added, False otherwise
    current_df = pd.read_csv('cyberbullying_tweets.csv')
    if len(current_df) > len(existing_df):
        return True
    return False

# Train the RF model
train_rf_model()

# Get a list of matching model file paths
model_files = glob.glob('RF_Auto_cyberbullying_model_*.pkl')

# Extract the timestamps from the file names and create a list of tuples (timestamp, file_path)
timestamps = [(re.search(r'(\d{8}_\d{6})', file_name).group(), file_name) for file_name in model_files]

# Sort the timestamps in reverse order
sorted_timestamps = sorted(timestamps, key=lambda x: x[0], reverse=True)

# Select the last three model files
models = [file_path for _, file_path in sorted_timestamps[:3]]

integrate_models(models)

# Start the continuous training process
continuous_training()