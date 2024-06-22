import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle

# Custom transformer for extracting email length
class EmailLengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    
    def transform(self, emails):
        return np.array([[len(email)] for email in emails])

# Custom transformer for extracting number of links
class LinksCountExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    
    def transform(self, emails):
        return np.array([[len(re.findall(r'http[s]?://', email))] for email in emails])

# Custom transformer for extracting number of special characters
class SpecialCharCountExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    
    def transform(self, emails):
        return np.array([[len(re.findall(r'[^a-zA-Z0-9\s]', email))] for email in emails])

# Custom transformer for extracting spam keywords
class SpamKeywordCountExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, spam_keywords):
        self.spam_keywords = spam_keywords
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, emails):
        return np.array([[sum(email.lower().count(keyword) for keyword in self.spam_keywords)] for email in emails])

# Load the dataset from CSV
def load_dataset(filename, encoding='utf-8'):
    df = pd.read_csv(filename, encoding=encoding, on_bad_lines='skip')
    df['label'] = df['type'].apply(lambda x: 1.0 if x == 'spam' else 0.0)
    return df

# Split the dataset into training and testing sets
def split_dataset(df, split_ratio):
    train_df, test_df = train_test_split(df, test_size=(1-split_ratio), stratify=df['label'], random_state=42)
    return train_df, test_df

# Preprocess the input email
def preprocess_input_email(input_email, vectorizer):
    input_email = str(input_email).lower()
    input_email = re.sub(r'[^a-z\s]', '', input_email)
    return vectorizer.transform([input_email]), input_email.split()

# Train the model with SMOTE
def train_model(train_df):
    spam_keywords = ['free', 'win', 'winner', 'urgent', 'click', 'offer']
    
    combined_features = FeatureUnion([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('email_length', Pipeline([
            ('extractor', EmailLengthExtractor()),
            ('scaler', MinMaxScaler())
        ])),
        ('links_count', Pipeline([
            ('extractor', LinksCountExtractor()),
            ('scaler', MinMaxScaler())
        ])),
        ('special_char_count', Pipeline([
            ('extractor', SpecialCharCountExtractor()),
            ('scaler', MinMaxScaler())
        ])),
        ('spam_keyword_count', Pipeline([
            ('extractor', SpamKeywordCountExtractor(spam_keywords)),
            ('scaler', MinMaxScaler())
        ]))
    ])
    
    X_train = combined_features.fit_transform(train_df['email'])
    y_train = train_df['label']
    
    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    model = MultinomialNB()
    model.fit(X_train_resampled, y_train_resampled)
    return model, combined_features

# Retrain the model with additional data
def retrain_model_with_additional_data(df, additional_data):
    additional_df = pd.DataFrame(additional_data, columns=['email', 'label'])
    combined_df = pd.concat([df, additional_df], ignore_index=True)
    train_df, test_df = split_dataset(combined_df, 0.8)
    model, vectorizer = train_model(train_df)
    return model, vectorizer, test_df

# Initialize session state variables
if 'additional_data' not in st.session_state:
    st.session_state.additional_data = []

if 'model' not in st.session_state:
    filename = 'modified_email_spam_latest.csv'
    df = load_dataset(filename)
    train_df, test_df = split_dataset(df, 0.8)
    st.session_state.model, st.session_state.vectorizer = train_model(train_df)
    st.session_state.df = df
    st.session_state.test_df = test_df

# Function to get the most influential words
def get_influential_words(model, vectorizer, email_words):
    tfidf = vectorizer.transformer_list[0][1]
    feature_names = np.array(tfidf.get_feature_names_out())
    email_word_indices = tfidf.transform([" ".join(email_words)]).nonzero()[1]
    class_log_prob = model.feature_log_prob_

    influential_words = {}
    for class_index, class_name in enumerate(["HAM", "SPAM"]):
        class_influences = {word: class_log_prob[class_index, idx] for word, idx in zip(feature_names[email_word_indices], email_word_indices)}
        influential_words[class_name] = sorted(class_influences.items(), key=lambda item: item[1], reverse=True)
    
    return influential_words

# Function to highlight influential words
def highlight_influential_words(email_text, influential_words, result):
    highlighted_text = email_text
    words_to_highlight = [word for word, _ in influential_words["SPAM"]] if result == 1.0 else [word for word, _ in influential_words["HAM"]]
    
    for word in words_to_highlight:
        highlighted_text = re.sub(f"\\b{word}\\b", f"<span style='color: {'red' if result == 1.0 else 'green'}; font-weight: bold;'>{word}</span>", highlighted_text, flags=re.IGNORECASE)
    
    return highlighted_text

# Main function to run the spam mail detection
def main():
    st.title("Spam Mail Detection")

    action = st.sidebar.selectbox("Choose Action", ["Test an Email", "Add Training Data", "Evaluate Model"])

    if action == "Test an Email":
        st.write("Enter an email below to determine if it's spam or not:")
        input_email = st.text_area("Enter an email:")
        if st.button("Check"):
            input_vector, email_words = preprocess_input_email(input_email, st.session_state.vectorizer)
            result = st.session_state.model.predict(input_vector)[0]
            st.write("Prediction: ", "SPAM" if result == 1.0 else "HAM")

            influential_words = get_influential_words(st.session_state.model, st.session_state.vectorizer, email_words)
            highlighted_text = highlight_influential_words(input_email, influential_words, result)
            st.markdown(highlighted_text, unsafe_allow_html=True)

    elif action == "Add Training Data":
        st.write("Enter an email and label it as spam or not spam to add to the training data:")
        input_email = st.text_area("Enter an email:")
        label = st.radio("Label this email as:", ('Not Spam', 'Spam'))
        label = 1.0 if label == 'Spam' else 0.0
        if st.button("Add to Training Data"):
            st.session_state.additional_data.append((input_email, label))
            st.success("Email added to training data!")
        
        if st.session_state.additional_data:
            if st.button("Train Model with Additional Data"):
                st.session_state.model, st.session_state.vectorizer, st.session_state.test_df = retrain_model_with_additional_data(st.session_state.df, st.session_state.additional_data)
                st.session_state.additional_data = []  # Clear additional data after training
                st.success("Model retrained successfully with additional data!")

    elif action == "Evaluate Model":
        st.write("Evaluating the model on the test set:")
        if 'test_df' in st.session_state:
            test_df = st.session_state.test_df
            X_test = st.session_state.vectorizer.transform(test_df['email'])
            y_test = test_df['label']
            y_pred = st.session_state.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Testing Accuracy: {accuracy * 100:.2f}%")
            st.text("Classification Report:\n" + classification_report(y_test, y_pred, target_names=["HAM", "SPAM"]))
        else:
            st.error("No test data available.")

if __name__ == "__main__":
    main()

#streamlit run Mail_detection_latest.py