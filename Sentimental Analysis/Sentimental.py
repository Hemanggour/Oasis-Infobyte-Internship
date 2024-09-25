import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
def load_data(filepath):
    """Load the dataset from CSV file."""
    df = pd.read_csv(filepath)
    print(f"Data loaded. Number of records: {df.shape[0]}")
    return df

# Handle missing values
def handle_missing_data(df):
    """Remove rows with missing values."""
    df = df.dropna(subset=['clean_text', 'category'])
    print(f"Data after removing missing values: {df.shape[0]} rows")
    return df

# Preprocess the dataset (since 'clean_text' already exists, no need to clean again)
def preprocess_data(df):
    """Ensure the dataset is clean and ready for feature extraction."""
    return df

# Feature extraction using TF-IDF
def extract_features(df):
    """Convert cleaned text into numerical features using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['category'].astype(int)
    print("Feature extraction completed.")
    return X, y, vectorizer

# Split the dataset into train and test sets
def split_data(X, y):
    """Split the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test

# Train the Naive Bayes model
def train_naive_bayes(X_train, y_train):
    """Train a Naive Bayes classifier."""
    model_nb = MultinomialNB()
    model_nb.fit(X_train, y_train)
    print("Naive Bayes model trained.")
    return model_nb

# Train the LinearSVC model (Faster SVM alternative)
def train_svm(X_train, y_train):
    """Train a Support Vector Machine classifier."""
    model_svm = LinearSVC()
    model_svm.fit(X_train, y_train)
    print("SVM model (LinearSVC) trained.")
    return model_svm

# Evaluate the model performance
def evaluate_model(model, X_test, y_test):
    """Evaluate the model using accuracy and confusion matrix."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    return cm

# Plot confusion matrix
def plot_confusion_matrix(cm, title):
    """Plot the confusion matrix."""
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Plot sentiment distribution
def plot_sentiment_distribution(df):
    """Plot the sentiment distribution in the dataset."""
    sns.countplot(x='category', data=df)
    plt.title('Sentiment Distribution')
    plt.show()

# Main function to run the sentiment analysis pipeline
def run_sentiment_analysis(filepath):
    """Main pipeline function for sentiment analysis."""
    df = load_data(filepath)
    
    df = handle_missing_data(df)
    
    plot_sentiment_distribution(df)
    
    df = preprocess_data(df)
    
    X, y, vectorizer = extract_features(df)
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Training Naive Bayes model...")
    model_nb = train_naive_bayes(X_train, y_train)
    
    print("Training LinearSVC model (faster SVM alternative)...")
    model_svm = train_svm(X_train, y_train)

    print("Evaluating Naive Bayes model...")
    cm_nb = evaluate_model(model_nb, X_test, y_test)
    plot_confusion_matrix(cm_nb, "Confusion Matrix - Naive Bayes")
    
    print("Evaluating LinearSVC model...")
    cm_svm = evaluate_model(model_svm, X_test, y_test)
    plot_confusion_matrix(cm_svm, "Confusion Matrix - LinearSVC")

if __name__=='__main__':
    file_path = 'Twitter_Data.csv'
    run_sentiment_analysis(file_path)
