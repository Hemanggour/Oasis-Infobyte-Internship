import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier

# 1. Load the dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# 2. Exploratory Data Analysis (EDA)
def explore_data(data):
    print(data.head())
    print(f"Shape of the data: {data.shape}")
    print("Missing values:\n", data.isna().sum())
    print(data.describe())

    print("Class distribution:\n", data['Class'].value_counts())
    sns.countplot(x="Class", data=data)
    plt.title("Class Distribution")
    plt.show()

# 3. Feature Engineering
def feature_engineering(data):

    data['Time'] = StandardScaler().fit_transform(data['Time'].values.reshape(-1, 1))
    data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    
    x = data.drop(columns=['Class', 'V1', 'V2', 'V3'])
    y = data['Class']
    
    return x, y

# 4. Handle Imbalanced Data
def balance_data(x, y):

    nm = NearMiss()
    x_resampled, y_resampled = nm.fit_resample(x, y)
    return x_resampled, y_resampled

# 5. Split the dataset
def split_data(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=11)

# 6. Train machine learning models
def train_models(x_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }
    
    trained_models = {}
    
    for name, model in models.items():
        model.fit(x_train, y_train)
        trained_models[name] = model
    
    return trained_models

# 7. Evaluate models
def evaluate_models(models, x_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model: {name}")
        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")
        print(f"Classification Report:\n {classification_report(y_test, y_pred)}")

# 8. Real-time Monitoring (Simulating real-time detection)
def real_time_monitoring(model, new_data, feature_names):

    new_data_df = pd.DataFrame(new_data, columns=feature_names)
    
    pred = model.predict(new_data_df)
    if pred == 1:
        print("Fraudulent activity detected!")
    else:
        print("Transaction appears normal.")
    return pred


# 9. Scalability consideration
def scalability_testing(data):

    large_data = pd.concat([data] * 10, ignore_index=True)
    print(f"New dataset shape after scalability testing: {large_data.shape}")
    return large_data

def main(filepath):

    data = load_data(filepath)

    explore_data(data)

    x, y = feature_engineering(data)

    x_balanced, y_balanced = balance_data(x, y)
    
    x_train, x_test, y_train, y_test = split_data(x_balanced, y_balanced)
    
    trained_models = train_models(x_train, y_train)
    
    evaluate_models(trained_models, x_test, y_test)
    
    large_data = scalability_testing(data)
    
    real_time_monitoring(trained_models['Logistic Regression'], x_test.iloc[0].values.reshape(1, -1), x_train.columns)

if __name__ == "__main__":
    main('creditcard.csv')
  
