import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    print("Missing Values:\n", df.isnull().sum())
    df_cleaned = df.dropna() 
    return df_cleaned

def descriptive_stats(df):

    numeric_cols = df.select_dtypes(include=[np.number])

    mean_values = numeric_cols.mean()
    print("\nMean:\n", mean_values)

    median_values = numeric_cols.median()
    print("\nMedian:\n", median_values)

    print(f"\nMode:\n{numeric_cols.mode().iloc[0]}")

    std_dev_values = numeric_cols.std()
    print("\nStandard Deviation:\n", std_dev_values)

def menu_item_analysis(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Category', y='Calories', data=df, estimator=np.mean)
    plt.title('Average Calories by Category')
    plt.xticks(rotation=90)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Category', y='Total Fat', data=df)
    plt.title('Fat Content Distribution by Category')
    plt.xticks(rotation=90)
    plt.show()

def nutritional_analysis(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Total Fat', y='Calories', data=df)
    plt.title('Calories vs Total Fat')
    plt.xlabel('Total Fat (g)')
    plt.ylabel('Calories')
    plt.show()

    numeric_data = df.select_dtypes(include=['number'])
    corr_matrix = numeric_data.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Nutritional Data Correlation Matrix Heatmap')
    plt.show()

def generate_recommendations():
    recommendations = """
    1. Focus on promoting items with lower calorie content within each category.
    2. Consider reformulating menu items with high levels of total fat and calories.
    3. Highlight the nutritional benefits of items with balanced calorie and fat content.
    4. Provide healthier alternatives in categories with the highest calorie averages.
    """
    print("Recommendations:\n", recommendations)

def main():
    file_path = 'menu.csv' 
    menu_data = load_data(file_path)
    menu_data_cleaned = clean_data(menu_data)
    descriptive_stats(menu_data_cleaned)
    menu_item_analysis(menu_data_cleaned)
    nutritional_analysis(menu_data_cleaned)
    generate_recommendations()

if __name__ == "__main__":
    main()