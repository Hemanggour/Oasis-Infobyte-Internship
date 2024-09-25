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
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date']) 
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

    mode_gender = df['Gender'].mode()[0] if 'Gender' in df.columns else None
    mode_product_category = df['Product Category'].mode()[0] if 'Product Category' in df.columns else None
    
    if mode_gender:
        print("\nMode of Gender:", mode_gender)
    if mode_product_category:
        print("Mode of Product Category:", mode_product_category)

    return mean_values, median_values, mode_gender, mode_product_category, std_dev_values


def time_series_analysis(df):

    monthly_sales = df.resample('ME', on='Date').sum() 
    
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales.index, monthly_sales['Total Amount'], marker='o')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Sales Amount')
    plt.grid(True)
    plt.show()

def customer_product_analysis(df):

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Gender')
    plt.title('Gender Distribution')
    plt.show()
    
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='Age', bins=15, kde=True)
    plt.title('Age Distribution')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Product Category')
    plt.title('Product Category Distribution')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Product Category', y='Total Amount', hue='Gender')
    plt.title('Purchasing Behavior by Gender and Product Category')
    plt.show()

def plot_correlation_heatmap(df):

    numeric_data = df.select_dtypes(include=['number'])
    corr_matrix = numeric_data.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def generate_recommendations():
    recommendations = """
    1. Focus on marketing campaigns targeted at the most frequent buyers identified by gender, age, and product category.
    2. Enhance inventory for the most popular product categories during peak sales months.
    3. Implement personalized offers for the age groups with the highest spending.
    4. Utilize the insights from time series analysis to optimize sales strategies during high-demand periods.
    """
    print("Recommendations:\n", recommendations)

def main():

    file_path = 'retail_sales_dataset.csv' 
    retail_sales_data = load_data(file_path)
    retail_sales_data = clean_data(retail_sales_data)
    descriptive_stats(retail_sales_data)
    time_series_analysis(retail_sales_data)
    customer_product_analysis(retail_sales_data)
    plot_correlation_heatmap(retail_sales_data)
    generate_recommendations()

if __name__ == "__main__":
    main()
