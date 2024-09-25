import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")


# Function to load the dataset
def load_data(apps_file_path, reviews_file_path=None):
    apps_with_duplicates = pd.read_csv(apps_file_path)
    apps = apps_with_duplicates.drop_duplicates()
    if reviews_file_path:
        reviews_df = pd.read_csv(reviews_file_path)
        return apps, reviews_df
    return apps


# Function to display basic dataset info
def display_basic_info(apps):
    print(f'Total number of apps in the dataset = {len(apps)}')
    print(apps.info())
    n = 5
    print(apps.sample(n))


# Function to clean 'Installs' and 'Price' columns
def clean_columns(apps, chars_to_remove, cols_to_clean):
    for col in cols_to_clean:
        for char in chars_to_remove:
            apps[col] = apps[col].astype(str).str.replace(char, '')
        apps[col] = pd.to_numeric(apps[col])
    return apps


# Function to plot the number of apps in each category using Matplotlib
def plot_category_distribution(apps):
    num_categories = len(apps['Category'].unique())
    print('Number of categories = ', num_categories)

    num_apps_in_category = apps['Category'].value_counts().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    num_apps_in_category.plot(kind='bar')
    plt.title('Number of Apps in Each Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Apps')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# Function to plot rating distribution
def plot_rating_distribution(apps):
    avg_app_rating = apps['Rating'].mean()
    print('Average app rating = ', avg_app_rating)
    
    plt.figure(figsize=(10, 6))
    plt.hist(apps['Rating'].dropna(), bins=30, alpha=0.7, color='b')
    plt.axvline(avg_app_rating, color='r', linestyle='dashdot', label=f'Avg Rating: {avg_app_rating:.2f}')
    plt.title('App Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Number of Apps')
    plt.legend()
    plt.show()


# Function to filter rows with 'Rating' and 'Size' present
def filter_apps_with_size_and_rating(apps):
    return apps[(~apps['Rating'].isnull()) & (~apps['Size'].isnull())]


# Function to plot size vs rating for large categories using a hex plot
def plot_size_vs_rating(apps):
    large_categories = apps.groupby('Category').filter(lambda x: len(x) >= 250).reset_index()
    plt.figure(figsize=(10, 6))
    sns.jointplot(x=large_categories['Size'], y=large_categories['Rating'], kind='hex', gridsize=30, cmap='Blues')
    plt.show()


# Function to plot price vs rating for paid apps
def plot_price_vs_rating(apps):
    paid_apps = apps[apps['Type'] == 'Paid']
    plt.figure(figsize=(10, 6))
    sns.jointplot(x=paid_apps['Price'], y=paid_apps['Rating'], kind='scatter')
    plt.show()


# Function to plot pricing trends
def plot_pricing_trend(apps):
    popular_app_cats = apps[apps["Category"].isin(['GAME', 'FAMILY', 'PHOTOGRAPHY', 'MEDICAL', 'TOOLS', 'FINANCE', 'LIFESTYLE', 'BUSINESS'])]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.stripplot(x='Price', y='Category', data=popular_app_cats, jitter=True, linewidth=1)
    ax.set_title('App Pricing Trend Across Categories')
    plt.show()

    apps_above_200 = popular_app_cats[popular_app_cats['Price'] > 200]
    print(apps_above_200[['Category', 'App', 'Price']])

    apps_under_100 = popular_app_cats[popular_app_cats['Price'] < 100]
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.stripplot(x='Price', y='Category', data=apps_under_100, jitter=True, linewidth=1)
    ax.set_title('App Pricing Trend (Filtered for Apps Below $100)')
    plt.show()


# Function to plot number of downloads for paid vs free apps
def plot_downloads_vs_app_type(apps):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Type', y='Installs', data=apps, showfliers=False)
    plt.yscale('log')
    plt.title('Number of Downloads of Paid Apps vs Free Apps')
    plt.ylabel('Number of Installs (Log Scale)')
    plt.show()


# Function to analyze user sentiment from merged app and review data
def analyze_sentiment(merged_df):
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.boxplot(x='Type', y='Sentiment_Polarity', data=merged_df)
    ax.set_title('Sentiment Polarity Distribution (Paid vs Free Apps)')
    plt.show()


# Main function to execute all steps
def main(apps_file_path, reviews_file_path=None):
    if reviews_file_path:
        apps, reviews_df = load_data(apps_file_path, reviews_file_path)
        merged_df = pd.merge(apps, reviews_df, on='App', how='inner').dropna(subset=['Sentiment', 'Translated_Review'])
    else:
        apps = load_data(apps_file_path)
        merged_df = None

    display_basic_info(apps)

    chars_to_remove = [',', '$', '+']
    cols_to_clean = ['Installs', 'Price']
    apps = clean_columns(apps, chars_to_remove, cols_to_clean)

    plot_category_distribution(apps)
    plot_rating_distribution(apps)

    apps_with_size_and_rating = filter_apps_with_size_and_rating(apps)
    plot_size_vs_rating(apps_with_size_and_rating)

    plot_price_vs_rating(apps_with_size_and_rating)
    plot_pricing_trend(apps)

    plot_downloads_vs_app_type(apps)

    if merged_df is not None:
        analyze_sentiment(merged_df)


# Execute main function
if __name__ == "__main__":
    apps_file_path = "apps.csv"
    reviews_file_path = "user_reviews.csv"
    main(apps_file_path, reviews_file_path)
