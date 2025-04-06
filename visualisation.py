import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file with sentiment and other metrics
def load_results(results_file):
    return pd.read_csv(results_file)

# Function to plot sentiment scores by persona
def plot_sentiment_by_persona(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='persona', y='sentiment_score', data=df)
    plt.title('Sentiment Scores by Persona')
    plt.xlabel('Persona')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to plot sentiment scores by task
def plot_sentiment_by_task(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='task', y='sentiment_score', data=df)
    plt.title('Sentiment Scores by Task')
    plt.xlabel('Task')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to plot satisfaction score distribution
def plot_satisfaction_score(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='satisfaction_score', data=df, palette='Blues')
    plt.title('Satisfaction Score Distribution')
    plt.xlabel('Satisfaction Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# Function to plot response length distribution
def plot_response_length(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['response_length'], bins=20, kde=True, color='skyblue')
    plt.title('Response Length Distribution')
    plt.xlabel('Response Length (Number of Words)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Function to plot response time distribution
def plot_response_time(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['response_time'], bins=20, kde=True, color='lightgreen')
    plt.title('Response Time Distribution')
    plt.xlabel('Response Time (seconds)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Function to analyze the relationship between sentiment score and response length
def plot_sentiment_vs_length(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='response_length', y='sentiment_score', data=df, hue='persona', palette='Set1')
    plt.title('Sentiment Score vs Response Length')
    plt.xlabel('Response Length (Number of Words)')
    plt.ylabel('Sentiment Score')
    plt.tight_layout()
    plt.show()

# Function to analyze the relationship between sentiment score and response time
def plot_sentiment_vs_time(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='response_time', y='sentiment_score', data=df, hue='persona', palette='Set1')
    plt.title('Sentiment Score vs Response Time')
    plt.xlabel('Response Time (seconds)')
    plt.ylabel('Sentiment Score')
    plt.tight_layout()
    plt.show()

# Main function to generate visualizations
def main():
    results_file = "test_results_with_sentiment.csv"  # Specify the path to your CSV file

    # Load the data
    df = load_results(results_file)

    # Plot the different visualizations
    plot_sentiment_by_persona(df)
    plot_sentiment_by_task(df)
    plot_satisfaction_score(df)
    plot_response_length(df)
    plot_response_time(df)
    plot_sentiment_vs_length(df)
    plot_sentiment_vs_time(df)

if __name__ == "__main__":
    main()
