import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_file):
    """
    Loads the CSV results produced by sentiment_analysis.py and returns a pandas DataFrame.
    """
    df = pd.read_csv(results_file)
    return df

def plot_demographic_parity(df):
    """
    Generates a bar plot comparing the rate of favorable outcomes (high satisfaction)
    across different demographic groups (e.g., personas).
    """
    # Calculate the favorable rate (satisfaction score 'High')
    favorable_rate = df.groupby('persona')['satisfaction_score'].apply(lambda x: (x == 'High').mean()).reset_index()
    favorable_rate.columns = ['persona', 'favorable_rate']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='persona', y='favorable_rate', data=favorable_rate, palette='viridis')
    plt.title('Favorable Outcome Rate by Persona')
    plt.xlabel('Persona')
    plt.ylabel('Favorable Outcome Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_disparate_impact(df):
    """
    Generates a bar plot to show disparate impact between demographic groups.
    Applies the 4/5ths rule to highlight groups with disparity.
    """
    # Calculate the favorable outcome rate per persona
    favorable_rate = df.groupby('persona')['satisfaction_score'].apply(lambda x: (x == 'High').mean()).reset_index()
    favorable_rate.columns = ['persona', 'favorable_rate']
    
    max_rate = favorable_rate['favorable_rate'].max()
    favorable_rate['impact_ratio'] = favorable_rate['favorable_rate'] / max_rate
    favorable_rate['disparate_impact'] = favorable_rate['impact_ratio'] < 0.8
    
    # Plot impact ratio with flagged disparity
    plt.figure(figsize=(10, 6))
    sns.barplot(x='persona', y='impact_ratio', data=favorable_rate, palette='coolwarm')
    plt.title('Impact Ratio by Persona (4/5ths Rule)')
    plt.xlabel('Persona')
    plt.ylabel('Impact Ratio')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Highlight disparity groups
    print("\nDisparate Impact (4/5ths Rule) Results:")
    print(favorable_rate[['persona', 'impact_ratio', 'disparate_impact']])

def plot_sentiment_by_task(df):
    """
    Generate a bar plot comparing sentiment scores across different tasks.
    """
    # Group by task and calculate mean sentiment score
    sentiment_by_task = df.groupby('task')['sentiment_score'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='sentiment_score', y='task', data=sentiment_by_task, palette='coolwarm')
    plt.title('Average Sentiment Score by Task')
    plt.xlabel('Average Sentiment Score')
    plt.ylabel('Task')
    plt.tight_layout()
    plt.show()

def plot_satisfaction_by_task(df):
    """
    Generate a bar plot comparing the satisfaction score across tasks.
    """
    # Calculate the satisfaction rate per task
    satisfaction_by_task = df.groupby('task')['satisfaction_score'].apply(lambda x: (x == 'High').mean()).reset_index()
    satisfaction_by_task.columns = ['task', 'satisfaction_rate']
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='satisfaction_rate', y='task', data=satisfaction_by_task, palette='viridis')
    plt.title('Satisfaction Rate by Task')
    plt.xlabel('Satisfaction Rate')
    plt.ylabel('Task')
    plt.tight_layout()
    plt.show()

def plot_response_time_by_persona(df):
    """
    Generate a boxplot to show response times by persona.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='persona', y='response_time', data=df, palette='viridis')
    plt.title('Response Time by Persona')
    plt.xlabel('Persona')
    plt.ylabel('Response Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_response_time_by_task(df):
    """
    Generate a boxplot to show response times by task.
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='task', y='response_time', data=df, palette='viridis')
    plt.title('Response Time by Task')
    plt.xlabel('Task')
    plt.ylabel('Response Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_satisfaction_distribution(df):
    """
    Generate a bar plot to show the distribution of satisfaction scores (High, Neutral, Low).
    """
    satisfaction_counts = df['satisfaction_score'].value_counts().reset_index()
    satisfaction_counts.columns = ['satisfaction_score', 'count']
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='satisfaction_score', y='count', data=satisfaction_counts, palette='viridis')
    plt.title('Satisfaction Score Distribution')
    plt.xlabel('Satisfaction Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def main():
    # Load the CSV results file generated from sentiment_analysis.py
    results_file = "test_results_2025-04-06_15-07-08_with_sentiment.csv"  # Change this to the actual file path
    df = pd.read_csv(results_file)

    # Perform the bias detection and visualizations
    plot_demographic_parity(df)
    plot_disparate_impact(df)
    plot_sentiment_by_task(df)
    plot_satisfaction_by_task(df)
    plot_response_time_by_persona(df)
    plot_response_time_by_task(df)
    plot_satisfaction_distribution(df)

if __name__ == "__main__":
    main()
