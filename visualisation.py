import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import os
from personas import personas

# Ensure the visualization directory exists
if not os.path.exists("visualisation"):
    os.makedirs("visualisation")

# Load the sentiment results from the output of sentiment analysis
def load_results(results_file):
    """Loads the processed sentiment analysis data into a DataFrame."""
    df = pd.read_csv(results_file)  # Using read_csv since the file is now saved as CSV in the main code
    return df

# Granular Heatmaps: Break down heatmaps by intersecting factors (e.g., age × race × ADHD severity)
def granular_heatmaps(df):
    """Creates granular heatmaps to display intersecting demographic factors."""
    # Since the 'age_group' column already has labeled categories, we use it directly
    print(f"Unique Age Groups: {df['age_group'].unique()}")

    # Create intersection of factors (age_group × race × ADHD severity)
    intersected_groups = df.groupby(['age_group', 'race', 'adhd_severity']).agg({
        'sentiment_score': ['mean', 'std'],
        'response_length': ['mean', 'std'],
        'response_time_seconds': ['mean', 'std']
    }).reset_index()

    # Pivot tables for heatmaps
    sentiment_heatmap = df.pivot_table(index='race', columns='age_group', values='sentiment_score', aggfunc='mean')
    response_length_heatmap = df.pivot_table(index='race', columns='age_group', values='response_length', aggfunc='mean')
    response_time_heatmap = df.pivot_table(index='race', columns='age_group', values='response_time_seconds', aggfunc='mean')

    # Plot heatmaps with ADHD severity included
    plt.figure(figsize=(12, 6))
    sns.heatmap(sentiment_heatmap, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Sentiment Score by Age Group and Race")
    plt.savefig("visualisation/sentiment_heatmap.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.heatmap(response_length_heatmap, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Response Length by Age Group and Race")
    plt.savefig("visualisation/response_length_heatmap.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.heatmap(response_time_heatmap, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Response Time by Age Group and Race")
    plt.savefig("visualisation/response_time_heatmap.png")
    plt.show()

# Demographic Parity Analysis: Check if response length, response time, and sentiment scores differ significantly across different demographic characteristics
def demographic_parity_analysis(df):
    """Analyze biases across demographic groups for response length, response time, and sentiment score."""
    demographic_columns = ['age_group', 'gender', 'race', 'adhd_severity']
    
    # Perform group-by for demographics
    demographic_groups = df.groupby(demographic_columns).agg({
        'sentiment_score': ['mean', 'std'],
        'response_length': ['mean', 'std'],
        'response_time_seconds': ['mean', 'std']
    }).reset_index()

    print("Demographic Parity Analysis:\n", demographic_groups)

    # Perform ANOVA to check for significant differences across groups for each feature
    for feature in ['sentiment_score', 'response_length', 'response_time_seconds']:
        anova_result = stats.f_oneway(*[group[1][feature] for group in df.groupby(demographic_columns)] )
        print(f"ANOVA result for {feature} -> F-statistic: {anova_result.statistic}, p-value: {anova_result.pvalue}")
        if anova_result.pvalue < 0.05:
            print(f"Significant differences found for {feature}.")
        else:
            print(f"No significant differences for {feature}.")

# Visualizations for biases across demographics
def visualize_demographics(df):
    """Visualize demographic distributions and biases across response length, response time, and sentiment score."""
    # Sentiment score distribution by gender, race, ADHD severity, and age range
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='gender', y='sentiment_score', data=df)
    plt.title('Sentiment Score Distribution by Gender')
    plt.savefig("visualisation/sentiment_gender_boxplot.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='race', y='sentiment_score', data=df)
    plt.title('Sentiment Score Distribution by Race')
    plt.savefig("visualisation/sentiment_race_boxplot.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='age_group', y='sentiment_score', data=df)
    plt.title('Sentiment Score Distribution by Age Group')
    plt.savefig("visualisation/sentiment_age_group_boxplot.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='adhd_severity', y='sentiment_score', data=df)
    plt.title('Sentiment Score Distribution by ADHD Severity')
    plt.savefig("visualisation/sentiment_adhd_severity_boxplot.png")
    plt.show()

    # Response length distribution by gender, race, ADHD severity, and age range
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='gender', y='response_length', data=df)
    plt.title('Response Length Distribution by Gender')
    plt.savefig("visualisation/response_length_gender_boxplot.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='race', y='response_length', data=df)
    plt.title('Response Length Distribution by Race')
    plt.savefig("visualisation/response_length_race_boxplot.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='age_group', y='response_length', data=df)
    plt.title('Response Length Distribution by Age Group')
    plt.savefig("visualisation/response_length_age_group_boxplot.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='adhd_severity', y='response_length', data=df)
    plt.title('Response Length Distribution by ADHD Severity')
    plt.savefig("visualisation/response_length_adhd_severity_boxplot.png")
    plt.show()

    # Response time distribution by gender, race, ADHD severity, and age range
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='gender', y='response_time_seconds', data=df)
    plt.title('Response Time Distribution by Gender')
    plt.savefig("visualisation/response_time_gender_boxplot.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='race', y='response_time_seconds', data=df)
    plt.title('Response Time Distribution by Race')
    plt.savefig("visualisation/response_time_race_boxplot.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='age_group', y='response_time_seconds', data=df)
    plt.title('Response Time Distribution by Age Group')
    plt.savefig("visualisation/response_time_age_group_boxplot.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='adhd_severity', y='response_time_seconds', data=df)
    plt.title('Response Time Distribution by ADHD Severity')
    plt.savefig("visualisation/response_time_adhd_severity_boxplot.png")
    plt.show()

# Main Function to run all analyses
def main():
    results_file = "real_responses_with_sentiment.csv"  # Example file name from the main code
    df = load_results(results_file)

    # Perform Demographic Parity Analysis to check if sentiment differs across demographic characteristics
    demographic_parity_analysis(df)

    # Perform Granular Heatmaps for demographic intersections (age × race × ADHD severity)
    granular_heatmaps(df)

    # Visualize the biases across different demographics
    visualize_demographics(df)

if __name__ == "__main__":
    main()
