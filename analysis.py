import pandas as pd

df = pd.read_csv('data/interactions.csv')

# Sentiment analysis on responses
df['satisfaction_score'] = df['response'].apply(analyze_sentiment)

# You can now perform demographic parity analysis and bias detection
# Example: Compare satisfaction scores across different demographics (age, race, ADHD severity)
demographic_analysis = df.groupby(['persona', 'adhd_severity'])['satisfaction_score'].mean()
print(demographic_analysis)
