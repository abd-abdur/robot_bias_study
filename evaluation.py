import json
import pandas as pd
import time
from textblob import TextBlob
from datetime import datetime

# Load the interaction test results from JSON file
def load_results(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results

# Perform sentiment analysis using TextBlob
def analyze_sentiment(response_text):
    # TextBlob returns polarity: -1 (negative) to 1 (positive)
    blob = TextBlob(response_text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Calculate response length
def calculate_response_length(response_text):
    return len(response_text.split())

# Function to analyze sentiment for all responses and collect required metrics
def perform_sentiment_analysis(results_file):
    # Load the results of the robot's responses
    results = load_results(results_file)
    
    # List to store processed results with sentiment scores, length, and time
    processed_results = []
    
    # Track the time of the task processing (for response time estimation)
    start_time = time.time()

    # Analyze sentiment for each response and collect the required metrics
    for result in results:
        response_text = result['robot_response']
        
        # Sentiment score
        sentiment_score = analyze_sentiment(response_text)
        
        # Response length
        response_length = calculate_response_length(response_text)
        
        # Calculate response time (simulating response time for now, can be adjusted)
        response_time = time.time() - start_time  # Response time for this task
        
        # Satisfaction score: mapped from sentiment score
        # This could be a more complex formula based on your system
        satisfaction_score = "High" if sentiment_score > 0.1 else ("Neutral" if sentiment_score > -0.1 else "Low")

        # Add all collected metrics to the result
        processed_results.append({
            'persona': result['persona'],
            'task': result['task'],
            'sentiment_score': sentiment_score,
            'satisfaction_score': satisfaction_score,
            'response_length': response_length,
            'response_time': response_time
        })

    # Save the processed results to a CSV file for further analysis
    output_filename = results_file.replace(".json", "_with_sentiment.csv")
    df = pd.DataFrame(processed_results)

    # Save DataFrame to CSV
    df.to_csv(output_filename, index=False)

    print(f"Sentiment analysis and data collection complete. Results saved to {output_filename}")
    
    # Return the processed results for further analysis
    return processed_results

# Main function to run sentiment analysis and save to CSV
def main():
    results_file = "test_results_2025-04-03_10-07-02.json"  # Specify your results file here
    results_with_sentiment = perform_sentiment_analysis(results_file)

    # You can now further analyze the CSV file generated or visualize it

if __name__ == "__main__":
    main()
