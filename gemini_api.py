from google import genai
import pandas as pd
import json
from datetime import datetime

# Import personas and tasks from separate files
from personas import personas
from tasks import tasks

# Initialize Gemini API client
API_KEY = "AIzaSyBU2x31lb-W1omCb2-Efra9VEQH8i5QtWA" 
client = genai.Client(api_key=API_KEY)

# Function to simulate task response from the robot
def get_robot_response(persona_name, task):
    # Retrieve persona data from the personas dictionary
    persona = personas[persona_name]
    
    # Construct the prompt with the persona's attributes and the task
    prompt = (
        f"Persona: {persona_name}\n"
        f"Age: {persona['age']}, Gender: {persona['gender']}, Race: {persona['race']}, "
        f"ADHD Severity: {persona['adhd_severity']}, Comorbidities: {', '.join(persona['comorbidities']) if persona['comorbidities'] else 'None'}\n"
        f"Profession: {persona['profession']}\n"
        f"Challenges: {', '.join(persona['challenges'])}\n"
        f"Needs: {', '.join(persona['needs'])}\n"
        f"Symptoms: {', '.join(persona['symptoms'])}\n"
        f"Task: {task}\n\n"
        "Respond to the task in a helpful, personalized way for the persona."
    )

    # Generate a response using Gemini API
    response_stream = client.models.generate_content_stream(
        model="gemini-2.0-flash",  # You can replace this with the specific model you are using
        contents=[prompt]
    )

    response_text = ""
    for chunk in response_stream:
        if chunk.text:
            response_text += chunk.text 

    return response_text.strip()

# Function to collect sentiment score (you can integrate sentiment analysis here if needed)
def collect_sentiment(response_text):
    # For simplicity, we'll mock a sentiment score (between -1 and 1) for now
    import random
    return random.uniform(-1, 1)

# Main function for running all tests
def run_interaction_test():
    results = []

    for persona_name in personas.keys():
        for task in tasks:
            print(f"Testing {persona_name} with task: {task}")
            robot_response = get_robot_response(persona_name, task)
            sentiment_score = collect_sentiment(robot_response)

            # Log the results
            results.append({
                'persona': persona_name,
                'task': task,
                'robot_response': robot_response,
                'sentiment_score': sentiment_score
            })

    # Save results to a JSON file for further analysis
    output_filename = f"test_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_filename}")

# Running the test
if __name__ == "__main__":
    run_interaction_test()
