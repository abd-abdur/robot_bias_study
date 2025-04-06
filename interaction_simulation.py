#interaction_simulation.py
from gemini_api import get_robot_response
from personas import personas
from tasks import tasks

interaction_logs = []

for persona_name, persona in personas.items():
    for task in tasks:
        response = get_robot_response(persona, task)
        interaction_logs.append({
            "persona": persona_name,
            "task": task,
            "response": response
        })

# Save interaction logs for further analysis
import pandas as pd
df = pd.DataFrame(interaction_logs)
df.to_csv('data/interactions.csv', index=False)
