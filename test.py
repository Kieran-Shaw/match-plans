import requests
import random
import string

def generate_mock_data():
    target = {
        "name": "".join(random.choices(string.ascii_letters, k=10)),
        "carrier": "".join(random.choices(string.ascii_letters, k=10)),
    }

    num_plans = random.randint(5, 10)
    plans = []
    for _ in range(num_plans):
        plan = {
            "id": "".join(random.choices(string.ascii_letters + string.digits, k=8)),
            "name": "".join(random.choices(string.ascii_letters, k=10)),
            "carrier": "".join(random.choices(string.ascii_letters, k=10)),
        }
        plans.append(plan)

    return {"target": target, "plans": plans}

# Start your Flask app (e.g., `python your_app.py`)

# Generate mock data
mock_data = generate_mock_data()

# Send a POST request to your local API
response = requests.post('http://localhost:5001/predict', json=mock_data)

# Check the response
if response.status_code == 200:
    predictions = response.json()['predictions']
    print(f"Top 5 predicted plans: {predictions}")
else:
    print(f"Error: {response.text}")