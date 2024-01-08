import requests
import json

# URL of the Flask app
url = 'http://127.0.0.1:5000/predict'

# Example input data (replace with your actual data)
input_data = {
    'input': [10,	122,	78,	31,	0,	27.6,	0.512,	45
]  # Example input
}

# Send a POST request
response = requests.post(url, json=input_data)
print("Response from server:", response.text)
