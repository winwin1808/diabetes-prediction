import requests
import json


url = 'http://127.0.0.1:5000/predict'


input_data = {
    'input': [10,	122,	78,	31,	0,	27.6,	0.512,	45
]  # Example input
}


response = requests.post(url, json=input_data)
print("Response from server:", response.text)
