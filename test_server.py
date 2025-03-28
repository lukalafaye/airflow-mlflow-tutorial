import requests
import json
import numpy as np

data = {
    "inputs": [
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 2.8, 4.8, 1.8]
    ]
}

response = requests.post(
    url="http://localhost:1234/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)
)

print("Status code:", response.status_code)
print("Predictions:", response.json())
