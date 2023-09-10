import requests
from termcolor import colored

# Change this URL to match the address where your Flask app is running
base_url = "http://127.0.0.1:8080"

# Test current price endpoint
current_price_url = f"{base_url}/current/bitcoin"
current_price_response = requests.get(current_price_url)
current_price_data = current_price_response.json()
current_price = current_price_data.get("current_price")
print(colored("Current Bitcoin Price:", "green"), current_price)

# Test prediction endpoint
prediction_url = f"{base_url}/predict/bitcoin"
prediction_response = requests.get(prediction_url)
prediction_data = prediction_response.json()
predicted_price = prediction_data.get("predicted_price")
print(colored("Predicted Bitcoin Price:", "green"), predicted_price)
