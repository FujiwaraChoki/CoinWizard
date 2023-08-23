import requests

url = 'http://localhost:8080/current/BTC'
response = requests.get(url)
print(response.json())

url = 'http://localhost:8080/predict/BTC'
response = requests.get(url)
print(response.json())
