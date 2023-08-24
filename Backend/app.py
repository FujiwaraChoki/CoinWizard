from flask import Flask, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
from flask_cors import CORS
import numpy as np
import requests
import csv

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


class PricePredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PricePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def preprocess_data(data):
    data_array = np.array(data)
    data_array = np.nan_to_num(data_array, nan=0.0)  # Replace NaN with 0.0
    data_array = data_array.astype(np.float32)

    feature_means = np.mean(data_array, axis=0)
    centered_data = data_array - feature_means
    return centered_data, feature_means


def load_data(crypto):
    with open(f'./data/{crypto}.csv', 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return data


def get_image_for_crypto(crypto):
    if 'bitcoin' in crypto.lower():
        return 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/2048px-Bitcoin.svg.png'
    elif 'ethereum' in crypto.lower():
        return 'https://www.pngall.com/wp-content/uploads/10/Ethereum-Logo-PNG-Pic.png'
    elif 'litecoin' in crypto.lower():
        return 'https://cryptologos.cc/logos/litecoin-ltc-logo.png'
    elif 'ripple' in crypto.lower():
        return 'https://o.remove.bg/downloads/183f79fa-8a44-4d31-b254-a61869f8b912/xrp-logo-removebg-preview.png'
    elif 'bitcoin cash' in crypto.lower():
        return 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Bitcoin_Cash.png/600px-Bitcoin_Cash.png?20210403103340'
    else:
        return 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/2048px-Bitcoin.svg.png'


@app.route("/predict/<crypto>", methods=["GET"])
def predict_crypto_price(crypto):
    api_key = open("./key", "r").read().strip()
    crypto_abbreviation = get_abbreviation(crypto)
    url = f'https://data.nasdaq.com/api/v3/datasets/BITFINEX/{crypto_abbreviation}USD/data.csv?api_key={api_key}'
    response = requests.get(url)
    current_price_data = response.text.split('\n')
    csv_reader = csv.reader(current_price_data)
    current_price_features = [row[4] for row in csv_reader if row]

    print("Current price features: ", current_price_features)
    current_price_features = [0.0 if item == 'Last' else float(item)
                              for item in current_price_features if item.replace('.', '', 1).isdigit()]

    model.eval()
    current_price_tensor = torch.tensor(
        current_price_features, dtype=torch.float32)
    current_price_tensor = current_price_tensor.unsqueeze(
        0)  # Add batch dimension

    predicted_price = model(current_price_tensor)
    predicted_price_value = predicted_price.item()

    return jsonify({"predicted_price": predicted_price_value})


def get_abbreviation(crypto):
    if 'bitcoin' in crypto.lower():
        return 'BTC'
    elif 'ethereum' in crypto.lower():
        return 'ETH'
    elif 'litecoin' in crypto.lower():
        return 'LTC'
    elif 'ripple' in crypto.lower():
        return 'XRP'
    elif 'bitcoin cash' in crypto.lower():
        return 'BCH'
    else:
        return crypto


@app.route("/current/<crypto>", methods=["GET"])
def get_current_crypto_price(crypto):
    api_key = open("./key", "r").read().strip()
    crypto_abbreviation = get_abbreviation(crypto)
    url = f'https://data.nasdaq.com/api/v3/datasets/BITFINEX/{crypto_abbreviation}USD/data.csv?api_key={api_key}'
    response = requests.get(url)
    with open(f'./data/{crypto}.csv', 'w') as f:
        f.write(response.text)
    current_price_data = response.text.split('\n')
    current_price = current_price_data[1].split(',')[4]
    print(current_price)
    return jsonify({"current_price": current_price, "image": get_image_for_crypto(crypto)})


if __name__ == "__main__":
    model = PricePredictionModel(input_size=3370, hidden_size=16)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    app.run(host="0.0.0.0", port=8080, debug=True)
