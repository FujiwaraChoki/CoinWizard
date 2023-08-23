from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import requests
import json

app = Flask(__name__)


class PricePredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PricePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Adjust the shape of the weight matrix in the first linear layer
model = PricePredictionModel(input_size=7, hidden_size=16)
model.fc1.weight.data = model.fc1.weight.data.t()  # Transpose the weights
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load and preprocess the training data


def preprocess_data(data):
    data_array = np.array(data)

    # Loop through the array and replace None values with 0.0
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            if data_array[i, j] is None:
                data_array[i, j] = 0.0

    # Convert the entire array to float type
    data_array = data_array.astype(float)

    feature_means = np.mean(data_array, axis=0)
    centered_data = data_array - feature_means
    return centered_data, feature_means


def load_data(crypto):
    with open(f'./data/{crypto}.json', 'r') as f:
        data = json.load(f)
    return data["dataset_data"]["data"]


@app.route("/predict/<crypto>", methods=["GET"])
def predict_crypto_price(crypto):
    # Load current crypto price from the Nasdaq API
    api_key = open("./key", "r").read().strip()  # Read the API key
    crypto_abbreviation = get_abbreviation(crypto)
    url = f'https://data.nasdaq.com/api/v3/datasets/BITFINEX/{crypto_abbreviation}USD/data.json?api_key={api_key}'
    response = requests.get(url)
    current_price_data = response.json()
    current_price_features = [
        float(item) if item is not None else 0.0 for item in current_price_data["dataset_data"]["data"][-1][2:]
    ]
    current_price_tensor = torch.tensor(
        current_price_features, dtype=torch.float32)

    # Load and preprocess the training data
    # Load and preprocess the training data
    training_data = load_data(crypto)

    # Exclude the date column before preprocessing
    training_data = [row[1:] for row in training_data]
    preprocessed_data, feature_means = preprocess_data(training_data)
    preprocessed_tensor = torch.tensor(preprocessed_data, dtype=torch.float32)

    # Training loop
    # Training loop
    model.train()  # Set the model to training mode
    num_epochs = 1000  # Adjust as needed
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(preprocessed_tensor)

        # Adjust the shape of the target tensor to match the output shape
        # Assuming last column is the target (price)
        target_tensor = preprocessed_tensor[:, -1].unsqueeze(1)

        loss = criterion(outputs, target_tensor)
        loss.backward()
        optimizer.step()

    # Make a prediction using the trained model
    model.eval()  # Set the model to evaluation mode

    # Ensure the current_price_tensor has the correct shape (1, 7)
    current_price_tensor = torch.tensor(
        [current_price_features], dtype=torch.float32)  # Wrap in a list
    # Prediction
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
    api_key = open("./key", "r").read().strip()  # Read the API key
    crypto_abbreviation = get_abbreviation(crypto)
    url = f'https://data.nasdaq.com/api/v3/datasets/BITFINEX/{crypto_abbreviation}USD/data.json?api_key={api_key}'
    response = requests.get(url)
    current_price_data = response.json()
    current_price = current_price_data["dataset_data"]["data"][0][4]
    return jsonify({"current_price": current_price})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
