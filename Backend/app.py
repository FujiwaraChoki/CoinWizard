import csv
import torch
import requests
import torch.nn as nn
import torch.optim as optim
from flask_cors import CORS
from flask import Flask, jsonify
from termcolor import colored

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def get_abbreviation(crypto):
    # Function to get the abbreviation of a
    # cryptocurrency based on its name

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

class PricePredictionModel(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(PricePredictionModel, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Sequential(nn.Linear(prev_size, 32), nn.ReLU())

    def forward(self, x):
        print("Size after hidden layers:", x.size())
        x = self.hidden_layers(x)
        print("Size before output layer:", x.size())
        x = self.output_layer(x)
        return x

    
def load_data(crypto):
    # Function to load data from a CSV file for a specific cryptocurrency
    with open(f'./data/{crypto}.csv', 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return data

model = None
@app.route("/predict/<crypto>", methods=["GET"])
def predict_crypto_price(crypto):
    api_key = open("./key", "r").read().strip()
    crypto_abbreviation = get_abbreviation(crypto)
    url = f'https://data.nasdaq.com/api/v3/datasets/BITFINEX/{crypto_abbreviation}USD/data.csv?api_key={api_key}'
    response = requests.get(url)
    current_price_data = response.text.split('\n')
    csv_reader = csv.reader(current_price_data)
    csv_reader = list(csv_reader)[1:]
    
    current_price_features = []
    for row in csv_reader:
        if row and row[4].replace('.', '', 1).isdigit():
            current_price_features.append(float(row[4]))
    
    current_price_features_tensor = torch.tensor(current_price_features, dtype=torch.float32)
    
    feature_means = torch.mean(current_price_features_tensor)
    feature_stds = torch.std(current_price_features_tensor)
    
    normalized_features = (current_price_features_tensor - feature_means) / feature_stds
    
    # Select the first 6 features from normalized_features
    selected_features = normalized_features[:6]

    # Reshape the selected_features to match the expected input size (1, 6)
    normalized_features_tensor = selected_features.view(1, -1)

    model.eval()
    with torch.no_grad():
        predicted_price = model(normalized_features_tensor)
        predicted_price_value = predicted_price.item()
    
    return jsonify({"predicted_price": predicted_price_value})

@app.route("/train/<crypto>", methods=["POST"])
def train_model(crypto):
    global model
    training_data = load_data(crypto)
    
    # Remove the header row
    training_data = training_data[1:]
    
    # Convert the data to numerical format
    data_array = []
    for row in training_data:
        numerical_row = [float(val) if val.replace('.', '', 1).isdigit() else 0.0 for val in row[1:]]  # Exclude the date column
        data_array.append(numerical_row)
    
    data_array = torch.tensor(data_array, dtype=torch.float32)
    
    # Adjust the input size to match the number of features
    input_size = data_array.shape[1] - 1
    
    feature_means = torch.mean(data_array, dim=0)
    feature_stds = torch.std(data_array, dim=0)
    
    normalized_data = (data_array - feature_means) / feature_stds
    
    # Only include the first 6 features as input (excluding date)
    features = normalized_data[:, :input_size]
    labels = normalized_data[:, -1]  # Assuming the last column is the label
    
    features_tensor = features.clone().detach()
    labels_tensor = labels.clone().detach()
    
    num_epochs = 100
    learning_rate = 0.001
    hidden_sizes = [64, 32]
    
    model = PricePredictionModel(input_size=input_size, hidden_sizes=hidden_sizes)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(features_tensor)
        loss = criterion(predictions, labels_tensor)
        loss.backward()
        optimizer.step()
    
    return jsonify({"message": "Model trained successfully"})


@app.route("/current/<crypto>", methods=["GET"])
def get_current_crypto_price(crypto):
    api_key = open("./key", "r").read().strip()
    crypto_abbreviation = get_abbreviation(crypto)
    url = f'https://data.nasdaq.com/api/v3/datasets/BITFINEX/{crypto_abbreviation}USD/data.csv?api_key={api_key}'
    response = requests.get(url)
    current_price_data = response.text.split('\n')
    csv_reader = csv.reader(current_price_data)
    current_price = None
    for row in csv_reader:
        if row and row[4] != "Last":
            current_price = row[4]
            break
    
    return jsonify({"current_price": current_price})

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection for debugging
    app.run(host="0.0.0.0", port=8080, debug=True)