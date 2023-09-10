import csv
import torch
import requests
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from flask_cors import CORS
from flask import Flask, jsonify
from termcolor import colored

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def get_abbreviation(crypto):
    # Function to get the abbreviation of a cryptocurrency based on its name
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

# Modify your PricePredictionModel class to load the weights
class PricePredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(PricePredictionModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        out = self.fc(out[:, -1, :])
        return out

# Create and initialize the model
input_size = 1
hidden_size = 100
output_size = 1
model = PricePredictionModel(input_size, hidden_size, output_size)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define a custom dataset class for your time series data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        sequence = self.data[idx : idx + self.sequence_length]
        input_sequence = sequence[:-1]
        target = sequence[-1:]
        return torch.tensor(input_sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def get_image_for_crypto(crypto):
    # Function to get the image for a specific cryptocurrency
    if 'btc' in crypto.lower():
        return 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/1200px-Bitcoin.svg.png'
    elif 'eth' in crypto.lower():
        return 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/1200px-Ethereum_logo_2014.svg.png'
    elif 'ltc' in crypto.lower():
        return 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Litecoin.svg/1200px-Litecoin.svg.png'
    elif 'xrp' in crypto.lower():
        return 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Ripple_logo.svg/1200px-Ripple_logo.svg.png'
    elif 'bch' in crypto.lower():
        return 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/BCH_Logo.svg/1200px-BCH_Logo.svg.png'
    else:
        return 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/1200px-Bitcoin.svg.png'

mean = None
std = None

def train_model(crypto_to_train):
    api_key = open("./key", "r").read().strip()
    crypto_abbreviation = get_abbreviation(crypto_to_train)
    url = f'https://data.nasdaq.com/api/v3/datasets/BITFINEX/{crypto_abbreviation}USD/data.csv?api_key={api_key}'
    response = requests.get(url)
    current_price_data = response.text.split('\n')
    csv_reader = csv.reader(current_price_data)
    csv_reader = list(csv_reader)[1:]

    prices = []
    for row in csv_reader:
        if row and row[4] != "Last":
            prices.append(float(row[4]))

    sequence_length = 100
    dataset = TimeSeriesDataset(prices, sequence_length)
    your_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop
    num_epochs = 100  # Adjust as needed
    for epoch in range(num_epochs):
        for inputs, targets in your_data_loader:
            optimizer.zero_grad()
            h = torch.zeros(1, inputs.size(0), hidden_size)  # Initialize the hidden state
            #outputs = model(inputs.unsqueeze(2), h)
            outputs = model(inputs.unsqueeze(2), h)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Save the model weights after training
    torch.save(model.state_dict(), 'model_weights.pth')

@app.route("/predict/<crypto>", methods=["GET"])
def predict_crypto_price(crypto):
    train_model(get_abbreviation(crypto))

    # Load the saved model weights (Move this above training if the weights exist)
    model_weights_path = 'model_weights.pth'
    if torch.cuda.is_available():
        PricePredictionModel.load_state_dict(torch.load(model_weights_path))
    else:
        PricePredictionModel.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    PricePredictionModel.eval()  # Set the model to evaluation mode

    global mean, std

    api_key = open("./key", "r").read().strip()
    crypto_abbreviation = get_abbreviation(crypto)
    url = f'https://data.nasdaq.com/api/v3/datasets/BITFINEX/{crypto_abbreviation}USD/data.csv?api_key={api_key}'
    response = requests.get(url)
    current_price_data = response.text.split('\n')
    csv_reader = csv.reader(current_price_data)
    csv_reader = list(csv_reader)[1:]

    prices = []
    for row in csv_reader:
        if row and row[4] != "Last":
            prices.append(float(row[4]))

    mean = sum(prices) / len(prices)
    std = (sum((x - mean) ** 2 for x in prices) / len(prices)) ** 0.5

    print(f"Loaded {len(prices)} prices for {crypto_abbreviation} from CSV file")

    last_observed_sequence = prices[-100:]

    # Normalize the input sequence using the loaded mean and std
    input_sequence = [(price - mean) / std for price in last_observed_sequence]
    predicted_sequence = []

    with torch.no_grad():
        h = torch.zeros(1, 1, hidden_size)  # Initialize the hidden state
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        output = PricePredictionModel(input_tensor, h)
        # Scale the predicted value back to the original range
        predicted_value = (output.squeeze().item() * std) + mean
        predicted_sequence.append(predicted_value)
        input_sequence = input_sequence[1:] + [predicted_value]

    # Optional: Print the predicted sequence to the console with colored output
    print(colored(f"Predicted sequence: {predicted_sequence}", "green"))

    return jsonify({"predicted_price": predicted_sequence[0], "image": get_image_for_crypto(crypto_abbreviation)})

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
    torch.autograd.set_detect_anomaly(True)

    # Start the Flask app
    app.run(host="0.0.0.0", port=8080, debug=True)