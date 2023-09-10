# CoinWizard

![Demo 1](repo/demo-1.png)
![Demo 2](repo/demo-2.png)


Predict the prices of certain cryptocurrencies by training a PyTorch Model on the history data from the NASDAQ API.

> **Important**: This code trains an RNN-Model, so it will take some time to train the model. The model will be saved in the `Backend` directory and will be loaded on the next start of the application.

## Installation

1. Clone the repository
```
git clone https://github.com/fujiwarachoki/CoinWizard.git
```

2. CD into the directory
```
cd CoinWizard
```

4. Continue with `Usage`.

## Usage

1. Get API Key from https://data.nasdaq.com and write it to the file
`key` in the Backend Directory.

2. Run Start Script
```
python3 start.py
```

**This will install all the dependencies and start the frontend as well as the backend of the application.**

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE)

## Authors

- [Sami Hindi](https://www.samihindi.com)

