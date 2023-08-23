const startButton = document.getElementById('start-button');
const cryptoSelector = document.getElementById('crypto-container');;
const cryptoSearchInput = document.getElementById('crypto-search');
const cryptoItems = document.querySelectorAll('.crypto-item');
const cryptoList = document.getElementById('crypto-list');
const result = document.getElementById('result');
const resultCryptoName = document.getElementById('result-crypto-name');
const resultCryptoImage = document.getElementById('result-crypto-image');
const resultActualCryptoPrice = document.getElementById('result-actual-crypto-price');
const resultPredictedCryptoPrice = document.getElementById('result-predicted-crypto-price');


startButton.addEventListener('click', () => {
    startButton.classList.add('hidden');
    cryptoSelector.classList.remove('hidden');
});

const predictPrice = (crypto) => {
    const response = fetch(`http://localhost:8080/predict/${crypto}`);
    const data = response.json();
    return data;
};

const getCurrentPrice = (crypto) => {
    const response = fetch(`http://localhost:8080/current/${crypto}`);
    const data = response.json();
    return data;
};


cryptoSearchInput.addEventListener('input', () => {
    const searchQuery = cryptoSearchInput.value.toLowerCase();

    if (searchQuery === '') {
        cryptoList.classList.add('hidden');
        return;
    }

    cryptoList.classList.remove('hidden');
    cryptoItems.forEach(item => {
        const itemName = item.textContent.toLowerCase();
        if (itemName.includes(searchQuery)) {
            item.style.display = 'block';
        } else {
            item.style.display = 'none';
        }

        item.addEventListener('click', () => {
            const crypto = item.textContent;
            const currentPrice = getCurrentPrice(crypto);
            //const { predictedPrice, cryptoImage } = predictPrice(crypto);
            resultCryptoName.textContent = crypto;

            //console.log(`${crypto}: Price: ${currentPrice}, Predicted Price: ${predictedPrice}`);
            console.log(`${crypto}: Price: ${currentPrice}`);
            cryptoSelector.classList.add('hidden');
            result.classList.remove('hidden');
            resultCryptoImage.src = cryptoImage;
            resultActualCryptoPrice.innerText = currentPrice;
            //resultPredictedCryptoPrice.innerText = predictedPrice;
        });
    });
});