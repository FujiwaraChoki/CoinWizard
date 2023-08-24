document.addEventListener('DOMContentLoaded', () => {
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

    startButton.addEventListener('click', (e) => {
        e.preventDefault();

        startButton.classList.add('hidden');
        cryptoSelector.classList.remove('hidden');
    });

    const predictPrice = async (crypto) => {
        const response = await fetch(`http://localhost:8080/predict/${crypto}`);
        const data = await response.json();
        return data.predicted_price;
    };

    const getCurrentPrice = async (crypto) => {
        const response = await fetch(`http://localhost:8080/current/${crypto.trim()}`);
        const data = await response.json();
        return {
            currentPrice: data.current_price,
            cryptoImage: data.image
        };
    };


    cryptoSearchInput.addEventListener('input', (e) => {
        e.preventDefault();
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

            item.addEventListener('click', async (e) => {
                e.preventDefault();

                const crypto = item.textContent;
                const { currentPrice, cryptoImage } = await getCurrentPrice(crypto);  // Await the result here
                const predictedPrice = await predictPrice(crypto); // Await the prediction too
                resultCryptoName.textContent = crypto;
                console.log(currentPrice);

                cryptoSelector.classList.add('hidden');
                result.classList.remove('hidden');
                resultCryptoImage.src = cryptoImage;
                resultActualCryptoPrice.innerText = `$${currentPrice}`;
                resultPredictedCryptoPrice.innerText = `$${predictPrice}`;

                return false;
            });
        });

        return false;
    });
});